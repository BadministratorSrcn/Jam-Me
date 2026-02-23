#!/usr/bin/env python3
"""
=============================================================
VISUAL GPS — Tile-Based Navigation for Large Areas (50+ km²)
=============================================================

Manages large coverage areas using a tile grid system:
  - Splits area into 500×500m tiles
  - Loads only nearby tiles based on drone position
  - Memory: ~200MB (3×3 neighborhood cached)
  - 50 km² = ~200 tiles, but only 9 in RAM at a time

Preparation:
    python3 prepare_tiles.py --top-left 39.95,32.80 \
        --bottom-right 39.90,32.87 --tile-size 500

    Place satellite images in tiles/ directory, then:
    python3 main_tile.py
=============================================================
"""

import cv2
import numpy as np
import json
import os
import time
import threading
import serial
import math
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
log = logging.getLogger("TileNav")


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class Config:
    fc_port: str = "/dev/ttyAMA0"
    gps_port: str = "/dev/ttyUSB0"
    fc_baud: int = 921600
    gps_baud: int = 9600

    cam_id: int = 0
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30

    tile_dir: str = "tiles"
    tile_config: str = "tile_config.json"

    feature_detector: str = "ORB"
    min_match_count: int = 15
    match_ratio: float = 0.75
    ransac_threshold: float = 5.0

    nmea_rate_hz: int = 5
    fix_quality: int = 1
    num_satellites: int = 12
    hdop: float = 0.9

    max_cached_tiles: int = 9           # Max tiles in RAM (3×3 grid)
    tile_preload_radius: int = 1        # Preload radius around current tile
    optical_flow_fallback: bool = True   # Use optical flow when matching fails

    @classmethod
    def load(cls, path="system_config.json"):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        return cls()


# ============================================================
# TILE CACHE — INTELLIGENT TILE MANAGEMENT
# ============================================================
class TileCache:
    """
    LRU cache for tile management.
    Only tiles near the drone are kept in RAM.
    """

    @dataclass
    class CachedTile:
        row: int
        col: int
        image: np.ndarray
        gray: np.ndarray
        keypoints: list
        descriptors: np.ndarray
        corners_latlon: np.ndarray
        corners_pixel: np.ndarray
        H_pixel_to_geo: np.ndarray
        center_lat: float = 0.0
        center_lon: float = 0.0
        feature_count: int = 0

    def __init__(self, config: Config):
        self.config = config
        self.cache: OrderedDict[str, TileCache.CachedTile] = OrderedDict()
        self.tile_meta: Dict[str, dict] = {}
        self.grid_rows = 0
        self.grid_cols = 0
        self.detector = self._create_detector()
        self.lock = threading.Lock()
        self._load_tile_config()

    def _create_detector(self):
        if self.config.feature_detector == "SIFT":
            return cv2.SIFT_create(nfeatures=2000)
        elif self.config.feature_detector == "AKAZE":
            return cv2.AKAZE_create()
        return cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)

    def _load_tile_config(self):
        """Load tile_config.json (metadata only, not images)"""
        cfg_path = os.path.join(self.config.tile_dir, self.config.tile_config)
        if not os.path.exists(cfg_path):
            log.error(f"Tile config not found: {cfg_path}")
            log.info("Run prepare_tiles.py first!")
            return

        with open(cfg_path) as f:
            data = json.load(f)

        self.grid_rows = data["grid"]["rows"]
        self.grid_cols = data["grid"]["cols"]

        for t in data["tiles"]:
            key = f"{t['row']}_{t['col']}"
            c = t["corners"]
            center_lat = (c["top_left"]["lat"] + c["bottom_left"]["lat"]) / 2
            center_lon = (c["top_left"]["lon"] + c["top_right"]["lon"]) / 2
            self.tile_meta[key] = {
                "row": t["row"], "col": t["col"],
                "image": t["image"], "corners": c,
                "center_lat": center_lat, "center_lon": center_lon
            }

        log.info(f"Tile grid: {self.grid_rows}x{self.grid_cols} = {len(self.tile_meta)} tiles")

    def find_tile_for_position(self, lat: float, lon: float) -> Optional[str]:
        """Find which tile contains a GPS position"""
        for key, meta in self.tile_meta.items():
            c = meta["corners"]
            if (c["bottom_left"]["lat"] <= lat <= c["top_left"]["lat"] and
                c["top_left"]["lon"] <= lon <= c["top_right"]["lon"]):
                return key
        return None

    def find_nearest_tile(self, lat: float, lon: float) -> Optional[str]:
        """Find the nearest tile to a GPS position"""
        best_key, best_dist = None, float('inf')
        for key, meta in self.tile_meta.items():
            d = (meta["center_lat"] - lat)**2 + (meta["center_lon"] - lon)**2
            if d < best_dist:
                best_dist, best_key = d, key
        return best_key

    def get_neighbors(self, row: int, col: int, radius: int = 1) -> List[str]:
        """Get tile keys for a neighborhood"""
        neighbors = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    key = f"{r}_{c}"
                    if key in self.tile_meta:
                        neighbors.append(key)
        return neighbors

    def load_tile(self, key: str) -> Optional['TileCache.CachedTile']:
        """Load a single tile (returns from cache if available)"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

        meta = self.tile_meta.get(key)
        if meta is None:
            return None

        img_path = os.path.join(self.config.tile_dir, meta["image"])
        if not os.path.exists(img_path):
            return None

        img = cv2.imread(img_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)

        if des is None or len(kp) < 50:
            log.warning(f"Insufficient features: {key} ({len(kp) if kp else 0})")
            return None

        c = meta["corners"]
        corners_latlon = np.array([
            [c["top_left"]["lat"],     c["top_left"]["lon"]],
            [c["top_right"]["lat"],    c["top_right"]["lon"]],
            [c["bottom_right"]["lat"], c["bottom_right"]["lon"]],
            [c["bottom_left"]["lat"],  c["bottom_left"]["lon"]]
        ], dtype=np.float64)

        h, w = gray.shape
        corners_pixel = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float64)
        H_p2g, _ = cv2.findHomography(corners_pixel, corners_latlon)

        tile = self.CachedTile(
            row=meta["row"], col=meta["col"],
            image=img, gray=gray, keypoints=kp, descriptors=des,
            corners_latlon=corners_latlon, corners_pixel=corners_pixel,
            H_pixel_to_geo=H_p2g,
            center_lat=meta["center_lat"], center_lon=meta["center_lon"],
            feature_count=len(kp)
        )

        with self.lock:
            self.cache[key] = tile
            self.cache.move_to_end(key)
            while len(self.cache) > self.config.max_cached_tiles:
                evicted_key, _ = self.cache.popitem(last=False)
                log.debug(f"Evicted from cache: {evicted_key}")

        log.info(f"Tile loaded: {key} ({len(kp)} features) "
                 f"[cache: {len(self.cache)}/{self.config.max_cached_tiles}]")
        return tile

    def ensure_neighborhood(self, row: int, col: int):
        """Preload neighboring tiles in background threads"""
        keys = self.get_neighbors(row, col, self.config.tile_preload_radius)
        for key in keys:
            if key not in self.cache:
                threading.Thread(target=self.load_tile, args=(key,), daemon=True).start()

    def get_active_tiles(self) -> List['TileCache.CachedTile']:
        """Return all currently cached tiles"""
        with self.lock:
            return list(self.cache.values())


# ============================================================
# TILE-BASED VISUAL LOCALIZER
# ============================================================
class TileLocalizer:

    @dataclass
    class MatchResult:
        found: bool = False
        lat: float = 0.0
        lon: float = 0.0
        heading: float = 0.0
        confidence: float = 0.0
        match_count: int = 0
        tile_key: str = ""
        timestamp: float = 0.0

    def __init__(self, tile_cache: TileCache, config: Config):
        self.cache = tile_cache
        self.config = config
        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING if config.feature_detector != "SIFT" else cv2.NORM_L2
        )
        self.detector = tile_cache.detector
        self.last_result: Optional[self.MatchResult] = None
        self.current_tile_key: Optional[str] = None
        self.prev_gray = None
        self.of_position = np.zeros(2)

    def match_frame(self, frame: np.ndarray) -> 'TileLocalizer.MatchResult':
        """Match frame against active tiles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) < 10:
            return self._optical_flow_fallback(gray)

        search_tiles = self._get_search_order()
        best = self.MatchResult()
        best_inliers = 0

        for tile in search_tiles:
            result = self._match_tile(kp_frame, des_frame, gray.shape, tile)
            if result.found and result.match_count > best_inliers:
                best_inliers = result.match_count
                best = result

        if best.found:
            self.last_result = best
            new_key = best.tile_key
            if new_key != self.current_tile_key:
                self.current_tile_key = new_key
                row, col = map(int, new_key.split('_'))
                self.cache.ensure_neighborhood(row, col)
                log.info(f"Active tile changed → [{row},{col}]")
            self.prev_gray = gray
            self.of_position = np.array([best.lat, best.lon])
        elif self.config.optical_flow_fallback:
            best = self._optical_flow_fallback(gray)

        return best

    def _get_search_order(self) -> List[TileCache.CachedTile]:
        """Optimize search: current tile first"""
        tiles = self.cache.get_active_tiles()
        if self.current_tile_key:
            tiles.sort(key=lambda t: 0 if f"{t.row}_{t.col}" == self.current_tile_key else 1)
        return tiles

    def _match_tile(self, kp_frame, des_frame, frame_shape,
                    tile: TileCache.CachedTile) -> 'MatchResult':
        """Match against a single tile"""
        result = self.MatchResult()

        try:
            matches = self.matcher.knnMatch(des_frame, tile.descriptors, k=2)
        except cv2.error:
            return result

        good = [m for m_pair in matches if len(m_pair) == 2
                for m in [m_pair[0]] if m.distance < self.config.match_ratio * m_pair[1].distance]

        if len(good) < self.config.min_match_count:
            return result

        pts_frame = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_tile = np.float32([tile.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_frame, pts_tile, cv2.RANSAC, self.config.ransac_threshold)
        if H is None:
            return result

        inliers = int(mask.sum())
        if inliers < self.config.min_match_count:
            return result

        fh, fw = frame_shape[:2]
        center = np.array([[[fw/2, fh/2]]], dtype=np.float32)
        center_on_tile = cv2.perspectiveTransform(center, H)[0][0]

        tile_px = np.array([[[center_on_tile[0], center_on_tile[1]]]], dtype=np.float64)
        geo = cv2.perspectiveTransform(tile_px, tile.H_pixel_to_geo)[0][0]

        top = np.array([[[fw/2, 0]]], dtype=np.float32)
        top_on_tile = cv2.perspectiveTransform(top, H)[0][0]
        dx = top_on_tile[0] - center_on_tile[0]
        dy = top_on_tile[1] - center_on_tile[1]
        heading = math.degrees(math.atan2(dx, -dy)) % 360

        result.found = True
        result.lat, result.lon = geo[0], geo[1]
        result.heading = heading
        result.confidence = inliers / len(good)
        result.match_count = inliers
        result.tile_key = f"{tile.row}_{tile.col}"
        result.timestamp = time.time()
        return result

    def _optical_flow_fallback(self, gray) -> 'MatchResult':
        """Estimate position using optical flow when matching fails"""
        result = self.MatchResult()
        if self.prev_gray is None or self.last_result is None:
            self.prev_gray = gray
            return result

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x, flow_y = np.median(flow[..., 0]), np.median(flow[..., 1])

        scale = 0.00001  # rough pixel-to-degree estimate
        self.of_position[0] -= flow_y * scale
        self.of_position[1] += flow_x * scale

        result.found = True
        result.lat, result.lon = self.of_position[0], self.of_position[1]
        result.heading = self.last_result.heading
        result.confidence = 0.2
        result.tile_key = "optical_flow"
        result.timestamp = time.time()
        self.prev_gray = gray
        return result


# ============================================================
# KALMAN FILTER
# ============================================================
class PositionKalman:
    def __init__(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100
        self.Q = np.diag([1e-8, 1e-8, 1e-5, 1e-5])
        self.R = np.diag([1e-6, 1e-6])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.initialized = False
        self.last_time = None

    def update(self, lat, lon, confidence=1.0):
        now = time.time()
        if not self.initialized:
            self.x[:2] = [lat, lon]
            self.initialized = True
            self.last_time = now
            return lat, lon

        dt = now - self.last_time
        if dt > 0:
            F = np.eye(4); F[0,2] = F[1,3] = dt
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + self.Q * dt
        self.last_time = now

        R = self.R / max(confidence, 0.1)
        z = np.array([lat, lon])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[0], self.x[1]

    def get_position(self):
        return self.x[0], self.x[1]


# ============================================================
# NMEA GPS SIMULATOR
# ============================================================
class NMEASimulator:
    def __init__(self, config: Config):
        self.config = config
        self.port: Optional[serial.Serial] = None
        self.running = False
        self.lat = self.lon = self.alt = self.heading = self.speed = 0.0
        self.confidence = 0.0
        self.valid = False
        self.lock = threading.Lock()

    def connect(self):
        try:
            self.port = serial.Serial(self.config.gps_port, self.config.gps_baud, timeout=1)
            log.info(f"NMEA port: {self.config.gps_port} @ {self.config.gps_baud}")
            return True
        except:
            log.warning("NMEA port unavailable — test mode (stdout)")
            return False

    def update_position(self, lat, lon, alt, heading, speed_ms, confidence):
        with self.lock:
            self.lat, self.lon, self.alt = lat, lon, alt
            self.heading = heading
            self.speed = speed_ms * 1.94384
            self.confidence = confidence
            self.valid = confidence > 0.2

    def _cksum(self, s):
        c = 0
        for ch in s: c ^= ord(ch)
        return f"{c:02X}"

    def _coord(self, dec, is_lat):
        d = ("N" if dec >= 0 else "S") if is_lat else ("E" if dec >= 0 else "W")
        dec = abs(dec); deg = int(dec); m = (dec - deg) * 60
        fmt = f"{deg:02d}{m:09.6f}" if is_lat else f"{deg:03d}{m:09.6f}"
        return fmt, d

    def _send(self):
        with self.lock:
            lat, lon, alt = self.lat, self.lon, self.alt
            hdg, spd, valid = self.heading, self.speed, self.valid
        t = datetime.now(timezone.utc)
        ts, ds = t.strftime("%H%M%S.00"), t.strftime("%d%m%y")
        la, lad = self._coord(lat, True)
        lo, lod = self._coord(lon, False)
        fq = self.config.fix_quality if valid else 0
        ns = self.config.num_satellites if valid else 0

        lines = []
        b = f"GPGGA,{ts},{la},{lad},{lo},{lod},{fq},{ns:02d},{self.config.hdop:.1f},{alt:.1f},M,0.0,M,,"
        lines.append(f"${b}*{self._cksum(b)}\r\n")
        b = f"GPRMC,{ts},{'A' if valid else 'V'},{la},{lad},{lo},{lod},{spd:.2f},{hdg:.2f},{ds},,,A"
        lines.append(f"${b}*{self._cksum(b)}\r\n")
        kmh = spd * 1.852
        b = f"GPVTG,{hdg:.2f},T,,M,{spd:.2f},N,{kmh:.2f},K,A"
        lines.append(f"${b}*{self._cksum(b)}\r\n")

        packet = "".join(lines)
        if self.port and self.port.is_open:
            try: self.port.write(packet.encode())
            except: pass

    def start(self):
        self.running = True
        def _loop():
            while self.running:
                self._send()
                time.sleep(1.0 / self.config.nmea_rate_hz)
        threading.Thread(target=_loop, daemon=True).start()
        log.info(f"NMEA output: {self.config.nmea_rate_hz} Hz")

    def stop(self):
        self.running = False
        if self.port: self.port.close()


# ============================================================
# FC TELEMETRY
# ============================================================
class FCTelemetry:
    def __init__(self, config: Config):
        self.config = config
        self.master = None
        self.data = {
            'alt': 0.0, 'groundspeed': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'battery_v': 0.0, 'battery_pct': 0,
            'mode': '', 'armed': False
        }
        self.lock = threading.Lock()

    def connect(self):
        try:
            from pymavlink import mavutil
            self.master = mavutil.mavlink_connection(self.config.fc_port, baud=self.config.fc_baud)
            self.master.wait_heartbeat(timeout=10)
            log.info(f"FC connected: sys={self.master.target_system}")
            for sid in range(13):
                self.master.mav.request_data_stream_send(
                    self.master.target_system, self.master.target_component, sid, 10, 1)
            return True
        except Exception as e:
            log.error(f"FC connection error: {e}")
            return False

    def get(self, key=None):
        with self.lock:
            return self.data.get(key, 0) if key else self.data.copy()

    def start(self):
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        from pymavlink import mavutil
        while True:
            try:
                msg = self.master.recv_match(blocking=True, timeout=1)
                if not msg: continue
                mt = msg.get_type()
                with self.lock:
                    if mt == 'ATTITUDE':
                        self.data.update(roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw)
                    elif mt == 'VFR_HUD':
                        self.data.update(groundspeed=msg.groundspeed, alt=msg.alt)
                    elif mt == 'SYS_STATUS':
                        self.data.update(battery_v=msg.voltage_battery/1000, battery_pct=msg.battery_remaining)
                    elif mt == 'HEARTBEAT':
                        self.data.update(
                            mode=mavutil.mode_string_v10(msg),
                            armed=bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
            except:
                time.sleep(0.1)


# ============================================================
# INITIAL POSITION FINDER
# ============================================================
class InitialLocalizer:
    """Scans all tiles on startup to determine initial drone position"""

    def __init__(self, tile_cache: TileCache, config: Config):
        self.cache = tile_cache
        self.config = config

    def find_initial_position(self, frame: np.ndarray) -> Optional[str]:
        """Quick-scan all tiles using downscaled images"""
        log.info("Searching for initial position (scanning all tiles)...")

        detector = self.cache.detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        small = cv2.resize(gray, (320, 240))
        kp, des = detector.detectAndCompute(small, None)

        if des is None or len(kp) < 10:
            log.warning("Insufficient features in initial frame")
            return None

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING if self.config.feature_detector != "SIFT" else cv2.NORM_L2)

        best_key, best_count = None, 0

        for key, meta in self.cache.tile_meta.items():
            img_path = os.path.join(self.config.tile_dir, meta["image"])
            if not os.path.exists(img_path):
                continue

            tile_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if tile_img is None:
                continue

            tile_small = cv2.resize(tile_img, (500, 500))
            tile_kp, tile_des = detector.detectAndCompute(tile_small, None)
            if tile_des is None:
                continue

            try:
                matches = matcher.knnMatch(des, tile_des, k=2)
            except:
                continue

            good = sum(1 for p in matches if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)

            if good > best_count:
                best_count = good
                best_key = key
            if good > 30:
                break

        if best_key and best_count >= 10:
            log.info(f"Initial position found: tile {best_key} ({best_count} matches)")
            row, col = map(int, best_key.split('_'))
            self.cache.load_tile(best_key)
            self.cache.ensure_neighborhood(row, col)
            return best_key
        else:
            log.warning("Initial position not found! Check reference maps.")
            return None


# ============================================================
# STATUS DISPLAY
# ============================================================
def print_status(match, fc, kalman, cache_info):
    k_lat, k_lon = kalman.get_position()
    print("\033[2J\033[H")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          VISUAL GPS — TILE-BASED NAVIGATION                  ║")
    print("╠════════════════════════════════════════════════════════════════╣")

    arm = "ARMED" if fc.get('armed') else "DISARMED"
    print(f"║ FC     {fc.get('mode','?'):10s} {arm:10s}  "
          f"Batt: {fc.get('battery_v',0):.1f}V ({fc.get('battery_pct',0):3d}%)     ║")
    print(f"║        Alt: {fc.get('alt',0):.1f}m  GS: {fc.get('groundspeed',0):.1f}m/s"
          f"                                 ║")
    print("╠════════════════════════════════════════════════════════════════╣")

    if match and match.found:
        src = "VISUAL" if match.tile_key != "optical_flow" else "OPT.FLOW"
        print(f"║ {src:8s}  Lat: {match.lat:.7f}  Lon: {match.lon:.7f}            ║")
        print(f"║           Conf: {match.confidence:.0%}  Matches: {match.match_count:3d}"
              f"  Hdg: {match.heading:.0f}°          ║")
        print(f"║           Tile: {match.tile_key:20s}                     ║")
    else:
        print("║ --- No match found ---                                         ║")

    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║ NMEA    Lat: {k_lat:.7f}  Lon: {k_lon:.7f}                   ║")
    print(f"║ CACHE   {cache_info:48s}     ║")
    print("╚════════════════════════════════════════════════════════════════╝")


# ============================================================
# MAIN
# ============================================================
def main():
    log.info("=" * 60)
    log.info("  VISUAL GPS — TILE-BASED NAVIGATION")
    log.info("=" * 60)

    config = Config.load()
    tile_cache = TileCache(config)

    if not tile_cache.tile_meta:
        log.error("No tiles found! Run prepare_tiles.py first.")
        return

    localizer = TileLocalizer(tile_cache, config)
    kalman = PositionKalman()
    nmea = NMEASimulator(config)
    telemetry = FCTelemetry(config)
    init_loc = InitialLocalizer(tile_cache, config)

    nmea.connect()
    fc_ok = telemetry.connect()

    cap = cv2.VideoCapture(config.cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.cam_height)
    cap.set(cv2.CAP_PROP_FPS, config.cam_fps)

    if not cap.isOpened():
        log.error("Camera failed to open!")
        return

    nmea.start()
    if fc_ok:
        telemetry.start()

    # Initial position scan
    log.info("Searching for initial position...")
    for _ in range(10):
        ret, frame = cap.read()
    if ret:
        init_loc.find_initial_position(frame)

    log.info("System ready — Ctrl+C to exit")

    count = 0
    match_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            match_result = localizer.match_frame(frame)

            if match_result.found:
                k_lat, k_lon = kalman.update(
                    match_result.lat, match_result.lon, match_result.confidence)
                fc = telemetry.get()
                nmea.update_position(
                    k_lat, k_lon, fc.get('alt', 50),
                    match_result.heading, fc.get('groundspeed', 0),
                    match_result.confidence)

            count += 1
            if count % 15 == 0:
                fc = telemetry.get() if fc_ok else {}
                ci = f"Loaded: {len(tile_cache.cache)}/{config.max_cached_tiles} tiles"
                print_status(match_result, fc, kalman, ci)

            time.sleep(0.01)

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        nmea.stop()
        cap.release()


if __name__ == '__main__':
    main()
