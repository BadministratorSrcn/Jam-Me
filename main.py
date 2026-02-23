#!/usr/bin/env python3
"""
=============================================================
VISUAL GPS — GPS-Denied Navigation with NMEA Simulation
=============================================================
Single reference map mode for areas under 1 km².
For larger areas, use main_tile.py with the tile system.

Requirements:
    pip install opencv-python-headless numpy pyserial pymavlink

Usage:
    1. Place reference images in reference_maps/
    2. Edit reference_maps/config.json with corner coordinates
    3. python3 main.py

    python3 main.py --params   # Show ArduPilot parameter reference
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
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
log = logging.getLogger("VisualGPS")


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class Config:
    # Serial ports
    fc_port: str = "/dev/ttyAMA0"       # FC telemetry (MAVLink)
    gps_port: str = "/dev/ttyUSB0"      # NMEA output (to FC GPS input)
    fc_baud: int = 921600
    gps_baud: int = 9600

    # Camera
    cam_id: int = 0
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30

    # Reference maps
    map_dir: str = "reference_maps"
    map_config: str = "config.json"

    # Feature matching
    feature_detector: str = "ORB"        # ORB, SIFT, AKAZE
    min_match_count: int = 15
    match_ratio: float = 0.75            # Lowe's ratio test threshold
    ransac_threshold: float = 5.0

    # NMEA output
    nmea_rate_hz: int = 5
    fix_quality: int = 1                 # 1=GPS fix, 4=RTK fixed
    num_satellites: int = 12
    hdop: float = 0.9

    # Kalman filter
    process_noise: float = 0.1
    measurement_noise: float = 1.0

    @classmethod
    def load(cls, path: str = "system_config.json"):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        return cls()


# ============================================================
# REFERENCE MAP MANAGER
# ============================================================
class ReferenceMapManager:
    """
    Loads and geo-references satellite imagery.

    config.json format:
    {
        "maps": [
            {
                "image": "area1.jpg",
                "corners": {
                    "top_left":     {"lat": 39.9200, "lon": 32.8540},
                    "top_right":    {"lat": 39.9200, "lon": 32.8640},
                    "bottom_left":  {"lat": 39.9150, "lon": 32.8540},
                    "bottom_right": {"lat": 39.9150, "lon": 32.8640}
                },
                "altitude_agl": 50
            }
        ]
    }
    """

    @dataclass
    class RefMap:
        image: np.ndarray
        gray: np.ndarray
        keypoints: list
        descriptors: np.ndarray
        corners_latlon: np.ndarray
        corners_pixel: np.ndarray
        H_pixel_to_geo: np.ndarray
        name: str = ""
        altitude: float = 50.0

    def __init__(self, config: Config):
        self.config = config
        self.maps: List[ReferenceMapManager.RefMap] = []
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()

    def _create_detector(self):
        if self.config.feature_detector == "SIFT":
            return cv2.SIFT_create(nfeatures=2000)
        elif self.config.feature_detector == "AKAZE":
            return cv2.AKAZE_create()
        return cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)

    def _create_matcher(self):
        if self.config.feature_detector == "SIFT":
            return cv2.BFMatcher(cv2.NORM_L2)
        return cv2.BFMatcher(cv2.NORM_HAMMING)

    def load_maps(self) -> bool:
        """Load all reference maps from config"""
        config_path = os.path.join(self.config.map_dir, self.config.map_config)

        if not os.path.exists(config_path):
            log.error(f"Map config not found: {config_path}")
            self._create_example_config(config_path)
            return False

        with open(config_path) as f:
            map_data = json.load(f)

        for entry in map_data.get("maps", []):
            img_path = os.path.join(self.config.map_dir, entry["image"])
            if not os.path.exists(img_path):
                log.warning(f"Map file not found: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                log.warning(f"Failed to read map: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.detector.detectAndCompute(gray, None)

            if des is None or len(kp) < 50:
                log.warning(f"Insufficient features: {img_path} ({len(kp) if kp else 0})")
                continue

            c = entry["corners"]
            corners_latlon = np.array([
                [c["top_left"]["lat"],     c["top_left"]["lon"]],
                [c["top_right"]["lat"],    c["top_right"]["lon"]],
                [c["bottom_right"]["lat"], c["bottom_right"]["lon"]],
                [c["bottom_left"]["lat"],  c["bottom_left"]["lon"]]
            ], dtype=np.float64)

            h, w = gray.shape
            corners_pixel = np.array([
                [0, 0], [w, 0], [w, h], [0, h]
            ], dtype=np.float64)

            H_p2g, _ = cv2.findHomography(corners_pixel, corners_latlon)

            ref = self.RefMap(
                image=img, gray=gray, keypoints=kp, descriptors=des,
                corners_latlon=corners_latlon, corners_pixel=corners_pixel,
                H_pixel_to_geo=H_p2g, name=entry["image"],
                altitude=entry.get("altitude_agl", map_data.get("default_altitude", 50))
            )
            self.maps.append(ref)
            log.info(f"Map loaded: {entry['image']} ({len(kp)} features, {w}x{h})")

        log.info(f"Total {len(self.maps)} reference map(s) loaded")
        return len(self.maps) > 0

    def _create_example_config(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        example = {
            "maps": [
                {
                    "image": "area1.jpg",
                    "corners": {
                        "top_left":     {"lat": 39.9200, "lon": 32.8540},
                        "top_right":    {"lat": 39.9200, "lon": 32.8640},
                        "bottom_left":  {"lat": 39.9150, "lon": 32.8540},
                        "bottom_right": {"lat": 39.9150, "lon": 32.8640}
                    },
                    "altitude_agl": 50
                }
            ],
            "default_altitude": 50
        }
        with open(path, "w") as f:
            json.dump(example, f, indent=4)
        log.info(f"Example config created: {path}")


# ============================================================
# VISUAL LOCALIZER
# ============================================================
class VisualLocalizer:
    """Matches camera frames against reference maps to determine position"""

    @dataclass
    class MatchResult:
        found: bool = False
        lat: float = 0.0
        lon: float = 0.0
        altitude: float = 0.0
        heading: float = 0.0
        confidence: float = 0.0
        match_count: int = 0
        map_name: str = ""
        pixel_pos: Tuple[float, float] = (0, 0)
        timestamp: float = 0.0

    def __init__(self, map_manager: ReferenceMapManager, config: Config):
        self.map_mgr = map_manager
        self.config = config
        self.detector = map_manager.detector
        self.matcher = map_manager.matcher
        self.last_result: Optional[self.MatchResult] = None

    def match_frame(self, frame: np.ndarray) -> 'VisualLocalizer.MatchResult':
        """Match a camera frame against all reference maps"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) < 10:
            return self.MatchResult()

        best_result = self.MatchResult()
        best_inliers = 0

        for ref_map in self.map_mgr.maps:
            result = self._match_single_map(kp_frame, des_frame, gray.shape, ref_map)
            if result.found and result.match_count > best_inliers:
                best_inliers = result.match_count
                best_result = result

        if best_result.found:
            self.last_result = best_result
        return best_result

    def _match_single_map(self, kp_frame, des_frame, frame_shape,
                          ref_map: ReferenceMapManager.RefMap) -> 'MatchResult':
        result = self.MatchResult()

        try:
            matches = self.matcher.knnMatch(des_frame, ref_map.descriptors, k=2)
        except cv2.error:
            return result

        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.config.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.config.min_match_count:
            return result

        pts_frame = np.float32(
            [kp_frame[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        pts_map = np.float32(
            [ref_map.keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_frame, pts_map, cv2.RANSAC, self.config.ransac_threshold)
        if H is None:
            return result

        inliers = int(mask.sum())
        if inliers < self.config.min_match_count:
            return result

        # Camera center → map pixel → lat/lon
        fh, fw = frame_shape[:2]
        center = np.array([[[fw / 2, fh / 2]]], dtype=np.float32)
        center_on_map = cv2.perspectiveTransform(center, H)[0][0]

        map_px = np.array([[[center_on_map[0], center_on_map[1]]]], dtype=np.float64)
        geo_pos = cv2.perspectiveTransform(map_px, ref_map.H_pixel_to_geo)[0][0]

        # Heading calculation
        top_center = np.array([[[fw / 2, 0]]], dtype=np.float32)
        top_on_map = cv2.perspectiveTransform(top_center, H)[0][0]
        dx = top_on_map[0] - center_on_map[0]
        dy = top_on_map[1] - center_on_map[1]
        heading = math.degrees(math.atan2(dx, -dy)) % 360

        confidence = inliers / len(good_matches)

        result.found = True
        result.lat = geo_pos[0]
        result.lon = geo_pos[1]
        result.altitude = ref_map.altitude
        result.heading = heading
        result.confidence = confidence
        result.match_count = inliers
        result.map_name = ref_map.name
        result.pixel_pos = (center_on_map[0], center_on_map[1])
        result.timestamp = time.time()
        return result


# ============================================================
# KALMAN FILTER
# ============================================================
class PositionKalman:
    """2D Kalman filter — position + velocity state"""

    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.x = np.zeros(4)  # [lat, lon, vlat, vlon]
        self.P = np.eye(4) * 100
        self.Q = np.eye(4) * process_noise
        self.Q[0, 0] = self.Q[1, 1] = process_noise * 0.01
        self.R = np.eye(2) * measurement_noise
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.initialized = False
        self.last_time = None

    def update(self, lat: float, lon: float, confidence: float = 1.0):
        now = time.time()
        if not self.initialized:
            self.x[:2] = [lat, lon]
            self.initialized = True
            self.last_time = now
            return lat, lon

        dt = now - self.last_time
        if dt > 0:
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
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

    def get_velocity(self):
        return self.x[2], self.x[3]


# ============================================================
# NMEA GPS SIMULATOR
# ============================================================
class NMEASimulator:
    """
    Generates NMEA 0183 GPS sentences and sends them over serial.
    Connected to the FC's GPS input port.

    ArduPilot parameters:
        GPS_TYPE = 5           (NMEA)
        SERIALn_PROTOCOL = 5   (GPS)
        SERIALn_BAUD = 9       (9600)
    """

    def __init__(self, config: Config):
        self.config = config
        self.port: Optional[serial.Serial] = None
        self.running = False
        self.lock = threading.Lock()
        self.lat = self.lon = self.alt = self.heading = self.speed_knots = 0.0
        self.confidence = 0.0
        self.valid = False

    def connect(self) -> bool:
        try:
            self.port = serial.Serial(
                self.config.gps_port, self.config.gps_baud,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=1.0
            )
            log.info(f"NMEA port opened: {self.config.gps_port} @ {self.config.gps_baud}")
            return True
        except Exception as e:
            log.warning(f"NMEA port error: {e} — running in test mode (stdout)")
            return False

    def update_position(self, lat, lon, alt, heading, speed_ms, confidence):
        with self.lock:
            self.lat, self.lon, self.alt = lat, lon, alt
            self.heading = heading
            self.speed_knots = speed_ms * 1.94384
            self.confidence = confidence
            self.valid = confidence > 0.3

    def _checksum(self, sentence: str) -> str:
        cksum = 0
        for ch in sentence:
            cksum ^= ord(ch)
        return f"{cksum:02X}"

    def _to_nmea_coord(self, decimal_deg: float, is_lat: bool):
        direction = ("N" if decimal_deg >= 0 else "S") if is_lat else ("E" if decimal_deg >= 0 else "W")
        decimal_deg = abs(decimal_deg)
        degrees = int(decimal_deg)
        minutes = (decimal_deg - degrees) * 60
        formatted = f"{degrees:02d}{minutes:09.6f}" if is_lat else f"{degrees:03d}{minutes:09.6f}"
        return formatted, direction

    def _generate_gga(self):
        with self.lock:
            lat, lon, alt, valid = self.lat, self.lon, self.alt, self.valid
        t = datetime.now(timezone.utc).strftime("%H%M%S.00")
        la, lad = self._to_nmea_coord(lat, True)
        lo, lod = self._to_nmea_coord(lon, False)
        fq = self.config.fix_quality if valid else 0
        ns = self.config.num_satellites if valid else 0
        b = f"GPGGA,{t},{la},{lad},{lo},{lod},{fq},{ns:02d},{self.config.hdop:.1f},{alt:.1f},M,0.0,M,,"
        return f"${b}*{self._checksum(b)}\r\n"

    def _generate_rmc(self):
        with self.lock:
            lat, lon, hdg, spd, valid = self.lat, self.lon, self.heading, self.speed_knots, self.valid
        t = datetime.now(timezone.utc)
        ts, ds = t.strftime("%H%M%S.00"), t.strftime("%d%m%y")
        st = "A" if valid else "V"
        la, lad = self._to_nmea_coord(lat, True)
        lo, lod = self._to_nmea_coord(lon, False)
        b = f"GPRMC,{ts},{st},{la},{lad},{lo},{lod},{spd:.2f},{hdg:.2f},{ds},,,A"
        return f"${b}*{self._checksum(b)}\r\n"

    def _generate_vtg(self):
        with self.lock:
            hdg, spd = self.heading, self.speed_knots
        kmh = spd * 1.852
        b = f"GPVTG,{hdg:.2f},T,,M,{spd:.2f},N,{kmh:.2f},K,A"
        return f"${b}*{self._checksum(b)}\r\n"

    def _generate_gsa(self):
        sats = ",".join([f"{i:02d}" for i in range(1, 13)])
        pdop, vdop = self.config.hdop * 1.2, self.config.hdop * 0.8
        b = f"GPGSA,A,3,{sats},{pdop:.1f},{self.config.hdop:.1f},{vdop:.1f}"
        return f"${b}*{self._checksum(b)}\r\n"

    def _send_packet(self):
        packet = self._generate_gga() + self._generate_rmc() + self._generate_vtg() + self._generate_gsa()
        if self.port and self.port.is_open:
            try:
                self.port.write(packet.encode('ascii'))
                self.port.flush()
            except Exception as e:
                log.error(f"NMEA send error: {e}")

    def start(self):
        self.running = True
        def _sender():
            while self.running:
                self._send_packet()
                time.sleep(1.0 / self.config.nmea_rate_hz)
        threading.Thread(target=_sender, daemon=True).start()
        log.info(f"NMEA output started ({self.config.nmea_rate_hz} Hz)")

    def stop(self):
        self.running = False
        if self.port and self.port.is_open:
            self.port.close()


# ============================================================
# FC TELEMETRY READER
# ============================================================
class FCTelemetry:
    """Reads telemetry data from ArduPilot via MAVLink"""

    def __init__(self, config: Config):
        self.config = config
        self.master = None
        self.data = {
            'alt_baro': 0.0, 'alt_rangefinder': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'groundspeed': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'battery_v': 0.0, 'battery_pct': 0,
            'mode': '', 'armed': False,
        }
        self.lock = threading.Lock()

    def connect(self) -> bool:
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
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        from pymavlink import mavutil
        while True:
            try:
                msg = self.master.recv_match(blocking=True, timeout=1.0)
                if not msg:
                    continue
                mt = msg.get_type()
                with self.lock:
                    if mt == 'ATTITUDE':
                        self.data.update(roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw)
                    elif mt == 'GLOBAL_POSITION_INT':
                        self.data['alt_baro'] = msg.relative_alt / 1000.0
                    elif mt == 'VFR_HUD':
                        self.data.update(groundspeed=msg.groundspeed, alt_baro=msg.alt)
                    elif mt == 'LOCAL_POSITION_NED':
                        self.data.update(vx=msg.vx, vy=msg.vy, vz=msg.vz)
                    elif mt == 'RANGEFINDER':
                        self.data['alt_rangefinder'] = msg.distance
                    elif mt == 'SYS_STATUS':
                        self.data.update(battery_v=msg.voltage_battery/1000, battery_pct=msg.battery_remaining)
                    elif mt == 'HEARTBEAT':
                        self.data.update(
                            mode=mavutil.mode_string_v10(msg),
                            armed=bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
            except Exception as e:
                log.error(f"Telemetry read error: {e}")
                time.sleep(0.1)


# ============================================================
# STATUS DISPLAY
# ============================================================
def print_status(match_result, fc_data, kalman):
    k_lat, k_lon = kalman.get_position()
    k_vlat, k_vlon = kalman.get_velocity()

    print("\033[2J\033[H")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          VISUAL GPS — GPS-DENIED NAVIGATION             ║")
    print("╠══════════════════════════════════════════════════════════╣")

    mode = fc_data.get('mode', '?')
    armed = "ARMED" if fc_data.get('armed') else "DISARMED"
    batt = fc_data.get('battery_v', 0)
    batt_pct = fc_data.get('battery_pct', 0)
    alt = fc_data.get('alt_baro', 0)
    gs = fc_data.get('groundspeed', 0)

    print(f"║ FC     Mode: {mode:12s}  {armed:10s}                ║")
    print(f"║        Batt: {batt:.1f}V ({batt_pct:3d}%)  Alt: {alt:.1f}m  GS: {gs:.1f}m/s   ║")
    print("╠══════════════════════════════════════════════════════════╣")

    if match_result and match_result.found:
        print(f"║ VISION  Lat: {match_result.lat:.7f}                         ║")
        print(f"║         Lon: {match_result.lon:.7f}                         ║")
        print(f"║         Conf: {match_result.confidence:.0%}  "
              f"Matches: {match_result.match_count:3d}  "
              f"Hdg: {match_result.heading:.0f}°       ║")
        print(f"║         Map: {match_result.map_name:20s}                ║")
    else:
        print("║ VISION  --- No match found ---                          ║")

    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║ NMEA    Lat: {k_lat:.7f}                                ║")
    print(f"║         Lon: {k_lon:.7f}                                ║")
    spd = math.sqrt(k_vlat**2 + k_vlon**2) * 111000
    print(f"║         Speed: {spd:.2f} m/s                              ║")
    print("╚══════════════════════════════════════════════════════════╝")


# ============================================================
# MAIN
# ============================================================
def main():
    log.info("Starting Visual GPS...")

    config = Config.load()

    # Load reference maps
    map_mgr = ReferenceMapManager(config)
    if not map_mgr.load_maps():
        log.error("No reference maps loaded! Check reference_maps/ directory.")
        log.info("Example config.json has been created — edit and restart.")
        return

    # Initialize modules
    localizer = VisualLocalizer(map_mgr, config)
    kalman = PositionKalman(config.process_noise, config.measurement_noise)
    nmea = NMEASimulator(config)
    telemetry = FCTelemetry(config)

    # Connect
    nmea.connect()
    fc_ok = telemetry.connect()

    # Camera
    cap = cv2.VideoCapture(config.cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.cam_height)
    cap.set(cv2.CAP_PROP_FPS, config.cam_fps)

    if not cap.isOpened():
        log.error("Camera failed to open!")
        return

    log.info(f"Camera opened: {config.cam_width}x{config.cam_height}")

    # Start threads
    nmea.start()
    if fc_ok:
        telemetry.start()

    log.info("System ready — Ctrl+C to exit")

    frame_count = 0
    match_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            match_result = localizer.match_frame(frame)

            if match_result.found:
                k_lat, k_lon = kalman.update(
                    match_result.lat, match_result.lon, match_result.confidence
                )
                fc = telemetry.get()
                nmea.update_position(
                    lat=k_lat, lon=k_lon,
                    alt=fc.get('alt_baro', match_result.altitude),
                    heading=match_result.heading,
                    speed_ms=fc.get('groundspeed', 0),
                    confidence=match_result.confidence
                )

            frame_count += 1
            if frame_count % 10 == 0:
                fc = telemetry.get() if fc_ok else {}
                print_status(match_result, fc, kalman)

            time.sleep(0.01)

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        nmea.stop()
        cap.release()
        log.info("System stopped.")


# ============================================================
# ARDUPILOT PARAMETER REFERENCE
# ============================================================
def print_ardupilot_params():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              ArduPilot PARAMETER REFERENCE                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  GPS Configuration (NMEA simulation):                        ║
║    GPS_TYPE         = 5    (NMEA)                            ║
║    GPS_TYPE2        = 0    (None — single GPS)               ║
║                                                              ║
║  Serial Port (cable connected to GPS port):                  ║
║    SERIALn_PROTOCOL = 5    (GPS)                             ║
║    SERIALn_BAUD     = 9    (9600)                            ║
║    * n = SERIAL number for the GPS port                      ║
║    * Example: SERIAL3=TELEM2, SERIAL4=GPS2                   ║
║                                                              ║
║  EKF Configuration:                                          ║
║    EK3_SRC1_POSXY   = 3    (GPS — via NMEA)                 ║
║    EK3_SRC1_POSZ    = 1    (Barometer)                       ║
║    EK3_SRC1_VELXY   = 3    (GPS)                             ║
║    EK3_SRC1_YAW     = 1    (Compass)                         ║
║                                                              ║
║  Arming:                                                     ║
║    ARMING_CHECK     = -1   (or remove GPS check)             ║
║                                                              ║
║  Wiring:                                                     ║
║    Raspberry Pi TX  →  FC GPS/SERIAL RX                      ║
║    Raspberry Pi GND →  FC GND                                ║
║    Voltage: 3.3V (level-compatible)                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    import sys
    if '--params' in sys.argv:
        print_ardupilot_params()
    else:
        main()
