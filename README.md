# JAM-Me — GPS-Denied Navigation with Visual Localization & NMEA Simulation

<p align="center">
  <img src="docs/architecture.png" alt="System Architecture" width="700">
</p>

> **Autonomous flight without GPS** — using only a Raspberry Pi camera and pre-loaded satellite imagery to determine position and feed NMEA GPS data to any ArduPilot flight controller.

## ⚠️ Disclaimer

**This project is provided strictly for educational and research purposes only.**

By using this software, you acknowledge and agree that:

- The author(s) assume **NO responsibility or liability** for any financial losses, property damage, personal injury, or any other damages arising from the use or misuse of this software.
- This software is **NOT certified for commercial, military, or any safety-critical applications**. It has not undergone any formal verification, validation, or certification process.
- **You are solely responsible** for ensuring compliance with all applicable local, national, and international laws and regulations regarding drone operation, airspace usage, and GPS simulation in your jurisdiction.
- GPS simulation and spoofing may be **illegal** in many countries. It is your responsibility to verify the legality of using this software in your region before use.
- The author(s) make **no warranties or guarantees** regarding the accuracy, reliability, or fitness of this software for any particular purpose.
- Use of this software in real flight scenarios is **entirely at your own risk**. Always have a safety pilot and failsafe mechanisms in place.
- The author(s) are **not liable** for any crashes, flyaways, loss of control, or other incidents resulting from the use of this software.

**If you do not agree with these terms, do not use this software.**

---

## Overview

Visual GPS enables autonomous drone flight in GPS-denied environments by matching real-time camera footage against geo-referenced satellite imagery. The system outputs standard NMEA 0183 sentences to the flight controller — from the FC's perspective, it's receiving data from a regular GPS module.

### Key Features

- **Visual Localization** — ORB/SIFT/AKAZE feature matching against reference maps
- **Tile System** — Supports 50+ km² areas by splitting maps into 500m tiles with LRU caching
- **NMEA GPS Simulation** — Outputs $GPGGA, $GPRMC, $GPVTG, $GPGSA to FC serial port
- **Kalman Filtering** — Smooths position estimates and fuses visual + telemetry data
- **Optical Flow Fallback** — Maintains position estimate when visual matching fails
- **FC Telemetry Fusion** — Reads altitude, speed, attitude from ArduPilot via MAVLink
- **Auto-initialization** — Scans all tiles on startup to determine initial position

### How It Works

```
Pre-loaded satellite imagery (geo-referenced tiles)
         │
         ▼
   ORB Feature Extraction  ◄──── Downward-facing camera (live)
         │                              │
         ▼                              ▼
   KNN + RANSAC Matching ────→ Homography Matrix
         │
         ▼
   Pixel Coordinate → Lat/Lon Conversion
         │
         ▼
   Kalman Filter (position smoothing)
         │
         ▼
   NMEA Sentences ($GPGGA, $GPRMC, $GPVTG)
         │
         ▼
   Serial TX → Flight Controller GPS Input
```

The flight controller treats this as a standard GPS source. **Waypoint missions, RTL, Loiter, and Auto modes all work normally.**

## Hardware Requirements

| Component | Specification | Notes |
|-----------|--------------|-------|
| Raspberry Pi | 4B or 5 (4GB+ RAM) | Pi 5 recommended for better performance |
| Camera | Any downward-facing camera | Pi Camera Module v2/v3, USB webcam, or OAK-D |
| USB-UART Adapter | CP2102 or FTDI | For NMEA output to FC GPS port |
| Flight Controller | Any ArduPilot-compatible | Pixhawk, Cube, MatekF405, etc. |

### Wiring Diagram

```
Raspberry Pi                         Pixhawk / FC
─────────────                        ────────────

[USB-UART TX]     ──────────────→   [GPS RX]          (NMEA data)
[GPIO 14 TX]      ──────────────→   [TELEM2 RX]       (MAVLink)
[GPIO 15 RX]      ◄──────────────   [TELEM2 TX]       (MAVLink)
[GND]             ──────────────→   [GND]

[CSI/USB Camera]  →  Raspberry Pi   (facing downward)
```

> **Note:** Two serial connections are required — one for NMEA GPS output (unidirectional TX) and one for MAVLink telemetry (bidirectional). A USB-UART adapter provides the second port.

> **Voltage:** Pixhawk GPS port is typically 3.3V, matching Raspberry Pi GPIO. If using a 5V port, add a level shifter.

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/visual-gps.git
cd visual-gps

# Install dependencies
pip install opencv-python-headless numpy pyserial pymavlink

# Create tile directory
mkdir -p tiles
```

## Quick Start

### 1. Prepare Reference Maps

#### Option A: Split a large satellite image

If you have a single large satellite image covering the area:

```bash
python3 prepare_tiles.py \
    --top-left 39.95,32.80 \
    --bottom-right 39.88,32.90 \
    --tile-size 500 \
    --source satellite_image.jpg \
    --output tiles/
```

#### Option B: Generate grid, download manually

```bash
python3 prepare_tiles.py \
    --top-left 39.95,32.80 \
    --bottom-right 39.88,32.90 \
    --tile-size 500 \
    --output tiles/ \
    --kml
```

This creates:
- `tile_config.json` — GPS coordinates for each tile
- `grid_overlay.kml` — Open in Google Earth to visualize the grid
- Download script (optional)

Then manually save high-resolution satellite imagery for each tile from Google Earth Pro.

#### Option C: Small area (single photo)

For areas under 1 km², use the simple single-map mode:

```bash
# Edit reference_maps/config.json with corner coordinates
python3 main.py
```

### 2. Configure ArduPilot

Set these parameters on your flight controller:

```
# GPS Configuration (NMEA input)
GPS_TYPE        = 5     # NMEA
GPS_TYPE2       = 0     # None

# Serial port connected to Raspberry Pi NMEA output
SERIALn_PROTOCOL = 5   # GPS
SERIALn_BAUD     = 9   # 9600 (or 38 for 38400)
# n = serial port number (e.g., SERIAL3=TELEM2, SERIAL4=GPS2)

# EKF Source
EK3_SRC1_POSXY  = 3    # GPS (via NMEA)
EK3_SRC1_POSZ   = 1    # Barometer
EK3_SRC1_VELXY  = 3    # GPS
EK3_SRC1_YAW    = 1    # Compass

# Arming
ARMING_CHECK    = -1    # Or remove GPS check from bitmask
```

### 3. Run

```bash
# Large area (tile system) — recommended
python3 main_tile.py

# Small area (single reference map)
python3 main.py

# Show ArduPilot parameter reference
python3 main.py --params
```

## Tile System Architecture

For areas larger than 1 km², the system uses a tile-based approach:

```
50 km² area = ~7km × 7km
                ↓
    500m × 500m tiles with 50m overlap
                ↓
    ~200 tiles (each 2000×2000px)
                ↓
    Only 9 tiles loaded at a time (3×3 neighborhood)
```

```
┌──────┬──────┬──────┬──────┬──────┐
│      │      │      │      │      │
├──────┼──────╔══════╦══════╦══════╗
│      │      ║loaded║loaded║loaded║
├──────┼──────╠══════╬══════╬══════╣  ← 3×3 neighborhood
│      │      ║loaded║DRONE ║loaded║    in RAM (~200MB)
├──────┼──────╠══════╬══════╬══════╣
│      │      ║loaded║loaded║loaded║
├──────┼──────╚══════╩══════╩══════╝
│      │      │      │      │      │
└──────┴──────┴──────┴──────┴──────┘
         Remaining tiles stay on disk
```

When the drone crosses into a new tile, old tiles are evicted and new neighbors are preloaded in background threads.

### Performance Comparison

| Approach | RAM Usage | Search Time | Accuracy |
|----------|-----------|-------------|----------|
| Single large image (50 km²) | 10+ GB | 5+ seconds | Low |
| **Tile system (500m tiles)** | **~200 MB** | **~50 ms** | **High** |

## Configuration

### system_config.json

```json
{
    "fc_port": "/dev/ttyAMA0",
    "gps_port": "/dev/ttyUSB0",
    "fc_baud": 921600,
    "gps_baud": 9600,
    "cam_id": 0,
    "cam_width": 640,
    "cam_height": 480,
    "cam_fps": 30,
    "tile_dir": "tiles",
    "feature_detector": "ORB",
    "min_match_count": 15,
    "match_ratio": 0.75,
    "nmea_rate_hz": 5,
    "num_satellites": 12,
    "hdop": 0.9,
    "max_cached_tiles": 9,
    "optical_flow_fallback": true
}
```

### Reference Map Config (single-map mode)

```json
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
```

## File Structure

```
visual-gps/
├── main.py                  # Single-map mode (small areas <1 km²)
├── main_tile.py             # Tile-based mode (large areas 1-100+ km²)
├── prepare_tiles.py         # Map preparation tool
├── system_config.json       # System configuration
├── reference_maps/          # Single-map mode reference images
│   └── config.json
├── tiles/                   # Tile-based mode (auto-generated)
│   ├── tile_config.json
│   ├── grid_overlay.kml
│   ├── tile_000_000.jpg
│   ├── tile_000_001.jpg
│   └── ...
└── docs/
    └── architecture.png
```

## How Matching Works

1. **Feature Extraction** — ORB detects keypoints and computes descriptors for both the camera frame and reference tiles
2. **KNN Matching** — Brute-force matcher finds the 2 nearest neighbors for each descriptor
3. **Lowe's Ratio Test** — Filters ambiguous matches (ratio threshold: 0.75)
4. **RANSAC Homography** — Computes geometric transformation, rejecting outliers
5. **Coordinate Transform** — Camera center pixel → tile pixel → lat/lon via homography chain
6. **Kalman Filter** — Smooths position, estimates velocity, handles measurement noise
7. **NMEA Output** — Converts filtered lat/lon to standard NMEA 0183 sentences

When visual matching fails (low texture, darkness, etc.), **optical flow fallback** kicks in — tracking pixel displacement from the last known position with reduced confidence.

## Tips for Best Results

| Factor | Recommendation |
|--------|---------------|
| Camera resolution | 640×480 (speed/quality balance) |
| Reference image resolution | 2000×2000+ pixels per tile |
| Flight altitude | Close to the altitude the reference images represent |
| Ground texture | Detailed surfaces (buildings, roads) work best |
| Lighting | Similar conditions to reference imagery is ideal |
| Feature detector | ORB (fast, good for RPi), SIFT (more accurate but slower) |
| Tile overlap | 50m minimum to ensure smooth transitions |

## Known Limitations

- **Night flight** — Requires IR camera or illumination
- **Low-texture terrain** — Desert, water, snow may have insufficient features
- **Altitude mismatch** — Large differences between flight altitude and reference image altitude reduce accuracy
- **Seasonal changes** — Snow cover, foliage changes affect matching
- **Initial localization** — Takes 1-3 seconds to scan tiles on startup
- **Position drift** — Optical flow fallback accumulates error over time without visual matches

## ArduPilot Compatibility

Tested with:
- ArduCopter 4.3+ (multirotor)
- ArduPlane 4.3+ (fixed-wing / VTOL)

Works with any flight mode that uses GPS: Auto, Guided, RTL, Loiter, PosHold.

## License

MIT License — see [LICENSE](LICENSE) for details.

This software is provided "AS IS" without warranty of any kind. See the [Disclaimer](#️-disclaimer) section for full terms.

## Contributing

Contributions welcome! Areas that could use improvement:
- Deep learning feature matching (SuperPoint/SuperGlue)
- Multi-scale matching for variable altitude
- IMU integration for better dead-reckoning
- Night/thermal camera support
- SLAM integration (ORB-SLAM3)
