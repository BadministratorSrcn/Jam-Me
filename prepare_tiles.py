#!/usr/bin/env python3
"""
=============================================================
TILE MAP PREPARATION TOOL
=============================================================

Splits large satellite imagery into geo-referenced tiles
for areas up to 50+ km².

Supports 3 source types:
  1. Single large satellite image → auto-split into tiles
  2. Manual download from Google Earth/Maps
  3. Drone orthophoto (GeoTIFF)

Usage:
  # Split a large image into tiles
  python3 prepare_tiles.py --source satellite.jpg \
      --top-left 39.95,32.80 --bottom-right 39.90,32.87 \
      --tile-size 500 --output tiles/

  # Generate grid config + KML for manual download
  python3 prepare_tiles.py --generate-grid \
      --top-left 39.95,32.80 --bottom-right 39.90,32.87 \
      --tile-size 500 --output tiles/ --kml
=============================================================
"""

import cv2
import numpy as np
import json
import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List


# ============================================================
# GEO UTILITIES
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Distance between two GPS points in meters"""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def meters_to_degrees_lat(meters: float) -> float:
    return meters / 111320.0


def meters_to_degrees_lon(meters: float, lat: float) -> float:
    return meters / (111320.0 * math.cos(math.radians(lat)))


@dataclass
class TileInfo:
    row: int
    col: int
    top_left_lat: float
    top_left_lon: float
    top_right_lat: float
    top_right_lon: float
    bottom_left_lat: float
    bottom_left_lon: float
    bottom_right_lat: float
    bottom_right_lon: float
    image_file: str
    tile_size_m: float

    def center(self) -> Tuple[float, float]:
        lat = (self.top_left_lat + self.bottom_left_lat) / 2
        lon = (self.top_left_lon + self.top_right_lon) / 2
        return lat, lon


# ============================================================
# TILE GENERATOR
# ============================================================

class TileGenerator:

    def __init__(self, top_left: Tuple[float, float],
                 bottom_right: Tuple[float, float],
                 tile_size_m: float = 500,
                 overlap_m: float = 50,
                 output_dir: str = "tiles"):
        """
        Args:
            top_left: (lat, lon) top-left corner
            bottom_right: (lat, lon) bottom-right corner
            tile_size_m: tile size in meters
            overlap_m: overlap between tiles in meters (for seamless transitions)
            output_dir: output directory
        """
        self.tl_lat, self.tl_lon = top_left
        self.br_lat, self.br_lon = bottom_right
        self.tile_size_m = tile_size_m
        self.overlap_m = overlap_m
        self.output_dir = output_dir

        self.width_m = haversine_distance(self.tl_lat, self.tl_lon, self.tl_lat, self.br_lon)
        self.height_m = haversine_distance(self.tl_lat, self.tl_lon, self.br_lat, self.tl_lon)
        self.area_km2 = (self.width_m * self.height_m) / 1e6

        step_m = tile_size_m - overlap_m
        self.n_rows = math.ceil(self.height_m / step_m)
        self.n_cols = math.ceil(self.width_m / step_m)
        self.total_tiles = self.n_rows * self.n_cols

        print(f"Area: {self.width_m:.0f}m x {self.height_m:.0f}m = {self.area_km2:.1f} km²")
        print(f"Tile size: {tile_size_m}m (overlap: {overlap_m}m)")
        print(f"Grid: {self.n_rows} rows x {self.n_cols} cols = {self.total_tiles} tiles")

    def generate_tile_grid(self) -> List[TileInfo]:
        """Calculate coordinates for all tiles"""
        tiles = []
        step_m = self.tile_size_m - self.overlap_m

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                offset_north = row * step_m
                offset_east = col * step_m

                tl_lat = self.tl_lat - meters_to_degrees_lat(offset_north)
                tl_lon = self.tl_lon + meters_to_degrees_lon(offset_east, self.tl_lat)

                tr_lat = tl_lat
                tr_lon = tl_lon + meters_to_degrees_lon(self.tile_size_m, tl_lat)

                bl_lat = tl_lat - meters_to_degrees_lat(self.tile_size_m)
                bl_lon = tl_lon

                br_lat = bl_lat
                br_lon = tr_lon

                tile = TileInfo(
                    row=row, col=col,
                    top_left_lat=tl_lat, top_left_lon=tl_lon,
                    top_right_lat=tr_lat, top_right_lon=tr_lon,
                    bottom_left_lat=bl_lat, bottom_left_lon=bl_lon,
                    bottom_right_lat=br_lat, bottom_right_lon=br_lon,
                    image_file=f"tile_{row:03d}_{col:03d}.jpg",
                    tile_size_m=self.tile_size_m
                )
                tiles.append(tile)

        return tiles

    def split_large_image(self, image_path: str, tiles: List[TileInfo]):
        """Split a large satellite image into tiles"""
        print(f"\nLoading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Failed to read image: {image_path}")
            return False

        img_h, img_w = img.shape[:2]
        print(f"Image size: {img_w}x{img_h} pixels")

        px_per_m_x = img_w / self.width_m
        px_per_m_y = img_h / self.height_m
        print(f"Resolution: {1/px_per_m_x:.2f} m/px (horizontal), {1/px_per_m_y:.2f} m/px (vertical)")

        os.makedirs(self.output_dir, exist_ok=True)
        step_m = self.tile_size_m - self.overlap_m
        created = 0

        for tile in tiles:
            x1 = int(tile.col * step_m * px_per_m_x)
            y1 = int(tile.row * step_m * px_per_m_y)
            x2 = int(x1 + self.tile_size_m * px_per_m_x)
            y2 = int(y1 + self.tile_size_m * px_per_m_y)

            x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
            y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))

            if x2 - x1 < 50 or y2 - y1 < 50:
                continue

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (2000, 2000), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(self.output_dir, tile.image_file)
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            created += 1

        print(f"{created} tiles created in {self.output_dir}/")
        return True

    def save_tile_config(self, tiles: List[TileInfo], filename: str = "tile_config.json"):
        """Save tile configuration"""
        config = {
            "area": {
                "top_left": {"lat": self.tl_lat, "lon": self.tl_lon},
                "bottom_right": {"lat": self.br_lat, "lon": self.br_lon},
                "width_m": round(self.width_m, 1),
                "height_m": round(self.height_m, 1),
                "area_km2": round(self.area_km2, 2)
            },
            "grid": {
                "rows": self.n_rows,
                "cols": self.n_cols,
                "tile_size_m": self.tile_size_m,
                "overlap_m": self.overlap_m,
                "total_tiles": self.total_tiles
            },
            "tiles": []
        }

        for t in tiles:
            config["tiles"].append({
                "row": t.row,
                "col": t.col,
                "image": t.image_file,
                "corners": {
                    "top_left":     {"lat": round(t.top_left_lat, 8),     "lon": round(t.top_left_lon, 8)},
                    "top_right":    {"lat": round(t.top_right_lat, 8),    "lon": round(t.top_right_lon, 8)},
                    "bottom_left":  {"lat": round(t.bottom_left_lat, 8),  "lon": round(t.bottom_left_lon, 8)},
                    "bottom_right": {"lat": round(t.bottom_right_lat, 8), "lon": round(t.bottom_right_lon, 8)}
                }
            })

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved: {out_path}")

    def generate_download_script(self, tiles: List[TileInfo],
                                  zoom: int = 18,
                                  filename: str = "download_tiles.sh"):
        """Generate a shell script to download tile images from a tile server"""
        script = "#!/bin/bash\n"
        script += f"# Tile download script — {len(tiles)} tiles\n"
        script += "# Uses Yandex Static Maps (satellite layer)\n"
        script += "# For best results, use Google Earth Pro for manual download\n\n"
        script += "mkdir -p tiles\n\n"

        for t in tiles:
            clat, clon = t.center()
            script += f"# Tile [{t.row},{t.col}] center: {clat:.6f},{clon:.6f}\n"
            script += (
                f'wget -q -O tiles/{t.image_file} '
                f'"https://static-maps.yandex.ru/1.x/?ll={clon:.6f},{clat:.6f}'
                f'&z={zoom}&l=sat&size=650,450"\n'
            )
            script += "sleep 1\n\n"

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w') as f:
            f.write(script)
        os.chmod(out_path, 0o755)
        print(f"Download script: {out_path}")
        print("NOTE: For better quality, manual download from Google Earth Pro is recommended.")


# ============================================================
# GOOGLE EARTH KML GENERATOR
# ============================================================

def generate_grid_kml(tiles: List[TileInfo], output_path: str = "grid_overlay.kml"):
    """Create a KML file to visualize the tile grid in Google Earth"""
    kml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    kml += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
    kml += '<Document>\n'
    kml += '  <name>Visual GPS Tile Grid</name>\n'
    kml += '  <Style id="gridStyle">\n'
    kml += '    <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>\n'
    kml += '    <PolyStyle><color>400000ff</color></PolyStyle>\n'
    kml += '  </Style>\n'

    for t in tiles:
        kml += f'  <Placemark>\n'
        kml += f'    <name>Tile [{t.row},{t.col}]</name>\n'
        kml += f'    <styleUrl>#gridStyle</styleUrl>\n'
        kml += f'    <Polygon><outerBoundaryIs><LinearRing><coordinates>\n'
        kml += f'      {t.top_left_lon},{t.top_left_lat},0\n'
        kml += f'      {t.top_right_lon},{t.top_right_lat},0\n'
        kml += f'      {t.bottom_right_lon},{t.bottom_right_lat},0\n'
        kml += f'      {t.bottom_left_lon},{t.bottom_left_lat},0\n'
        kml += f'      {t.top_left_lon},{t.top_left_lat},0\n'
        kml += f'    </coordinates></LinearRing></outerBoundaryIs></Polygon>\n'
        kml += f'  </Placemark>\n'

    kml += '</Document>\n</kml>'

    with open(output_path, 'w') as f:
        f.write(kml)
    print(f"KML file created: {output_path}")
    print("Open in Google Earth Pro to visualize the grid overlay.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare geo-referenced tile maps for Visual GPS navigation"
    )

    parser.add_argument('--top-left', required=True,
                        help='Top-left corner: lat,lon (e.g., 39.95,32.80)')
    parser.add_argument('--bottom-right', required=True,
                        help='Bottom-right corner: lat,lon (e.g., 39.90,32.87)')
    parser.add_argument('--tile-size', type=int, default=500,
                        help='Tile size in meters (default: 500)')
    parser.add_argument('--overlap', type=int, default=50,
                        help='Tile overlap in meters (default: 50)')
    parser.add_argument('--output', default='tiles',
                        help='Output directory (default: tiles)')
    parser.add_argument('--source',
                        help='Large satellite image to split into tiles')
    parser.add_argument('--generate-grid', action='store_true',
                        help='Only generate grid config and KML (no image splitting)')
    parser.add_argument('--kml', action='store_true',
                        help='Generate Google Earth KML overlay')

    args = parser.parse_args()

    tl = tuple(map(float, args.top_left.split(',')))
    br = tuple(map(float, args.bottom_right.split(',')))

    gen = TileGenerator(
        top_left=tl, bottom_right=br,
        tile_size_m=args.tile_size, overlap_m=args.overlap,
        output_dir=args.output
    )

    tiles = gen.generate_tile_grid()
    gen.save_tile_config(tiles)

    if args.kml or args.generate_grid:
        kml_path = os.path.join(args.output, "grid_overlay.kml")
        generate_grid_kml(tiles, kml_path)

    if args.source and not args.generate_grid:
        gen.split_large_image(args.source, tiles)
    elif not args.generate_grid:
        gen.generate_download_script(tiles)
        print("\nTo download tiles manually:")
        print("  1. Open grid_overlay.kml in Google Earth Pro")
        print("  2. Save high-resolution satellite imagery for each tile")
        print(f"  3. Save as tiles/tile_ROW_COL.jpg in {args.output}/")

    print(f"\nDone. {len(tiles)} tiles prepared.")
    print(f"Next step: run main_tile.py with tile_config.json")


if __name__ == '__main__':
    main()
