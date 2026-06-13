"""City / region presets + viewport helper for the demos.

Cities use fine H3 (res 8) over a small bbox; regions use coarse H3 over a wide bbox to
show a real geographic temperature gradient. Cell counts are kept <=400 because the
Open-Meteo adapter makes one API call per cell.
"""

import math

PRESETS: dict[str, dict] = {
    "london": {"south": 51.46, "west": -0.20, "north": 51.55, "east": -0.05,
               "res": 8, "lat": 51.505, "lon": -0.12, "zoom": 10.6, "pitch": 50.0},
    "nyc": {"south": 40.66, "west": -74.03, "north": 40.78, "east": -73.87,
            "res": 8, "lat": 40.72, "lon": -73.95, "zoom": 10.4, "pitch": 50.0},
    "tokyo": {"south": 35.63, "west": 139.66, "north": 35.74, "east": 139.84,
              "res": 8, "lat": 35.685, "lon": 139.75, "zoom": 10.4, "pitch": 50.0},
    # region: Great Britain, north-south gradient (res 4 ~ 312 cells)
    "uk": {"south": 50.0, "west": -6.0, "north": 58.7, "east": 1.8,
           "res": 4, "lat": 54.5, "lon": -2.6, "zoom": 4.9, "pitch": 45.0},
    # region: British Isles incl. Ireland (res 4 ~ 541 cells; res 5 ~ 3800, gridded only)
    "isles": {"south": 49.8, "west": -11.0, "north": 59.0, "east": 2.0,
              "res": 4, "lat": 54.0, "lon": -4.5, "zoom": 4.7, "pitch": 45.0},
}


def bbox_and_zoom(
    preset: dict, radius_km: float | None, res_override: int | None
) -> tuple[float, float, float, float, float, int]:
    """Resolve a preset (+ optional --radius / --res overrides) to a bbox, zoom, and res.
    With a radius, build a square of +/-radius_km around the preset centre and fit zoom."""
    res = res_override if res_override is not None else preset["res"]
    if radius_km is not None:
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * math.cos(math.radians(preset["lat"])))
        south, west = preset["lat"] - dlat, preset["lon"] - dlon
        north, east = preset["lat"] + dlat, preset["lon"] + dlon
        zoom = math.log2(360.0 / max(north - south, east - west)) - 0.4
    else:
        south, west, north, east = preset["south"], preset["west"], preset["north"], preset["east"]
        zoom = preset.get("zoom", 10.6)
    return south, west, north, east, zoom, res
