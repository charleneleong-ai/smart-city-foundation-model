"""City / region presets for the weather demo. Pure data, no deps.

Cities use fine H3 (res 8) over a small bbox; regions use coarse H3 over a wide bbox to
show a real geographic temperature gradient. Cell counts are kept <=400 because the
Open-Meteo adapter makes one API call per cell.
"""

PRESETS: dict[str, dict] = {
    "london": {"south": 51.46, "west": -0.20, "north": 51.55, "east": -0.05,
               "res": 8, "lat": 51.505, "lon": -0.12, "zoom": 10.6, "pitch": 50.0},
    "nyc": {"south": 40.66, "west": -74.03, "north": 40.78, "east": -73.87,
            "res": 8, "lat": 40.72, "lon": -73.95, "zoom": 10.4, "pitch": 50.0},
    "tokyo": {"south": 35.63, "west": 139.66, "north": 35.74, "east": 139.84,
              "res": 8, "lat": 35.685, "lon": 139.75, "zoom": 10.4, "pitch": 50.0},
    # region: Great Britain, north-south gradient (res 4 ~ 312 cells)
    "uk": {"south": 50.0, "west": -6.0, "north": 58.7, "east": 1.8,
           "res": 4, "lat": 54.5, "lon": -2.6, "zoom": 4.9, "pitch": 35.0},
}
