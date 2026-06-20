"""Render the firefighter deployment engine onto the 3D twin: a Fire domain (smoke/heat/dose
hexes) plus risk-coloured crew markers and a roster panel. Deterministic — fixed scenario."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make render_3d importable from repo root

from render_3d import to_self_contained_html  # noqa: E402

from sctwin.deploy import Constraints, FireScenario, deploy, sample_roster  # noqa: E402
from sctwin.deploy.viz import deploy_map  # noqa: E402

PRESET = {"name": "Camden", "lat": 51.54, "lon": -0.14, "zoom": 12.5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="london_fire_3d.html")
    args = ap.parse_args()

    scenario = FireScenario(cell="8a1fb46622dffff", fire_type="grass", size=4.0, pm25=120.0,
                            temp_c=36.0, wind_speed=11.0, wind_dir=70.0, duration_min=180.0)
    roster = sample_roster()
    plan = deploy(scenario, roster, Constraints(required_capacity=3.0))
    m = deploy_map(scenario, plan, roster, preset=PRESET)
    html = to_self_contained_html([m], title="Firefighter Deployment — Camden grassfire",
                                  about="Personalised exposure→health deployment. Crew coloured by risk; "
                                        "hover a firefighter for their score; roster panel top-right.")
    Path(args.out).write_text(html)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
