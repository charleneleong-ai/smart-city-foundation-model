"""Demo: personalised deployment for a hot, smoky grassfire over the sample watch."""
from sctwin.deploy import FireScenario, Constraints, deploy, explain, sample_roster


def main() -> None:
    scenario = FireScenario(
        cell="8a1fb46622dffff", fire_type="grass", size=4.0, pm25=120.0,
        temp_c=36.0, wind_speed=11.0, wind_dir=70.0, duration_min=180.0,
    )
    roster = sample_roster()
    plan = deploy(scenario, roster, Constraints(required_capacity=3.0))
    print(explain(plan, roster))


if __name__ == "__main__":
    main()
