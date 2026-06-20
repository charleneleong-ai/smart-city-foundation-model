from dataclasses import dataclass
from datetime import datetime

# fire type -> relative combustion-product toxicity of its smoke (dimensionless multiplier)
FIRE_TOXICITY: dict[str, float] = {
    "grass": 0.6,
    "dwelling": 1.0,
    "commercial": 1.3,
    "chemical": 2.0,
    "ev_lithium": 2.5,
}


@dataclass(frozen=True)
class FireScenario:
    """The fireground hazard state a deployment is scored against."""

    cell: str  # H3 cell of the incident
    fire_type: str  # key into FIRE_TOXICITY
    size: float  # fire size in suppression-demand units (drives required capacity K)
    pm25: float  # ambient smoke / PM2.5 at scene (ug/m3)
    temp_c: float  # ambient air temperature
    wind_speed: float  # 10 m wind speed (m/s)
    wind_dir: float  # 10 m wind direction, degrees [0, 360)
    duration_min: float  # expected incident duration

    def __post_init__(self) -> None:
        if self.fire_type not in FIRE_TOXICITY:
            raise ValueError(f"fire_type must be one of {tuple(FIRE_TOXICITY)}, got {self.fire_type!r}")
        if not 0.0 <= self.wind_dir < 360.0:
            raise ValueError(f"wind_dir must be in [0, 360), got {self.wind_dir}")

    def toxicity(self) -> float:
        return FIRE_TOXICITY[self.fire_type]

    @classmethod
    def from_live(
        cls,
        cell_h3: str,
        res: int,
        *,
        fire_type: str,
        size: float,
        duration_min: float,
        pm25: float,
        when: datetime,
        adapter,
    ) -> "FireScenario":
        """Build a scenario from live NWP. `adapter` is an OpenMeteoForecastAdapter configured with
        WEATHER_VARS; `pm25` (smoke) is supplied for now — CAMS/FRP wiring is a follow-up."""
        from sctwin.geo import Cell

        df = adapter.fetch([Cell(h3=cell_h3, res=res)], when, when)
        latest = df.sort("time").group_by("layer", maintain_order=True).last()
        vals = dict(zip(latest["layer"].to_list(), latest["value"].to_list()))
        return cls(
            cell=cell_h3,
            fire_type=fire_type,
            size=size,
            pm25=pm25,
            temp_c=vals["t2m"],
            wind_speed=vals["wind_speed"],
            wind_dir=vals["wind_dir"],
            duration_min=duration_min,
        )
