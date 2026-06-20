import pytest
from sctwin.deploy.hazard import FIRE_TOXICITY, FireScenario


def _scn(**kw):
    base = dict(cell="8a1fb46622dffff", fire_type="grass", size=4.0, pm25=50.0,
               temp_c=30.0, wind_speed=8.0, wind_dir=70.0, duration_min=120.0)
    return FireScenario(**{**base, **kw})


def test_toxicity_looks_up_fire_type():
    assert _scn(fire_type="grass").toxicity() == FIRE_TOXICITY["grass"]
    assert _scn(fire_type="ev_lithium").toxicity() > _scn(fire_type="grass").toxicity()


@pytest.mark.parametrize("bad", [{"fire_type": "volcano"}, {"wind_dir": 360.0}, {"wind_dir": -1.0}])
def test_rejects_invalid_fields(bad):
    with pytest.raises(ValueError):
        _scn(**bad)
