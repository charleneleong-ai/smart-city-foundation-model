from sctwin.deploy.exposure import heat_load, toxicant_dose
from sctwin.deploy.hazard import FireScenario

SCN = FireScenario("c", "grass", 4.0, pm25=50.0, temp_c=30.0, wind_speed=8.0, wind_dir=70.0, duration_min=120.0)


def test_toxicant_dose_monotone_in_time_and_ppe():
    assert toxicant_dose(SCN, 30, "ba") > toxicant_dose(SCN, 20, "ba")  # more time -> more dose
    assert toxicant_dose(SCN, 20, "ba") < toxicant_dose(SCN, 20, "standard")  # BA cuts inhaled dose
    assert toxicant_dose(SCN, 20, "staging") < toxicant_dose(SCN, 20, "ba")  # reserve = least


def test_heat_load_rises_with_temp_and_exertion():
    hot = FireScenario("c", "grass", 4.0, 50.0, temp_c=38.0, wind_speed=8.0, wind_dir=70.0, duration_min=120.0)
    assert heat_load(hot, 20, "ba", "ba") > heat_load(SCN, 20, "ba", "ba")  # hotter -> more load
    assert heat_load(SCN, 20, "ba", "ba") > heat_load(SCN, 20, "command", "command")  # exertion
