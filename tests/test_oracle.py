import polars as pl
import pytest

from sctwin.adapters.demand import LCLTariffAdapter, NEEDRetrofitAdapter
from sctwin.demand import lcl_group_profile, need_measure_split
from sctwin.reason.intervention import InterventionEnvironment, measured_question

# LCL ToU trial: one Std + one ToU household; ToU flattens the 18:00 peak (energy held)
_LCL = pl.DataFrame(
    {
        "stdorToU": ["Std", "Std", "ToU", "ToU"],
        "DateTime": [
            "2013-01-01 18:00:00.0000000",
            "2013-01-01 03:00:00.0000000",
            "2013-01-01 18:00:00.0000000",
            "2013-01-01 03:00:00.0000000",
        ],
        "value": [3.0, 1.0, 2.0, 2.0],  # Std peak (18:00) = 3.0; ToU flattened to 2.0 (peak cut)
    }
)
# NEED: two properties with loft insulation (saved ~1500 kWh/yr), one untreated control
_NEED = pl.DataFrame(
    {
        "LOFT_FLAG": [1, 1, 0],
        "Econ2010": [5000.0, 6000.0, 9000.0],  # pre
        "Econ2013": [4000.0, 4500.0, 8800.0],  # post
    }
)


class TestTariffOracle:
    """Measured tariff Δ = peak(ToU) − peak(Std) from the LCL trial profiles."""

    def test_group_profile_means_per_timestamp(self):
        std = lcl_group_profile(_LCL, "Std", cell="c").sort("time")
        assert std["value"].to_list() == [1.0, 3.0]  # 03:00 then 18:00
        assert std["layer"].unique().to_list() == ["load"] and std["cell"].unique().to_list() == [
            "c"
        ]

    def test_measured_question_uses_the_real_peak_delta(self):
        control = lcl_group_profile(_LCL, "Std", cell="c")
        treated = lcl_group_profile(_LCL, "ToU", cell="c")
        q = measured_question("tariff", control, treated, cell="c", metric="peak")
        assert q.intervention.kind == "tariff"
        assert q.true_delta == pytest.approx(-1.0)  # 1.0 - 2.0: the trial cut peak load


class TestRetrofitOracle:
    """Measured retrofit Δ = mean(post) − mean(pre) for NEED properties with the measure."""

    def test_split_keeps_only_treated_properties(self):
        pre, post = need_measure_split(
            _NEED, measure_col="LOFT_FLAG", pre_col="Econ2010", post_col="Econ2013", cell="c"
        )
        assert pre["value"].to_list() == [5000.0, 6000.0]  # the untreated row (flag 0) is excluded
        assert post["value"].to_list() == [4000.0, 4500.0]

    def test_measured_question_is_an_annual_saving(self):
        pre, post = need_measure_split(
            _NEED, measure_col="LOFT_FLAG", pre_col="Econ2010", post_col="Econ2013", cell="c"
        )
        q = measured_question("retrofit", pre, post, cell="c", metric="mean")
        assert q.intervention.kind == "retrofit"
        assert q.true_delta == pytest.approx(-1250.0)  # mean(4000,4500) - mean(5000,6000)


def test_from_questions_env_scores_an_oracle_policy():
    control = lcl_group_profile(_LCL, "Std", cell="c")
    treated = lcl_group_profile(_LCL, "ToU", cell="c")
    env = InterventionEnvironment.from_questions(
        [measured_question("tariff", control, treated, cell="c", metric="peak")]
    )
    assert env.rollout(lambda q: q.true_delta) == {"mean_reward": pytest.approx(1.0), "n": 1.0}


def test_measured_question_rejects_an_empty_frame():
    treated = lcl_group_profile(_LCL, "ToU", cell="c")
    empty = lcl_group_profile(_LCL, "Nonexistent", cell="c")  # an unknown group filters to empty
    with pytest.raises(ValueError, match="empty"):
        measured_question("tariff", empty, treated, cell="c", metric="peak")


class _StubLCL(LCLTariffAdapter):
    def _read(self) -> pl.DataFrame:
        return _LCL


class _StubNEED(NEEDRetrofitAdapter):
    def _read(self) -> pl.DataFrame:
        return _NEED


def test_adapters_yield_before_after_pairs():
    control, treated = _StubLCL("unused").profiles("c")  # _read is stubbed; source is irrelevant
    assert (
        float(control["value"].max()) == 3.0 and float(treated["value"].max()) == 2.0
    )  # ToU cuts peak
    pre, post = _StubNEED("unused").split("c")
    assert (
        float(pre["value"].mean()) == 5500.0 and float(post["value"].mean()) == 4250.0
    )  # retrofit saving
