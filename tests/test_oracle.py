import polars as pl
import pytest

from sctwin.adapters.demand import LCLTariffAdapter, NEEDRetrofitAdapter
from sctwin.demand import lcl_diurnal_profile, lcl_group_profile, need_measure_split
from sctwin.reason.intervention import (
    InterventionEnvironment,
    did_effect,
    did_question,
    measured_question,
)

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


def _vals(vals: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"value": vals}, schema={"value": pl.Float64})


# LCL across the pre (2012, both on standard tariff) and trial (2013) years: ToU cuts its peak, Std flat
_LCL_2Y = pl.DataFrame(
    {
        "stdorToU": ["ToU", "ToU", "Std", "Std"],
        "DateTime": [
            "2012-06-01 18:00:00.0000000",
            "2013-06-01 18:00:00.0000000",
            "2012-06-01 18:00:00.0000000",
            "2013-06-01 18:00:00.0000000",
        ],
        "value": [5.0, 3.0, 4.0, 4.0],  # ToU 5->3 (cut 2); Std 4->4 (flat)
    }
)
# NEED with a control group + a secular trend the DiD nets out
_NEED_DID = pl.DataFrame(
    {
        "LOFT_FLAG": [1, 1, 0, 0],
        "Econ2010": [5000.0, 6000.0, 5500.0, 6500.0],  # treated pre / control pre
        "Econ2013": [
            4000.0,
            4500.0,
            5400.0,
            6300.0,
        ],  # treated post (−1250) / control post (−150 trend)
    }
)


class _StubLCL2Y(LCLTariffAdapter):
    def _read(self) -> pl.DataFrame:
        return _LCL_2Y


class _StubNEEDDID(NEEDRetrofitAdapter):
    def _read(self) -> pl.DataFrame:
        return _NEED_DID


class TestDiD:
    """Difference-in-differences nets out the treated/control baseline gap a plain Δ carries."""

    def test_did_effect_removes_the_selection_bias(self):
        # treated cut peak 5->3 (true effect -2); control flat 4->4, but the groups differ at baseline
        tp_pre, tp_post, cp_pre, cp_post = _vals([5.0]), _vals([3.0]), _vals([4.0]), _vals([4.0])
        assert did_effect(tp_pre, tp_post, cp_pre, cp_post, metric="peak") == pytest.approx(-2.0)
        # the naive treated−control(post) estimand is biased by the 5-vs-4 selection gap:
        naive = measured_question("tariff", cp_post, tp_post, cell="c", metric="peak").true_delta
        assert naive == pytest.approx(-1.0)  # wrong: −1 vs the true −2

    def test_did_question_rejects_an_empty_group(self):
        with pytest.raises(ValueError, match="empty"):
            did_question(
                "tariff",
                _vals([]),
                _vals([3.0]),
                _vals([4.0]),
                _vals([4.0]),
                cell="c",
                metric="peak",
            )

    def test_did_question_scales_on_the_treated_pre_spread(self):
        # only treated_pre is multi-valued (std = sqrt(2)); the rest are single -> std None -> 1.0,
        # so this pins that scale comes from treated_pre specifically, not another group
        q = did_question(
            "tariff",
            _vals([2.0, 4.0]),
            _vals([1.0]),
            _vals([9.0]),
            _vals([9.0]),
            cell="c",
            metric="peak",
        )
        assert q.scale == pytest.approx(2.0**0.5)

    def test_lcl_did_profiles_feed_a_debiased_question(self):
        tp_pre, tp_post, cp_pre, cp_post = _StubLCL2Y("x").did_profiles("c")
        q = did_question("tariff", tp_pre, tp_post, cp_pre, cp_post, cell="c", metric="peak")
        assert q.intervention.kind == "tariff" and q.true_delta == pytest.approx(
            -2.0
        )  # (3-5) - (4-4)

    def test_need_did_split_nets_out_the_secular_trend(self):
        tp_pre, tp_post, cp_pre, cp_post = _StubNEEDDID("x").did_split("c")
        q = did_question("retrofit", tp_pre, tp_post, cp_pre, cp_post, cell="c", metric="mean")
        # (4250 - 5500) - (5850 - 6000) = -1250 - (-150) = -1100 (vs the biased naive post−pre = -1250)
        assert q.true_delta == pytest.approx(-1100.0)


# two ToU days at 18:00 (hh-of-day 36) avg to 5; one at 03:00 (hh-of-day 6); over two years for DiD
_LCL_DIURNAL = pl.DataFrame(
    {
        "stdorToU": ["ToU", "ToU", "ToU", "ToU", "Std", "Std"],
        "DateTime": [
            "2012-06-01 18:00:00.0000000",
            "2012-06-02 18:00:00.0000000",  # ToU 2012 18:00 -> mean(6,4)=5
            "2013-06-01 18:00:00.0000000",
            "2013-06-02 18:00:00.0000000",  # ToU 2013 18:00 -> mean(2,4)=3 (cut 2)
            "2012-06-01 18:00:00.0000000",
            "2013-06-01 18:00:00.0000000",  # Std flat 4->4
        ],
        "value": [6.0, 4.0, 2.0, 4.0, 4.0, 4.0],
    }
)


class TestDiurnal:
    """Diurnal (half-hour-of-day) profiles average across days into a typical day."""

    def test_profile_averages_across_days_and_buckets_half_hours(self):
        raw = pl.DataFrame(
            {
                "stdorToU": ["ToU", "ToU", "ToU", "ToU"],
                "DateTime": [
                    "2012-06-01 18:00:00.0000000",
                    "2012-06-02 18:00:00.0000000",  # 18:00 -> hod 36, mean(6,4)=5
                    "2012-06-01 18:30:00.0000000",  # 18:30 -> hod 37 (the minute//30 branch)
                    "2012-06-01 03:00:00.0000000",  # 03:00 -> hod 6
                ],
                "value": [6.0, 4.0, 7.0, 2.0],
            }
        )
        prof = lcl_diurnal_profile(raw, "ToU", cell="c").sort("hod")
        assert prof["hod"].to_list() == [6, 36, 37]  # incl. the :30 half-hour bucket
        assert prof["value"].to_list() == [2.0, 5.0, 7.0]

    def test_year_filter_excludes_the_other_year(self):
        prof = lcl_diurnal_profile(_LCL_DIURNAL, "ToU", cell="c", year=2012)
        assert prof["value"].to_list() == [
            5.0
        ]  # only the two 2012 days (mean 5), not the 2013 rows

    def test_did_uses_typical_day_peaks(self):
        args = [
            lcl_diurnal_profile(_LCL_DIURNAL, g, cell="c", year=y)
            for g, y in (("ToU", 2012), ("ToU", 2013), ("Std", 2012), ("Std", 2013))
        ]
        q = did_question("tariff", *args, cell="c", metric="peak")
        assert q.true_delta == pytest.approx(-2.0)  # (3-5) - (4-4), peaks from averaged days
