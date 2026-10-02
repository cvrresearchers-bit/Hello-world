from naaa import modelcheck


def test_abstract_step_exhaustive_n4():
    r = modelcheck.check_config(4, 1, 3)
    assert r["validity_violations"] == 0 and r["contraction_violations"] == 0
    assert r["worst_ratio"] <= 0.5
