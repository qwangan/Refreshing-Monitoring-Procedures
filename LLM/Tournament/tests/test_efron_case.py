from __future__ import annotations

import numpy as np

import efron_contract
import run_efron_case


def test_locked_source_and_hwhwh_boundaries():
    text = efron_contract.validate_source()
    assert len(text.split()) == 365
    blocks, horizon = run_efron_case._build_plan_from_source_ids(
        None,
        list(range(443)),
        watermark_tokens=100,
        minimum_human_tokens=81,
    )
    assert horizon == 443
    assert tuple(block["length"] for block in blocks) == (81, 100, 81, 100, 81)
    assert tuple(
        (block["start"], block["end"])
        for block in blocks
        if block["kind"] == "watermarked"
    ) == ((82, 181), (263, 362))


def test_report_union_uses_one_based_closed_intervals():
    report = type("Report", (), {"interval_start": 2, "interval_end": 4})()
    assert run_efron_case.report_union_mask([report], 6).tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_locked_threshold_and_general_bound():
    assert run_efron_case.THRESHOLD_EXACT == 49
    assert (1.0 + np.log(49.0)) / 49.0 < 0.10

