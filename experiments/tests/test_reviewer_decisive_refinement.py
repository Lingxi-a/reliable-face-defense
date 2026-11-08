from paper_exp.reviewer_decisive_refinement import ReviewerDecisiveRefiner


def _record(**updates):
    checks = {
        "worst_similarity_ok": True,
        "conditionwise_no_regression": True,
        "visual_no_regression": True,
    }
    checks.update(updates)
    return {"checks": checks}


def test_gate_hierarchy():
    assert ReviewerDecisiveRefiner._passes_gate(_record(), "g0_ungated")
    assert ReviewerDecisiveRefiner._passes_gate(_record(), "g1_worst_gain")
    assert ReviewerDecisiveRefiner._passes_gate(_record(), "g2_conditionwise")
    assert ReviewerDecisiveRefiner._passes_gate(_record(), "g3_full")

    weak = _record(worst_similarity_ok=False)
    assert ReviewerDecisiveRefiner._passes_gate(weak, "g0_ungated")
    assert not ReviewerDecisiveRefiner._passes_gate(weak, "g1_worst_gain")

    condition = _record(conditionwise_no_regression=False)
    assert ReviewerDecisiveRefiner._passes_gate(condition, "g1_worst_gain")
    assert not ReviewerDecisiveRefiner._passes_gate(condition, "g2_conditionwise")

    visual = _record(visual_no_regression=False)
    assert ReviewerDecisiveRefiner._passes_gate(visual, "g2_conditionwise")
    assert not ReviewerDecisiveRefiner._passes_gate(visual, "g3_full")

