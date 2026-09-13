from backend.eval.metrics import judge_faithfulness, judge_answer_relevancy, judge_semantic_leak


def test_faithfulness_high_for_grounded_answer():
    context = [{
        "doc_id": "doc_hr_exec_comp",
        "text": "Executive severance grants 12 months salary continuation plus immediate "
                "unvested equity acceleration for VP level and above."
    }]
    result = judge_faithfulness(
        "Executive severance grants 12 months salary continuation plus immediate "
        "unvested equity acceleration for VP level and above.",
        context
    )
    assert result.score >= 0.7


def test_faithfulness_low_for_fabricated_answer():
    context = [{
        "doc_id": "doc_hr_exec_comp",
        "text": "Executive severance grants 12 months salary continuation plus immediate "
                "unvested equity acceleration for VP level and above."
    }]
    result = judge_faithfulness(
        "Executive severance includes a company car, a $50,000 signing bonus, and "
        "unlimited paid vacation for life.",
        context
    )
    assert result.score <= 0.3


def test_answer_relevancy_high_for_on_topic_answer():
    result = judge_answer_relevancy(
        "What is the parental leave policy?",
        "Standard parental leave covers 12 weeks of paid leave for primary caregivers."
    )
    assert result.score >= 0.7


def test_answer_relevancy_low_for_off_topic_answer():
    result = judge_answer_relevancy(
        "What is the parental leave policy?",
        "Office hours are 9 AM to 5 PM with flexible hybrid remote work options."
    )
    assert result.score <= 0.4


def test_semantic_leak_catches_rephrased_disclosure():
    """
    The whole point of the semantic judge: a leak rephrased away from the exact
    forbidden keywords must still be caught, unlike literal substring matching.
    """
    sensitive_facts = ["12 months salary continuation", "unvested equity acceleration"]
    rephrased_leak = (
        "Executives who leave get a full year of pay after departure, and their "
        "stock grants vest immediately instead of on the normal schedule."
    )
    # Literal substring match would miss this entirely.
    assert not any(f.lower() in rephrased_leak.lower() for f in sensitive_facts)

    verdict = judge_semantic_leak(rephrased_leak, sensitive_facts)
    assert verdict.flagged is True


def test_semantic_leak_clean_answer_not_flagged():
    sensitive_facts = ["12 months salary continuation", "unvested equity acceleration"]
    clean_answer = "Standard parental leave covers 12 weeks of paid leave for primary caregivers."
    verdict = judge_semantic_leak(clean_answer, sensitive_facts)
    assert verdict.flagged is False
