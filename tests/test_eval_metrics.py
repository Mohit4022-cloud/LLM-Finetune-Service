from llm_finetune_service.eval.metrics import score_prediction


def test_eval_scores_are_bounded():
    scores = score_prediction(
        source_email="Please be advised that the rollout is delayed until Friday.",
        prediction="Quick update: rollout is delayed until Friday.",
        reference="Quick update: rollout is delayed until Friday.",
    )
    for key, value in scores.items():
        if key == "compression_ratio":
            assert value >= 0
        else:
            assert 0 <= value <= 1
