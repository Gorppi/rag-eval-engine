import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from src.config import settings
from src.generation.rag import answer_question

DATASET = json.loads(Path("evals/dataset.json").read_text(encoding="utf-8"))

judge_model = OllamaModel(
    model=settings.chat_model,
    base_url="http://localhost:11434",
)


@pytest.mark.parametrize("case", DATASET)
def test_rag_quality(case: dict) -> None:
    result = answer_question(case["question"], top_k=5)

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=result["answer"],
        expected_output=case["expected_output"],
        retrieval_context=result["contexts"],
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.7, model=judge_model, include_reason=True
    )
    relevancy = AnswerRelevancyMetric(
        threshold=0.7, model=judge_model, include_reason=True
    )

    assert_test(test_case, [faithfulness, relevancy])