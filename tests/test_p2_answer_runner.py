from researchpilot.answer_runner import ModelCompletion
from researchpilot.evaluation import ExperimentConfig
from researchpilot.p2_answer_runner import generate_p2_answer
from researchpilot.p2_runner import P2ExecutionConfig


class _Client:
    def complete(self, **kwargs):
        return ModelCompletion(
            '{"answer":"ok","citations":[],"claims":[],"refused":false,"refusal_reason":null}', 10, 3
        )


def _config():
    experiment = ExperimentConfig.from_dict(
        {
            "schema_version": "researchpilot_retrieval_experiment_v1",
            "name": "t",
            "description": "t",
            "ingestion": {
                "parser_name": "docling",
                "parser_version": "1",
                "enable_ocr": False,
                "chunker_version": "structure_aware_v1",
                "parent_chunk_size": 10,
                "parent_chunk_overlap": 0,
                "child_chunk_size": 10,
                "child_chunk_overlap": 0,
            },
            "models": {"embedding_model": "e", "reranker_model": None, "generation_model": None},
            "retrieval": {
                "strategy": "hybrid_rrf",
                "query_expansion": False,
                "query_expansion_count": 0,
                "dense_limit_per_query": 1,
                "bm25_limit_per_query": 1,
                "fusion": "rrf",
                "reranker": False,
                "result_limit": 1,
                "rrf_k": 60,
            },
        }
    )
    return P2ExecutionConfig(
        experiment,
        __import__("researchpilot.query_plan", fromlist=["P2RunMode"]).P2RunMode.SCOPED,
        1,
        1000,
        10,
        0.0,
        None,
        None,
        "u",
        "s",
    )


def test_empty_context_refuses_without_calling_model():
    outcome = generate_p2_answer(
        question="q", selected=(), required_paper_keys=("a", "b"), decision=None, config=_config(), client=_Client()
    )
    assert outcome.answer.refused is True
    assert outcome.prompt_tokens == 0
