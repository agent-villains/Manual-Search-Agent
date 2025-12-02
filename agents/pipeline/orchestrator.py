import json
from typing import Any, Dict

from .agent import routing_pipeline_agent, answer_pipeline_agent
from .sub_agents.final_category_agent.agent import final_category_agent


def _safe_load_json(maybe_json: str) -> Dict[str, Any]:
    """JSON 파싱 헬퍼: JSON이 아니면 단순 category 문자열로 래핑."""
    try:
        return json.loads(maybe_json)
    except Exception:
        return {
            "category": maybe_json,
            "need_clarification": False,
            "clarification_question": "",
            "candidate_categories": [],
        }


def start_question_flow(user_question: str) -> Dict[str, Any]:
    """
    전체 프로세스 1단계:
    - query_normalizer → summary_router → clarifier_agent 까지만 실행
    - 애매하면 clarification 질문과 상태(state)를 반환
    - 명확하면 바로 검색 + ANSWER까지 실행해서 최종 답변 반환
    """
    routing_resp = routing_pipeline_agent.run(user_question)

    normalized_query_json = getattr(routing_resp, "normalized_query_json", None)
    category_json_str = getattr(routing_resp, "category_json", "{}")
    category_data = _safe_load_json(category_json_str)

    need_clarification = category_data.get("need_clarification", False)

    if need_clarification:
        # 프론트/상위 서비스가 그대로 사용자에게 물어볼 수 있도록 질문과 상태를 반환
        return {
            "phase": "clarification",
            "clarification_question": category_data.get("clarification_question", ""),
            "candidate_categories": category_data.get("candidate_categories", []),
            "state": {
                "normalized_query_json": normalized_query_json,
                "candidate_categories": category_data.get("candidate_categories", []),
            },
        }

    # 애매하지 않은 경우: 바로 검색 + 최종 답변 생성
    answer_resp = answer_pipeline_agent.run(
        "",
        normalized_query_json=normalized_query_json,
        category_json=category_json_str,
    )

    final_answer_json = getattr(answer_resp, "final_answer", None)
    final_answer = {}
    if final_answer_json:
        try:
            final_answer = json.loads(final_answer_json)
        except Exception:
            final_answer = {"answer": str(final_answer_json), "source": None}

    return {
        "phase": "answer",
        "category_json": category_json_str,
        "answer": final_answer,
    }


def continue_with_clarification(user_reply: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    전체 프로세스 2단계:
    - 사용자가 clarification_question에 답변한 뒤 호출
    - final_category_agent로 최종 카테고리 하나 선택
    - answer_pipeline_agent로 검색 + 최종 답변 생성
    """
    normalized_query_json = state.get("normalized_query_json")
    candidate_categories = state.get("candidate_categories", [])

    # 최종 카테고리 결정
    final_category_resp = final_category_agent.run(
        user_reply,
        normalized_query_json=normalized_query_json,
        clarification_answer=user_reply,
        candidate_categories_json=json.dumps(candidate_categories, ensure_ascii=False),
    )
    final_category_json_str = getattr(final_category_resp, "category_json", "{}")

    # 최종 카테고리로 검색 + 답변 생성
    answer_resp = answer_pipeline_agent.run(
        "",
        normalized_query_json=normalized_query_json,
        category_json=final_category_json_str,
    )

    final_answer_json = getattr(answer_resp, "final_answer", None)
    final_answer = {}
    if final_answer_json:
        try:
            final_answer = json.loads(final_answer_json)
        except Exception:
            final_answer = {"answer": str(final_answer_json), "source": None}

    return {
        "phase": "answer",
        "category_json": final_category_json_str,
        "answer": final_answer,
    }


__all__ = ["start_question_flow", "continue_with_clarification"]


