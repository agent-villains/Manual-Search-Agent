from google.adk.agents import SequentialAgent

from .sub_agents.query_normalizer_agent.agent import query_agent
# from .sub_agents.title_router.agent import router_agent
from .sub_agents.ai_search_agent.agent import root_agent as ai_search_agent
from .sub_agents.answer_agent.agent import answer_agent
# from .sub_agents.keyword_router.agent import keyword_router
from .sub_agents.summary_router.agent import summary_router
from .sub_agents.clarifier_agent.agent import clarifier_agent

"""
1단계 파이프라인: 카테고리 결정/clarification 까지만 수행
- 사용자 질문 정규화 (query_agent)
- 1차 카테고리 라우팅 (summary_router)
- 애매하면 재질문 설계 및 최종 카테고리 제안 (clarifier_agent)

이 단계에서는 실제로 RAG 검색이나 최종 답변(answer_agent)을 수행하지 않습니다.
프론트엔드/상위 서비스는 여기서 나온 category_json의
- need_clarification == true 이면: clarification_question을 사용자에게 보여주고 답을 받은 뒤
- false 이면: 바로 2단계(답변 생성 파이프라인)를 호출하면 됩니다.
"""
routing_pipeline_agent = SequentialAgent(
    name="routing_pipeline_agent",
    description=(
        "사용자 질문을 정규화 → 카테고리 라우팅 → (필요 시) 카테고리 재질문 설계까지만 수행하는 파이프라인"
    ),
    sub_agents=[
        query_agent,
        summary_router,
        clarifier_agent,
    ],
)

"""
2단계 파이프라인: 최종 카테고리가 정해진 뒤 실제 검색/답변 생성
- 입력 컨텍스트에는 normalized_query_json, category_json 이 모두 있어야 합니다.
- routing_pipeline_agent 실행 + (필요시) 사용자 재질문/선택 이후에 호출하는 용도입니다.
"""
answer_pipeline_agent = SequentialAgent(
    name="answer_pipeline_agent",
    description=(
        "최종 결정된 카테고리와 정규화된 쿼리를 기반으로 Vertex AI Search 검색과 최종 답변을 생성하는 파이프라인"
    ),
    sub_agents=[
        ai_search_agent,
        answer_agent,
    ],
)

# 기본 root_agent는 1단계(카테고리 결정/clarification) 파이프라인으로 둡니다.
root_agent = routing_pipeline_agent
