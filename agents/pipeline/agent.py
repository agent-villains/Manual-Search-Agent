from google.adk.agents import SequentialAgent

from .sub_agents.query_normalizer_agent.agent import query_agent
# from .sub_agents.title_router.agent import router_agent
from .sub_agents.ai_search_agent.agent import root_agent as ai_search_agent
from .sub_agents.answer_agent.agent import answer_agent
# from .sub_agents.keyword_router.agent import keyword_router
from .sub_agents.summary_router.agent import summary_router
pipeline_agent = SequentialAgent(
    name="pipeline_agent",
    description="사용자 질문을 정규화 → 라우팅 생성, ai search를 이용한 답변 생성까지 순차 처리하는 파이프라인 에이전트",
    sub_agents=[
        query_agent,
        summary_router,
        ai_search_agent,
        answer_agent
    ]
)

root_agent = pipeline_agent
