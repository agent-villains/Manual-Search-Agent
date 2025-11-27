from google.adk.agents import SequentialAgent

from .sub_agents.query_normalizer_agent.agent import query_agent
from .sub_agents.title_router.agent import router_agent

pipeline_agent = SequentialAgent(
    name="pipeline_agent",
    description="사용자 질문을 정규화 → 라우팅 생성까지 순차 처리하는 파이프라인 에이전트",
    sub_agents=[
        query_agent,
        router_agent,
    ]
)

root_agent = pipeline_agent
