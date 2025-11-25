from google.adk.agents.llm_agent import Agent
from title_router.router import route_by_title


def route_direct(query: str):
    return route_by_title(query)


root_agent = Agent(
    model="gemini-2.5-flash",
    name="title_router_agent",
    description="질문을 가장 유사한 중분류 제목으로 라우팅하는 에이전트",
    instruction="""
당신은 LLM이 아니라 Python 로직을 실행하는 시스템입니다.

사용자 질문을 입력받으면,
route_direct(question)을 호출하여 나온 JSON을 그대로 출력하세요.

절대로 설명을 추가하지 말고 JSON만 출력하세요.
"""
)


def __call__(query: str):
    return route_direct(query)


__all__ = ["root_agent"]
