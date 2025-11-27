from google.adk.agents.llm_agent import Agent

query_agent = Agent(
    model='gemini-2.5-flash',
    name='query_normalizer_agent',
    description='사용자 질문을 검색 친화적 형태로 확장하는 Query Normalizer',
    instruction="""
당신은 Query Normalizer Agent입니다.
입력된 자연어 질문을 검색에 최적화된 형태로 바꿉니다.

출력은 반드시 JSON 형식이어야 합니다.
{
  "normalized_query": "...",
  "keywords": ["...", "..."],
  "expansion_reason": "..."
}
텍스트 설명은 JSON 밖에 절대 추가하지 마세요.
"""
)
