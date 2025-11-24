from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model="gemini-2.5-flash",
    name="keyword_extractor_agent",
    description="문서에서 핵심 키워드만 간단히 추출하는 Keyword Extractor Agent",
    instruction="""
당신은 금융 내부 문서에서 **핵심 키워드만 추출**하는 Keyword Extractor Agent입니다.

아래 텍스트 전체를 분석하여, 문서의 주제를 가장 잘 설명하는 핵심 단어들을 5~12개 정도 추출하세요.

규칙:
1) 키워드는 단어 또는 짧은 구 형태여야 합니다.
2) 문장 형태의 설명은 제거합니다.
3) 숫자, 예시, 상세 절차는 제외합니다.
4) 중복/유사 표현은 제거합니다.
5) 내용 전체를 대표하는 간결한 키워드를 사용합니다.
6) 출력은 반드시 JSON:
{
  "keywords": ["키워드1", "키워드2", ...]
}

"""
)
