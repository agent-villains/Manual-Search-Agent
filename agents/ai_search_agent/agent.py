import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import VertexAiSearchTool

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
MODEL_ID = os.getenv("MODEL_ID")
VERTEX_SEARCH_DATASTORE_ID = os.getenv("VERTEX_SEARCH_DATASTORE_ID")

vertex_search_tool = VertexAiSearchTool(
    data_store_id=VERTEX_SEARCH_DATASTORE_ID
)

root_agent = Agent(
    name="ai_search_agent",
    model=MODEL_ID,
    description=(
        "Vertex AI Search 데이터스토어를 사용해서 ISA/연금/세제/국세청 안내 등 "
        "금융 관련 문서를 검색하고, 관련 스니펫을 기반으로 답변하는 검색 에이전트입니다."
    ),
    instruction=(
        "너는 금융 도메인 내부 문서를 검색하는 전문 검색 에이전트야.\n"
        "반드시 Vertex AI Search 도구를 사용해서 관련 문서를 먼저 검색해.\n"
        "검색 결과의 스니펫(문장 조각)을 기반으로 사용자가 이해하기 쉽게 정리해서 답변해줘.\n"
        "검색 결과에서 정확한 근거를 찾지 못하면, 모른다고 말하고 어떤 문서를 추가로 확인해야 할지 안내해줘.\n\n"
        "답변 형식은 다음과 같이 맞춰줘.\n\n"
        "1) 한 줄 요약\n"
        "2) 핵심 내용: 번호 매겨 정리 (검색된 스니펫을 재구성해서 설명)\n"
        "3) 추가로 참고하면 좋은 키워드나 문서 유형 (예: ISA 업무지침, 국세청 안내자료, 상품설명서 등)\n"
    ),
    tools=[
        vertex_search_tool,
    ],
)
