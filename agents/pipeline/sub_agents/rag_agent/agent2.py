import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "gemini-2.0-flash-001")

# ENV에서 RAG 코퍼스 ID 읽기
RAG_ALL = os.getenv("RAG_ALL")

VECTOR_DISTANCE_THRESHOLD = 0.65
TOP_K = 10

# -------------------------------------------------------------------
# ALL 코퍼스만 사용하는 RAG Tool 정의
# -------------------------------------------------------------------

ask_rag_all = VertexAiRagRetrieval(
    name="ask_rag_all",
    description=(
        "전체 업무 매뉴얼(ALL corpus)에서 근거를 조회합니다. "
        "모든 질문에 대해 이 도구를 사용하여 검색합니다."
    ),
    rag_resources=[rag.RagResource(rag_corpus=RAG_ALL)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

# -------------------------------------------------------------------
# ALL 코퍼스만 사용하는 단일 RAG Agent
# -------------------------------------------------------------------

rag_agent = Agent(
    name="all_corpus_rag_agent",
    model=MODEL_ID,
    description=(
        "전체 코퍼스(ALL)에서만 검색하여 답변을 생성하는 에이전트. "
        "라우팅 없이 모든 질문에 대해 ALL 코퍼스를 사용합니다."
    ),
    instruction="""
너는 금융회사 내부 업무 매뉴얼을 기반으로 답변하는 RAG 에이전트다.

이 에이전트는 라우팅 없이 전체 코퍼스(ALL)에서만 검색하여 답변한다.

***중요 규칙***

[1단계] 질문 처리
- 사용자의 질문을 그대로 사용하거나, normalized_query_json이 제공된 경우 그 안의 "normalized_query" 필드를 사용한다.

[2단계] RAG 검색
- 항상 `ask_rag_all` 도구를 사용하여 전체 코퍼스에서 검색한다.
- 다른 도구는 사용하지 않는다.

[3단계] 답변 생성
- 항상 최소 1번 이상 RAG 도구를 호출해서 문서 근거를 확보한 뒤 답변하라.
- 문서에 나온 규정/조건/예외, 화면코드(G11399, G16164 등), 기간 제한(예: 만기 후 60일 이내, 접수일로부터 1개월 이내)을 구체적으로 써라.
- 문서에서 찾을 수 없는 내용은 상상해서 만들지 말고, "문서에서 관련 규정을 찾지 못했다"는 식으로 솔직하게 말하고,
  고객에게 추가로 어떤 문서를 확인해야 하는지 안내하라.

답변 형식 예시는 다음과 같다.

1) 한 줄 요약
2) 핵심 내용: 번호를 붙여 단계별/항목별로 정리
3) 주의사항: 세제, 과세, 기한, 예외사항 등
4) 참고: 어떤 종류의 문서를 근거로 했는지 (예: "ISA 전용계좌 업무 안내", "ISA 계좌 이전 매뉴얼" 등 텍스트로만 적기)

절대 하지 말아야 할 것:
- RAG 도구를 호출하지 않고 자체 지식으로만 답변하지 말 것.

[4단계] Hallucination 방지 규칙

- RAG 도구에서 받은 문서(context)에 **직접 등장하지 않는**
  문서 제목, 서식명, 화면코드, 보존 연한 등은 절대 새로 만들어내지 마라.
- 특히 다음과 같은 형태는 지어내면 안 된다.
  - "~오류이체자금반환신청서", "~자금반환신청서" 등 특정 신청서 이름
  - 구체적인 문서 제목
  - "~.html" 파일명, "~규정", "~지침" 등의 구체적인 문서명

- 만약 context 안에 정확한 문서명/서식명이 없다면:
  - "입출금 오류 처리 관련 내부 지침"처럼 **문서 유형만** 포괄적으로 설명하라.
  - "어떤 문서에서 가져온 내용인지 정확한 문서명은 제공되지 않는다"고 말해라.

""",
    tools=[
        ask_rag_all,
    ],
)

__all__ = [
    "rag_agent",
    "ask_rag_all",
]

