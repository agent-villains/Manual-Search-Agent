import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "gemini-2.0-flash-001")

# ENV에서 RAG 코퍼스 ID 읽기
RAG_ISA = os.getenv("RAG_ISA")
RAG_NON = os.getenv("RAG_NON")          # 비대면제휴업무
RAG_OTHER = os.getenv("RAG_OTHER")      # 기타매매
RAG_DEPOSIT = os.getenv("RAG_DEPOSIT")  # 입출금관리
RAG_BRANCH = os.getenv("RAG_BRANCH")    # 창구관리및책임자거래업무
RAG_ALL = os.getenv("RAG_ALL")

VECTOR_DISTANCE_THRESHOLD = 0.65
TOP_K = 10

# -------------------------------------------------------------------
# 1) 카테고리별 RAG Tool 정의
# -------------------------------------------------------------------

ask_rag_isa = VertexAiRagRetrieval(
    name="ask_rag_isa",
    description=(
        "ISA 관련 문서(개설, 이전, 중도인출, 만기, 연금전환 등)에서만 근거를 조회합니다. "
        "category가 'ISA'일 때만 호출해야 합니다. 다른 category에서는 절대 호출하지 마세요."
    ),
    rag_resources=[rag.RagResource(rag_corpus=RAG_ISA)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

ask_rag_non = VertexAiRagRetrieval(
    name="ask_rag_non",
    description=(
        "비대면 제휴업무(파운트, TOSS, PASS, 프리즘 등 제휴 서비스) 관련 문서에서만 근거를 조회합니다. "
        "category가 '비대면제휴업무'일 때만 호출해야 합니다."
    ),
    rag_resources=[rag.RagResource(rag_corpus=RAG_NON)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

ask_rag_other = VertexAiRagRetrieval(
    name="ask_rag_other",
    description=(
        "기타매매(소수점 거래, 프로그램매매, 정기투자 등) 관련 문서에서만 근거를 조회합니다. "
        "category가 '기타매매'일 때만 호출해야 합니다."
    ),
    rag_resources=[rag.RagResource(rag_corpus=RAG_OTHER)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

ask_rag_deposit = VertexAiRagRetrieval(
    name="ask_rag_deposit",
    description=(
        "입출금·이체·수표 및 자금관리 관련 문서에서만 근거를 조회합니다. "
        "특히 잘못 송금, 오송금, 착오송금, 수취인 불명, 계좌 오류, 자금 반환 요청 등은 "
        "모두 이 도구를 사용해야 합니다. "
        "category가 '입출금관리'일 때만 호출해야 합니다."
    ),
    rag_resources=[rag.RagResource(rag_corpus=RAG_DEPOSIT)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

ask_rag_branch = VertexAiRagRetrieval(
    name="ask_rag_branch",
    description=(
        "창구관리 및 책임자 거래·전결규정·일일마감 등 내부 관리 문서에서만 근거를 조회합니다. "
        "category가 '창구관리및책임자거래업무'일 때만 호출해야 합니다."
    ),
    rag_resources=[rag.RagResource(rag_corpus=RAG_BRANCH)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

# fallback: 카테고리 애매하거나 근거 부족할 때만 사용
ask_rag_all = VertexAiRagRetrieval(
    name="ask_rag_all",
    description=(
        "전체 업무 매뉴얼(ALL corpus)에서 폭넓게 근거를 조회합니다. "
        "카테고리가 애매하거나 해당 코퍼스에서 근거가 부족한 경우에만 사용하세요. "
        "category가 명확히 주어졌는데도 다른 도구로 충분한 답변이 가능하면, "
        "이 도구는 호출하지 마세요."
    ),
    rag_resources=[rag.RagResource(rag_corpus=RAG_ALL)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

# -------------------------------------------------------------------
# 2) summary_router 결과(category_json)를 이용하는 단일 RAG Agent
# -------------------------------------------------------------------

rag_agent = Agent(
    name="summary_routed_rag_agent",
    model=MODEL_ID,
    description=(
        "Query Normalizer + Summary Router 결과를 참고해서 "
        "카테고리별 전용 RAG 코퍼스에서만 검색해 답변을 만드는 에이전트"
    ),
    instruction="""
너는 금융회사 내부 업무 매뉴얼을 기반으로 답변하는 RAG 에이전트다.

이 에이전트가 호출되기 전에 두 개의 에이전트가 이미 실행되었다:
1) Query Normalizer
2) Summary Router

앞 단계에서 남겨둔 상태는 다음과 같다.

- Query Normalizer 결과(JSON 문자열 또는 객체): {normalized_query_json}
- Summary Router 결과(JSON 문자열 또는 객체): {category_json}

***아주 중요: 아래 규칙은 시스템 규칙이며, 반드시 지켜야 한다. 어기면 에러로 간주된다.***

[1단계] normalized_query 추출
- normalized_query_json 안에서 "normalized_query" 필드를 찾아라.
- 있으면 이를 **실제 검색/답변에 사용할 질문**으로 쓴다.
- 없으면 사용자의 원래 질문을 그대로 사용해도 된다.

[2단계] category 확인
- category_json 안에서 "category" 필드를 찾아라.
- 값은 다음 중 하나일 것이다.
  - "ISA"
  - "비대면제휴업무"
  - "기타매매"
  - "입출금관리"
  - "창구관리및책임자거래업무"

지금 category_json 값에 따라 **반드시 아래 규칙**을 따른다.

[3단계] 카테고리 → RAG 도구 매핑 (하드 룰)

- category = "ISA"일 때만 → 도구 `ask_rag_isa`를 사용할 수 있다.
  - 이 경우 `ask_rag_isa`를 사용하여 검색해야 한다.
  - `ask_rag_deposit`, `ask_rag_non`, `ask_rag_other`, `ask_rag_branch`는 절대 호출하지 말 것.

- category = "비대면제휴업무"일 때만 → 도구 `ask_rag_non`을 사용할 수 있다.
  - 이 경우 `ask_rag_non`을 사용하여 검색해야 한다.
  - `ask_rag_isa`, `ask_rag_deposit`, `ask_rag_other`, `ask_rag_branch`는 호출하지 말 것.

- category = "기타매매"일 때만 → 도구 `ask_rag_other`를 사용할 수 있다.
  - 이 경우 `ask_rag_other`를 사용하여 검색해야 한다.
  - 이 경우 `ask_rag_isa`, `ask_rag_non`, `ask_rag_deposit`, `ask_rag_branch`는 절대 호출하지 말 것.

- category = "입출금관리"일 때만 → 도구 `ask_rag_deposit`을 사용할 수 있다.
  - 이 경우 반드시 `ask_rag_deposit`을 사용해 검색해야 한다.
  - 이 경우 `ask_rag_isa`, `ask_rag_non`, `ask_rag_other`, `ask_rag_branch`는 절대 호출하지 말 것.
  - 잘못 송금, 오송금, 착오송금, 수취인 불명, 계좌 오류, 자금 반환 문의는 모두 `ask_rag_deposit`으로 처리한다.

- category = "창구관리및책임자거래업무"일 때만 → 도구 `ask_rag_branch`를 사용할 수 있다.
  - 이 경우 반드시 'ask_rag_branch'를 사용하여 검색해야 한다.
  - 이 경우 'ask_rag_isa', 'ask_rag_non', 'ask_rag_other', 'ask_rag_deposit'는 절대 호출하지 말 것.
  - 창구관리 및 책임자 거래·전결규정·일일마감 등 내부 관리 문서에서만 근거를 조회해야 한다.


  명확한 category가 있으면 그 해당 카테고리 도구를 사용해야 한다.

[4단계] 답변 생성

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
- category가 "입출금관리"인데 `ask_rag_isa` 등 다른 카테고리 도구를 호출하지 말 것.


    [4-1단계] Hallucination 방지 규칙

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
        ask_rag_isa,
        ask_rag_non,
        ask_rag_other,
        ask_rag_deposit,
        ask_rag_branch,
        # ask_rag_all,
    ],
)

__all__ = [
    "rag_agent",
    "ask_rag_isa",
    "ask_rag_non",
    "ask_rag_other",
    "ask_rag_deposit",
    "ask_rag_branch",
    # "ask_rag_all",
]
