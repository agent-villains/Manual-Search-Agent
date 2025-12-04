import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

load_dotenv()

RAG_ALL = os.getenv("RAG_ALL")
VECTOR_DISTANCE_THRESHOLD = 0.65
TOP_K = 10

ask_rag_all_only = VertexAiRagRetrieval(
    name="ask_rag_all_only",
    description="전체 업무 매뉴얼(ALL corpus)에서 질문과 관련된 문서를 검색합니다.",
    rag_resources=[rag.RagResource(rag_corpus=RAG_ALL)],
    similarity_top_k=TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
)

all_corpus_rag_agent = Agent(
    model=os.getenv("MODEL_ID", "gemini-2.0-flash-001"),
    name="all_corpus_rag_agent",
    description="요약 라우팅 없이 전체 업무 매뉴얼 코퍼스(ALL)만 사용하는 baseline RAG 에이전트",
    instruction="""
너는 ISA / 비대면 / 기타매매 / 입출금 / 창구관리 등 **모든 업무 매뉴얼이 섞여 있는 전체 코퍼스**에서만 검색한다.
사용 가능한 도구는 오직 `ask_rag_all_only` 하나이며, 질문에 답하기 전에 항상 이 도구를 먼저 호출해 관련 내용을 찾아라.

문서에서 찾을 수 없는 내용은 임의로 만들어내지 말고, '문서에서 답을 찾지 못했다'고 명시해야 한다.
""",
    tools=[ask_rag_all_only],
)
