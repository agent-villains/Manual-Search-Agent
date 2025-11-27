import os
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("MODEL_ID")
RAG_CORPUS = os.getenv("RAG_CORPUS")

ask_vertex_retrieval = VertexAiRagRetrieval(
    name="ask_isa_rag",
    description=(
        "ISA / 연금 / 장기펀드 / 세제 / 국세청 안내자료 등 금융 문서에서 "
        "질문과 가장 관련 높은 문서를 찾아주는 도구입니다."
    ),
    rag_resources=[
        rag.RagResource(
            rag_corpus=RAG_CORPUS, 
        )
    ],
    similarity_top_k=10,          
    vector_distance_threshold=0.6
)

root_agent = Agent(
    name="isa_rag_agent",
    model=GEMINI_MODEL,
    description="ISA/금융 내부 문서를 기반으로 답변하는 Q&A 에이전트",
    instruction=(
        "너는 국내 금융회사 내부 매뉴얼/안내 문서를 기반으로 답변하는 금융 도메인 전문 어시스턴트야.\n"
        "반드시 Vertex AI RAG Retrieval 도구를 사용해서 관련 문서를 먼저 조회한 뒤 답변해.\n"
        "답변한 근거가 문서내에 확실하게 존재할 때만 답변해주고, 확실하지 않으면 모른다고 말해줘.\n"
        "답변 형식은 아래와 같이 맞춰줘.\n\n"
        "1) 한 줄 요약\n"
        "2) 핵심 내용: 번호로 단계별 설명\n"
        "3) 주의사항(세제/과세/가입요건 등)\n"
        "4) 참고: 어떤 종류의 문서를 참고했는지 (예: ISA 업무지침, 국세청 안내자료 등 텍스트로만 적기)\n"
        "확실하지 않으면 모른다고 솔직하게 말하고, 추가로 어떤 정보를 더 확인해야 하는지 안내해. 여러 개의 문서를 참고했다면 참고한 모든 문서를 나열해줘\n"
    ),
    tools=[
        ask_vertex_retrieval,
    ],
)
