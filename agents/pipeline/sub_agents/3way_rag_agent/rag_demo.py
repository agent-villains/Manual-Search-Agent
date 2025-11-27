import os

import vertexai
from google import genai
from vertexai import rag
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from dotenv import load_dotenv

# 환경변수 로드

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

MODEL_ID = os.getenv("MODEL_ID")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# 다양한 CORPUS로 테스트 가능
CORPUS_ADOCTOHTML = os.getenv("CORPUS_ADOCTOHTML")
CORPUS_RAWHTML = os.getenv("CORPUS_RAWHTML")
CORPUS_CLEANHTML = os.getenv("CORPUS_CLEANHTML")

GCS_BUCKET_ADOCTOHTML = os.getenv("GCS_BUCKET_ADOCTOHTML")
GCS_BUCKET_RAWHTML = os.getenv("GCS_BUCKET_RAWHTML")
GCS_BUCKET_CLEANHTML = os.getenv("GCS_BUCKET_CLEANHTML")

# vertex 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")


# RAG Corpus 생성

rag_corpus = rag.create_corpus(
    display_name=CORPUS_ADOCTOHTML,
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=EMBEDDING_MODEL
            )
        )
    ),
)

print("\nRAG Corpus 생성 완료")
print("corpus_name:", rag_corpus.name)

print("\n현재 Corpus 목록:")
print(rag.list_corpora())

# GCS 버킷에서 파일 import

try:
    response = rag.import_files(
        corpus_name=rag_corpus.name,
        paths=[GCS_BUCKET_ADOCTOHTML],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
        ),
        max_embedding_requests_per_min=900,
    )
    print("response:", response)
except Exception as e:
    print("오류", e)


# 컨텍스트 retrieval 테스트

retrieval_response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=rag_corpus.name,
            # rag_file_ids=["..."]  # 특정 파일만 쓰고 싶을 때 id 넣기
        )
    ],
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=5,
        filter=rag.Filter(
            vector_distance_threshold=0.5,
        ),
    ),
    text="ISA 계좌 이전 조건이 뭐야?",
)

print("Retrieval 결과:",retrieval_response)


# Gemini + RAG Retrieval Tool로 최종 답변 생성

rag_retrieval_tool = Tool(
    retrieval=Retrieval(
        vertex_rag_store=VertexRagStore(
            rag_corpora=[rag_corpus.name],
            similarity_top_k=10,
            vector_distance_threshold=0.5,
        )
    )
)

gen_response = client.models.generate_content(
    model=MODEL_ID,
    contents="만 23세이고 근로소득은 5000만원이야. 가입 서류 뭐가 필요해?",
    config=GenerateContentConfig(tools=[rag_retrieval_tool]),
)

print("\n최종 답변 (Gemini + RAG):\n",gen_response.text)