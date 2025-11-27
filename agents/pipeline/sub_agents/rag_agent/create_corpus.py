import os

import vertexai
from google import genai
from vertexai import rag
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from dotenv import load_dotenv

# 환경변수 로드

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GEMINI_MODEL = os.getenv("MODEL_ID")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CORPUS_RELATED = os.getenv("CORPUS_RELATED")
GCS_BUCKET_RELATED = os.getenv("GCS_BUCKET_RELATED")
GCS_BUCKET_CLEANHTML = os.getenv("GCS_BUCKET_CLEANHTML")


# vertex 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")


# RAG Corpus 생성

rag_corpus = rag.create_corpus(
    display_name=CORPUS_RELATED,
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
        paths=[GCS_BUCKET_RELATED, GCS_BUCKET_CLEANHTML],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
        ),
        max_embedding_requests_per_min=900,
    )
    print("response:", response)
except Exception as e:
    print("오류", e)

