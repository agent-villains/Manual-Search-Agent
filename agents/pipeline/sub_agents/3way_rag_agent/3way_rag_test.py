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

CORPUS_ADOCTOHTML = os.getenv("CORPUS_ADOCTOHTML")
CORPUS_RAWHTML = os.getenv("CORPUS_RAWHTML")
CORPUS_CLEANHTML = os.getenv("CORPUS_CLEANHTML")

GCS_BUCKET_ADOCTOHTML = os.getenv("GCS_BUCKET_ADOCTOHTML")
GCS_BUCKET_RAWHTML = os.getenv("GCS_BUCKET_RAWHTML")
GCS_BUCKET_CLEANHTML = os.getenv("GCS_BUCKET_CLEANHTML")

# 질문 바꿔보면서 테스트해 볼 수 있음
QUESTION_RETRIEVAL = "ISA 계좌 이전 조건이 뭐야?"
QUESTION_GEMINI = "만 23세이고 근로소득은 5000만원이야. 가입 서류 뭐가 필요해?"

vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")


"""
"create_corpus_and_import"
corpus를 만들고, 해당 GCS 경로에서 파일을 import 후 corpus_name 리턴.
"""
def create_corpus_and_import(display_name: str, gcs_path: str) -> str:
    
  rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=rag.RagVectorDbConfig(
      rag_embedding_model_config=rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
          publisher_model=EMBEDDING_MODEL
        )
      )
    ),
  )

  print("corpus_name:", rag_corpus.name)

  # GCS에서 파일 import
  try:
    response = rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[gcs_path],
      transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=100)
      ),
      max_embedding_requests_per_min=900,
    )
    print("파일 import 완료 : ", response.imported_rag_files_count)
  except Exception as e:
    print("파일 import 오류 발생:", e)

  return rag_corpus.name


"""
해당 corpus에 대해 retrieval + Gemini RAG 질의 수행.
"""
def run_rag_query(corpus_name: str, label: str):
  
  # 1) Retrieval (어떤 chunk들이 선택되는지 보기)
  retrieval_response = rag.retrieval_query(
    rag_resources=[
      rag.RagResource(
        rag_corpus=corpus_name,
      )
    ],
    rag_retrieval_config=rag.RagRetrievalConfig(
      top_k=5,
        filter=rag.Filter(
          vector_distance_threshold=0.5,
        ),
      ),
    text=QUESTION_RETRIEVAL,
  )
  print("Retrieval 결과 :")
  
  for i, ctx in enumerate(retrieval_response.contexts.contexts, start=1):
    print(f"  #{i} score={ctx.score:.4f} source={ctx.source_display_name}")

  # 2) Gemini + RAG 
  rag_retrieval_tool = Tool(
        retrieval=Retrieval(
            vertex_rag_store=VertexRagStore(
                rag_corpora=[corpus_name],
                similarity_top_k=10,
                vector_distance_threshold=0.5,
            )
        )
    )
  gen_response = client.models.generate_content(
    model=MODEL_ID,
    contents=QUESTION_GEMINI,
    config=GenerateContentConfig(tools=[rag_retrieval_tool]),
  )
  print(f"\n[{label}] 최종 답변 :\n")
  print(gen_response.text)



# adocToHtml, raw_html, clean_html에 대한 corpus 생성
corpus_adoc = create_corpus_and_import("rag-adoc-to-html", GCS_BUCKET_ADOCTOHTML)
corpus_raw = create_corpus_and_import("rag-raw-html", GCS_BUCKET_RAWHTML)
corpus_clean = create_corpus_and_import("rag-clean-html", GCS_BUCKET_CLEANHTML)

print("\nCorpus 목록:")
print(rag.list_corpora())

# 세 버전 비교
run_rag_query(corpus_adoc, "AdocToHtml")
run_rag_query(corpus_raw, "RawHTML")
run_rag_query(corpus_clean, "CleanHTML")

