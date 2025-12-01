import os
import sys
import time
import importlib
import vertexai
from google import genai
from vertexai import rag
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from dotenv import load_dotenv

# -----------------------------
#   코드실행
#   python multi_test.py <생성할 로그 파일 이름> <GROUND_TRUTH_DATA를 포함한 파일 이름>
#   ex ) python multi_test.py rag_test1.log test1_GT
#        python multi_test.py rag_test1.log test1_GT.py
# -----------------------------


if len(sys.argv) < 3:
    print("예시: python multi_test.py rag_test1.log test1_GT")
    sys.exit(1)

LOG_FILE = sys.argv[1]
GT_MODULE_NAME = sys.argv[2]

if GT_MODULE_NAME.endswith(".py"):
    GT_MODULE_NAME = GT_MODULE_NAME[:-3]

# import 및 데이터 로드
try:
    gt_module = importlib.import_module(GT_MODULE_NAME)
except ModuleNotFoundError as e:
    print(f"찾을 수 없음")
    raise e

if not hasattr(gt_module, "GROUND_TRUTH_DATA"):
    raise AttributeError(f"GROUND_TRUTH_DATA 변수 없음")

GROUND_TRUTH_DATA = gt_module.GROUND_TRUTH_DATA

# 환경변수 로드
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL_ID = os.getenv("MODEL_ID")
RAG_ALL_CORPUS = os.getenv("RAG_ALL_CORPUS")
ISA_CORPUS = os.getenv("RAG_ISA_CORPUS")

K_VALUE = 10
VECTOR_DISTANCE_THRESHOLD = 0.65

# Vertex 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

# 로그 파일
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"RAG 실험 로그 — log_file={LOG_FILE}, GT={GT_MODULE_NAME}.GROUND_TRUTH_DATA \n\n")

def log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Metric 계산 (calculate_metrics)
def calculate_metrics(contexts, ground_truth_sources, top_k):
    """
    검색 청크 목록을 분석하여
      - P5: 평균 점수
      - Hit@k
      - MRR
      - Noise Ratio
    계산
    """
    effective_k = len(contexts)
    retrieved_sources = [ctx.source_display_name for ctx in contexts]

    # 평균 점수
    total_score = sum(ctx.score for ctx in contexts)
    average_score = total_score / effective_k if effective_k else 0.0

    # Hit@k
    hit_at_k = 1 if any(src in retrieved_sources for src in ground_truth_sources) else 0

    # MRR
    reciprocal_rank = 0.0
    ranks = []
    for gt in ground_truth_sources:
        try:
            rank = retrieved_sources.index(gt) + 1
            ranks.append(rank)
        except ValueError:
            pass
    if ranks:
        reciprocal_rank = 1.0 / min(ranks)

    # Noise Ratio
    relevant_chunks = sum(
        1 for ctx in contexts if ctx.source_display_name in ground_truth_sources
    )
    noise_ratio = (effective_k - relevant_chunks) / effective_k if effective_k else 1.0

    return average_score, hit_at_k, reciprocal_rank, noise_ratio


# 단일 코퍼스 테스트 함수 (measure_retrieval_performance)
def measure_retrieval_performance(corpus_arn: str, label: str, question: str, ground_truth_sources: list):

    all_contexts = []
    total_retrieval_latency = 0.0

    retrieval_filter = rag.Filter(
        vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
    )
    
    generation_start_time_global = time.time()

    # Retrieval
    retrieval_start = time.time()
    retrieval_response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_arn)],
        rag_retrieval_config=rag.RagRetrievalConfig(
            top_k=K_VALUE,
            filter=retrieval_filter
        ),
        text=question,
    )
    total_retrieval_latency += (time.time() - retrieval_start)
    all_contexts.extend(retrieval_response.contexts.contexts)

    # Top-K 정렬
    all_contexts.sort(key=lambda x: x.score, reverse=True)
    final_contexts = all_contexts[:K_VALUE]

    avg_score, hit_at_k, mrr, noise_ratio = calculate_metrics(
        final_contexts, ground_truth_sources, K_VALUE
    )

    log("=====================")
    log(f"{label} 테스트 시작")
    log(f"질문: {question}\n")

    log("--- [검색 지표] ---")
    log(f"P3. 검색 지연 시간: {total_retrieval_latency * 1000:.2f} ms")
    log(f"P5. 평균 검색 점수: {avg_score:.4f}")
    log(f"Hit@{K_VALUE}: {hit_at_k}")
    log(f"MRR: {mrr:.4f}")
    log(f"noise ratio: {noise_ratio:.4f}")
    log("\n")
    log("P2. Top-K 검색 결과:")
    for i, ctx in enumerate(final_contexts, start=1):
        is_gt = "✅ GT" if ctx.source_display_name in ground_truth_sources else ""
        log(f" #{i:02d} Score={ctx.score:.4f} | Source={ctx.source_display_name} {is_gt}")


    rag_tool = Tool(
        retrieval=Retrieval(
            vertex_rag_store=VertexRagStore(
                rag_corpora=[corpus_arn],
                similarity_top_k=K_VALUE,
                vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
            )
        )
    )

    # Generation
    gen_response = client.models.generate_content(
        model=MODEL_ID,
        contents=question,
        config=GenerateContentConfig(tools=[rag_tool]),
    )

    total_latency = (time.time() - generation_start_time_global) * 1000
    log("\n")
    log("P1. 최종 답변 : ")
    log(gen_response.text)
    log("\n")
    log(f"전체 지연 시간 : {total_latency:.2f} ms")
    log("=====================")

# -----------------------------
# 실행
# -----------------------------
for i, (question, gt_sources) in enumerate(GROUND_TRUTH_DATA.items()):
    log(f"\n### 테스트 질문 {i+1} / {len(GROUND_TRUTH_DATA)} ###\n")

    # 전체 corpus
    measure_retrieval_performance(
        RAG_ALL_CORPUS,
        "ALL corpus",
        question,
        gt_sources
    )

    # ISA 전용 corpus
    measure_retrieval_performance(
        ISA_CORPUS,
        "ISA corpus",
        question,
        gt_sources
    )
