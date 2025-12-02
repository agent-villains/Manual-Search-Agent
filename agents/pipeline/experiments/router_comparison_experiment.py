"""
라우터 사용 여부에 따른 RAG 성능 비교 실험 스크립트

사용법:
    python router_comparison_experiment.py <로그 파일 이름> <GROUND_TRUTH_DATA 파일 이름>
    
예시:
    python router_comparison_experiment.py router_comparison.log test1_GT
"""
import os
import sys
import time
import importlib
import json
import vertexai
from google import genai
from vertexai import rag
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from dotenv import load_dotenv

# 현재 디렉토리에서 상위 디렉토리로 이동하여 모듈 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from agents.pipeline.sub_agents.summary_router.agent import summary_router
from google.adk.agents import SequentialAgent

if len(sys.argv) < 3:
    print("사용법: python router_comparison_experiment.py <로그 파일 이름> <GROUND_TRUTH_DATA 파일 이름>")
    sys.exit(1)

LOG_FILE = sys.argv[1]
GT_MODULE_NAME = sys.argv[2]

if GT_MODULE_NAME.endswith(".py"):
    GT_MODULE_NAME = GT_MODULE_NAME[:-3]

# import 및 데이터 로드
try:
    gt_module = importlib.import_module(GT_MODULE_NAME)
except ModuleNotFoundError as e:
    print(f"모듈을 찾을 수 없습니다: {GT_MODULE_NAME}")
    raise e

if not hasattr(gt_module, "GROUND_TRUTH_DATA"):
    raise AttributeError(f"GROUND_TRUTH_DATA 변수가 없습니다")

GROUND_TRUTH_DATA = gt_module.GROUND_TRUTH_DATA

# 환경변수 로드
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL_ID = os.getenv("MODEL_ID")
RAG_ALL_CORPUS = os.getenv("RAG_ALL_CORPUS")

# 카테고리별 코퍼스 매핑
CATEGORY_CORPUS_MAP = {
    "ISA": os.getenv("RAG_ISA_CORPUS"),
    "비대면제휴업무": os.getenv("RAG_비대면제휴업무_CORPUS"),
    "기타매매": os.getenv("RAG_기타매매_CORPUS"),
    "입출금관리": os.getenv("RAG_입출금관리_CORPUS"),
    "창구관리및책임자거래업무": os.getenv("RAG_창구관리및책임자거래업무_CORPUS"),
}

K_VALUE = 10
VECTOR_DISTANCE_THRESHOLD = 0.65

# Vertex 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

# 로그 파일 초기화
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"라우터 비교 실험 로그 — log_file={LOG_FILE}, GT={GT_MODULE_NAME}.GROUND_TRUTH_DATA\n\n")

def log(text: str):
    """로그 파일에 텍스트 기록"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

def calculate_metrics(contexts, ground_truth_sources, top_k):
    """
    검색 청크 목록을 분석하여 지표 계산
    - P5: 평균 점수
    - Hit@k
    - MRR
    - Noise Ratio
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

def get_router_category(question: str):
    """summary_router를 사용하여 질문의 카테고리 선택"""
    try:
        # summary_router 실행
        router_response = summary_router.run(question)
        
        # JSON 파싱
        if hasattr(router_response, 'category_json'):
            category_json_str = router_response.category_json
        else:
            category_json_str = str(router_response)
        
        # JSON 문자열에서 카테고리 추출
        if isinstance(category_json_str, str):
            # JSON 파싱 시도
            try:
                category_data = json.loads(category_json_str)
            except json.JSONDecodeError:
                # JSON이 아닌 경우 직접 추출 시도
                if '"category"' in category_json_str:
                    import re
                    match = re.search(r'"category"\s*:\s*"([^"]+)"', category_json_str)
                    if match:
                        category_data = {"category": match.group(1)}
                    else:
                        category_data = {"category": category_json_str}
                else:
                    category_data = {"category": category_json_str}
        else:
            category_data = category_json_str
        
        category = category_data.get("category", category_data.get("category_json", "Unknown"))
        return category
    except Exception as e:
        log(f"라우터 실행 오류: {e}")
        return "Unknown"

def measure_retrieval_performance(corpus_arn: str, label: str, question: str, ground_truth_sources: list):
    """단일 코퍼스에 대한 검색 성능 측정"""
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
    
    return {
        "avg_score": avg_score,
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "noise_ratio": noise_ratio,
        "retrieval_latency": total_retrieval_latency * 1000,
        "total_latency": total_latency,
        "answer": gen_response.text,
        "contexts": final_contexts
    }

# -----------------------------
# 실험 실행
# -----------------------------
log("=" * 80)
log("라우터 사용 여부에 따른 RAG 성능 비교 실험 시작")
log("=" * 80)
log(f"테스트 질문 수: {len(GROUND_TRUTH_DATA)}\n")

# 전체 통계
all_corpus_stats = []
router_based_stats = []
router_accuracy = []

for i, (question, gt_sources) in enumerate(GROUND_TRUTH_DATA.items()):
    log(f"\n{'='*80}")
    log(f"### 테스트 질문 {i+1} / {len(GROUND_TRUTH_DATA)} ###")
    log(f"{'='*80}\n")

    # 1. 라우터 기반 버전 테스트
    log("\n[1단계] 라우터 기반 RAG 테스트")
    log("-" * 80)
    
    router_category = get_router_category(question)
    log(f"라우터 선택 카테고리: {router_category}")
    
    # Ground Truth 문서가 속한 카테고리 확인 (간단한 휴리스틱)
    # 실제로는 문서 메타데이터에서 확인해야 함
    gt_category = "Unknown"  # TODO: 실제 카테고리 매핑 로직 필요
    
    router_corpus = CATEGORY_CORPUS_MAP.get(router_category)
    if not router_corpus:
        log(f"⚠️ 경고: 카테고리 '{router_category}'에 대한 코퍼스가 설정되지 않았습니다.")
        log("전체 코퍼스를 사용합니다.")
        router_corpus = RAG_ALL_CORPUS
    else:
        log(f"사용 코퍼스: {router_category} 전용 코퍼스")
    
    router_result = measure_retrieval_performance(
        router_corpus,
        f"Router-based ({router_category})",
        question,
        gt_sources
    )
    router_based_stats.append({
        "question": question,
        "category": router_category,
        "gt_category": gt_category,
        "router_correct": router_category == gt_category,  # TODO: 실제 비교 로직 필요
        **router_result
    })
    
    # 2. 전체 코퍼스 버전 테스트
    log("\n[2단계] 전체 코퍼스 RAG 테스트")
    log("-" * 80)
    
    all_corpus_result = measure_retrieval_performance(
        RAG_ALL_CORPUS,
        "All-corpus",
        question,
        gt_sources
    )
    all_corpus_stats.append({
        "question": question,
        **all_corpus_result
    })
    
    # 3. 결과 비교
    log("\n[3단계] 결과 비교")
    log("-" * 80)
    log(f"Hit@{K_VALUE}:")
    log(f"  라우터 기반: {router_result['hit_at_k']}")
    log(f"  전체 코퍼스: {all_corpus_result['hit_at_k']}")
    log(f"MRR:")
    log(f"  라우터 기반: {router_result['mrr']:.4f}")
    log(f"  전체 코퍼스: {all_corpus_result['mrr']:.4f}")
    log(f"Noise Ratio:")
    log(f"  라우터 기반: {router_result['noise_ratio']:.4f}")
    log(f"  전체 코퍼스: {all_corpus_result['noise_ratio']:.4f}")
    log(f"검색 지연 시간:")
    log(f"  라우터 기반: {router_result['retrieval_latency']:.2f} ms")
    log(f"  전체 코퍼스: {all_corpus_result['retrieval_latency']:.2f} ms")
    log(f"전체 지연 시간:")
    log(f"  라우터 기반: {router_result['total_latency']:.2f} ms")
    log(f"  전체 코퍼스: {all_corpus_result['total_latency']:.2f} ms")

# 최종 통계 요약
log("\n\n" + "=" * 80)
log("최종 통계 요약")
log("=" * 80)

def calculate_average_stats(stats_list):
    """통계 리스트의 평균 계산"""
    if not stats_list:
        return {}
    return {
        "avg_hit_at_k": sum(s["hit_at_k"] for s in stats_list) / len(stats_list),
        "avg_mrr": sum(s["mrr"] for s in stats_list) / len(stats_list),
        "avg_noise_ratio": sum(s["noise_ratio"] for s in stats_list) / len(stats_list),
        "avg_retrieval_latency": sum(s["retrieval_latency"] for s in stats_list) / len(stats_list),
        "avg_total_latency": sum(s["total_latency"] for s in stats_list) / len(stats_list),
        "avg_score": sum(s["avg_score"] for s in stats_list) / len(stats_list),
    }

router_avg = calculate_average_stats(router_based_stats)
all_corpus_avg = calculate_average_stats(all_corpus_stats)

log("\n[라우터 기반 RAG 평균 지표]")
log(f"  Hit@{K_VALUE}: {router_avg['avg_hit_at_k']:.4f}")
log(f"  MRR: {router_avg['avg_mrr']:.4f}")
log(f"  Noise Ratio: {router_avg['avg_noise_ratio']:.4f}")
log(f"  평균 검색 점수: {router_avg['avg_score']:.4f}")
log(f"  평균 검색 지연 시간: {router_avg['avg_retrieval_latency']:.2f} ms")
log(f"  평균 전체 지연 시간: {router_avg['avg_total_latency']:.2f} ms")

log("\n[전체 코퍼스 RAG 평균 지표]")
log(f"  Hit@{K_VALUE}: {all_corpus_avg['avg_hit_at_k']:.4f}")
log(f"  MRR: {all_corpus_avg['avg_mrr']:.4f}")
log(f"  Noise Ratio: {all_corpus_avg['avg_noise_ratio']:.4f}")
log(f"  평균 검색 점수: {all_corpus_avg['avg_score']:.4f}")
log(f"  평균 검색 지연 시간: {all_corpus_avg['avg_retrieval_latency']:.2f} ms")
log(f"  평균 전체 지연 시간: {all_corpus_avg['avg_total_latency']:.2f} ms")

log("\n[성능 개선율]")
log(f"  Hit@{K_VALUE} 개선: {((router_avg['avg_hit_at_k'] - all_corpus_avg['avg_hit_at_k']) / max(all_corpus_avg['avg_hit_at_k'], 0.001) * 100):.2f}%")
log(f"  MRR 개선: {((router_avg['avg_mrr'] - all_corpus_avg['avg_mrr']) / max(all_corpus_avg['avg_mrr'], 0.001) * 100):.2f}%")
log(f"  Noise Ratio 개선: {((all_corpus_avg['avg_noise_ratio'] - router_avg['avg_noise_ratio']) / max(all_corpus_avg['avg_noise_ratio'], 0.001) * 100):.2f}%")
log(f"  검색 지연 시간 개선: {((all_corpus_avg['avg_retrieval_latency'] - router_avg['avg_retrieval_latency']) / max(all_corpus_avg['avg_retrieval_latency'], 0.001) * 100):.2f}%")
log(f"  전체 지연 시간 개선: {((all_corpus_avg['avg_total_latency'] - router_avg['avg_total_latency']) / max(all_corpus_avg['avg_total_latency'], 0.001) * 100):.2f}%")

# 라우터 정확도
if router_based_stats:
    router_correct_count = sum(1 for s in router_based_stats if s.get("router_correct", False))
    router_accuracy_rate = router_correct_count / len(router_based_stats)
    log(f"\n[라우터 정확도]")
    log(f"  정확도: {router_accuracy_rate * 100:.2f}% ({router_correct_count}/{len(router_based_stats)})")

log("\n" + "=" * 80)
log("실험 완료")
log("=" * 80)

