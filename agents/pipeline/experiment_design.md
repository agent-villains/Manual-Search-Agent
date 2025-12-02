# 라우터 사용 여부에 따른 RAG 성능 비교 실험 설계

## 실험 목적
`summary_router`를 사용하여 카테고리별 코퍼스를 선택하는 방식과 전체 코퍼스를 사용하는 방식의 성능 차이를 비교 분석

## 실험 변인

### 1. 라우터 사용 버전 (Router-based RAG)
- `summary_router`가 질문을 분석하여 카테고리 선택
- 선택된 카테고리만을 코퍼스로 사용
- 카테고리별 코퍼스 매핑:
  - "ISA" → `RAG_ISA_CORPUS`
  - "비대면제휴업무" → `RAG_비대면제휴업무_CORPUS`
  - "기타매매" → `RAG_기타매매_CORPUS`
  - "입출금관리" → `RAG_입출금관리_CORPUS`
  - "창구관리및책임자거래업무" → `RAG_창구관리및책임자거래업무_CORPUS`

### 2. 라우터 미사용 버전 (All-corpus RAG)
- 전체 4분야 문서를 하나의 코퍼스에 담은 버전
- `RAG_ALL_CORPUS` 사용

## 평가 지표

### 검색 성능 지표
1. **Hit@K**: Ground Truth 문서가 Top-K 검색 결과에 포함되는지 여부
2. **MRR (Mean Reciprocal Rank)**: 첫 번째 관련 문서의 역순위 평균
3. **Noise Ratio**: 관련 없는 문서의 비율
4. **평균 검색 점수 (Average Score)**: 검색된 문서들의 평균 유사도 점수

### 답변 품질 지표
1. **정확도**: Ground Truth 문서를 기반으로 한 답변의 정확성
2. **완전성**: 필요한 정보를 모두 포함하는지 여부
3. **관련성**: 질문과 답변의 관련성

### 성능 지표
1. **검색 지연 시간**: RAG 검색에 소요된 시간
2. **전체 지연 시간**: 질문 입력부터 답변 생성까지의 총 시간

### 라우터 정확도
- 라우터가 올바른 카테고리를 선택했는지 여부
- 각 질문의 Ground Truth 문서가 속한 카테고리와 라우터 선택 카테고리 비교

## 실험 설계

### 테스트 데이터
- `data_preprocessing/rag_multicorpus/test1_GT.py`의 `GROUND_TRUTH_DATA` 사용
- 각 질문에 대해 Ground Truth 문서 목록이 정의되어 있음

### 실험 절차
1. 각 테스트 질문에 대해:
   - **라우터 사용 버전**:
     - `summary_router`로 카테고리 선택
     - 선택된 카테고리 코퍼스로 RAG 검색 및 답변 생성
     - 라우터 선택 카테고리 기록
   - **라우터 미사용 버전**:
     - 전체 코퍼스로 RAG 검색 및 답변 생성
   
2. 각 버전에 대해 다음 지표 측정:
   - 검색 성능 지표 (Hit@K, MRR, Noise Ratio, 평균 점수)
   - 검색 지연 시간
   - 전체 지연 시간
   - 최종 답변 텍스트

3. 결과 비교 및 분석:
   - 두 버전 간 지표 비교
   - 라우터 정확도 분석
   - 카테고리별 성능 분석

## 예상 결과 분석

### 라우터 사용의 장점
- 검색 공간 축소로 인한 정확도 향상 가능
- 노이즈 감소 가능
- 검색 속도 향상 가능

### 라우터 사용의 단점
- 라우터가 잘못된 카테고리를 선택하면 관련 문서를 찾지 못함
- 여러 카테고리에 걸친 질문 처리 어려움

### 라우터 미사용의 장점
- 모든 문서를 검색하므로 누락 가능성 낮음
- 여러 카테고리에 걸친 질문 처리 용이

### 라우터 미사용의 단점
- 검색 공간이 넓어 노이즈 증가 가능
- 검색 속도 저하 가능

## 구현 파일 구조

```
agents/pipeline/
├── sub_agents/
│   ├── rag_agent/
│   │   ├── router_based_rag_agent.py  # 라우터 기반 RAG 에이전트
│   │   └── all_corpus_rag_agent.py    # 전체 코퍼스 RAG 에이전트
│   └── summary_router/
│       └── agent.py
└── experiments/
    └── router_comparison_experiment.py  # 실험 실행 스크립트
```

## 실행 방법

```bash
python agents/pipeline/experiments/router_comparison_experiment.py \
    <로그 파일 이름> \
    <GROUND_TRUTH_DATA 파일 이름>
```

예시:
```bash
python agents/pipeline/experiments/router_comparison_experiment.py \
    router_comparison.log \
    data_preprocessing/rag_multicorpus/test1_GT
```

