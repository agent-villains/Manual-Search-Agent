from google.adk.agents.llm_agent import Agent
import json

CATEGORY_KEYWORDS = {
    "입출금관리": [
        "자금 이체",
        "입출금 관리",
        "수표 처리",
        "착오송금",
        "오류 정정",
        "이체수수료",
        "금융사고 예방",
        "고객 확인",
        "일일 자금 마감"
    ],
    "서비스": [
        "CMA 서비스",
        "CMA 운용상품",
        "온라인 거래",
        "주문 위임",
        "알리미 서비스",
        "보안 및 인증",
        "사업자전용CMA",
        "거래 위험 관리",
        "수수료 정책",
        "계좌 개설/변경",
        "구비 서류"],
    "자금세탁방지": [
        "자금세탁방지 (AML)",
        "고객확인의무 (KYC)",
        "계좌 관리",
        "온라인/디지털 서비스",
        "금융 상품",
        "알림 서비스",
        "주문 위임",
        "위험 관리",
        "수수료 정책",
        "정보 보안"
    ],
    "비대면 제휴업무": [
        "제휴 앱 플랫폼",
        "비대면 계좌개설",
        "자산관리 서비스",
        "투자 상품",
        "포트폴리오 자문",
        "계좌 운영 및 관리",
        "이체 및 금융사고 대응",
        "수수료",
        "주문 위임"
    ]
}

keyword_router = Agent(
    model="gemini-2.5-flash",
    name="keyword_router",
    description="키워드 기반 중분류 라우팅 에이전트",
    instruction=f"""
당신의 임무는 사용자의 질문을 보고,
중분류별 대표 키워드를 참고하여
가장 관련 있는 중분류 제목 하나만 선택해 JSON으로 출력하는 것입니다.

출력 형식:
{{
  "category": "선택된 중분류 제목"
}}

중분류별 대표 키워드 목록:
{json.dumps(CATEGORY_KEYWORDS, ensure_ascii=False)}
""",
    output_key="category_json",
)

__all__ = ["keyword_router"]
