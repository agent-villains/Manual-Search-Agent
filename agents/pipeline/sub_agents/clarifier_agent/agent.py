import os
import json

from google.adk.agents.llm_agent import Agent

# summary_router에서 사용하는 카테고리 요약을 그대로 재사용하기 위해 경로 계산
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_ROUTER_DIR = os.path.join(CURRENT_DIR, "..", "summary_router")
SUMMARY_PATH = os.path.join(SUMMARY_ROUTER_DIR, "category_summary.json")

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    CATEGORY_SUMMARIES = json.load(f)


clarifier_agent = Agent(
    model="gemini-2.5-flash",
    name="category_clarifier_agent",
    description="중분류 카테고리가 애매할 때 사용자에게 재질문을 제안하고 최종 카테고리를 정제하는 에이전트",
    instruction=f"""
당신은 Category Clarifier Agent입니다.

이전 단계 에이전트의 결과는 다음과 같습니다.
1) Query Normalizer 결과(JSON): {{normalized_query_json}}
2) Summary Router 결과(JSON): {{category_json}}

또한 각 카테고리의 요약 정보는 다음과 같습니다:
{json.dumps(CATEGORY_SUMMARIES, ensure_ascii=False)}

당신의 임무:
- 사용자의 질문(normalized_query_json 안의 normalized_query)과 summary_router의 category_json을 보고
  해당 카테고리 선택이 애매한지 여부를 판단합니다.
- 애매한 경우, 사용자에게 다시 물어볼 **clarification 질문**을 설계하고,
  어떤 카테고리 후보들 사이에서 애매한지 후보 목록을 함께 제시합니다.
- 애매하지 않은 경우에는 summary_router가 선택한 카테고리를 그대로 최종 카테고리로 확정합니다.

중요:
- 이 에이전트는 실제로 사용자를 인터랙티브하게 다시 질문하지 않습니다.
- 대신, 프론트엔드/상위 애플리케이션이 참고할 수 있도록 아래 JSON 형태로
  "추가 질문이 필요한지 여부"와 "추가 질문 문장", "후보 카테고리"를 제안합니다.

출력 형식 (반드시 이 JSON 형식만 출력할 것):
{{
  "category": "최종으로 선택한 하나의 카테고리 제목",
  "need_clarification": true 또는 false,
  "clarification_question": "need_clarification이 true일 때, 사용자에게 던질 추가 질문. 아니면 빈 문자열",
  "candidate_categories": [
    {{
      "category": "후보 카테고리 제목",
      "reason": "이 후보가 고려 대상이 되는 이유"
    }}
  ]
}}

규칙:
- JSON 바깥에 다른 텍스트를 절대 출력하지 마세요.
- category 값은 항상 CATEGORY_SUMMARIES의 키 중 하나여야 합니다.
- need_clarification이 false인 경우:
  - clarification_question은 ""(빈 문자열)로 둡니다.
  - candidate_categories에는 최종 category 하나만 넣고, reason에 간단히 이유를 적습니다.
- need_clarification이 true인 경우:
  - candidate_categories에 최소 2개 이상의 후보 카테고리를 넣고,
  - clarification_question은 사용자가 둘(or 그 이상)의 후보 중 하나를 고를 수 있도록 자연스럽게 작성합니다.
""",
    # downstream 에이전트(ai_search_agent)가 그대로 {category_json}을 참조하므로
    # output_key는 summary_router와 동일하게 유지한다.
    # 이 경우, 파이프라인 상에서 "마지막에 정의된 category_json"이 사용된다.
    output_key="category_json",
)

__all__ = ["clarifier_agent"]


