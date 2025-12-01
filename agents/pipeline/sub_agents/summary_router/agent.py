import os
from google.adk.agents.llm_agent import Agent
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(CURRENT_DIR, "category_summary.json")

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    CATEGORY_SUMMARIES = json.load(f)

summary_router = Agent(
    model="gemini-2.5-flash",
    name="summary_router",
    description="요약 기반 중분류 라우팅 에이전트",
    instruction=f"""
당신의 임무는 사용자의 질문을 보고,
아래 중분류 요약문 중에서 의미적으로 가장 가까운 것을 하나 선택하여
JSON으로 출력하는 것입니다.

출력 형식:
{{
  "category": "선택된 중분류 제목"
}}

중분류 요약문 목록:
{json.dumps(CATEGORY_SUMMARIES, ensure_ascii=False)}
""",
    output_key="category_json",
)

__all__ = ["summary_router"]
