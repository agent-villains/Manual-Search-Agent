# agents/doc_router_agent/prompt.py

ROUTER_PROMPT = """
다음 질문이 어떤 문서 도메인에 속하는지 분류해주세요.

도메인 리스트:
- branch_operations (영업점 업무)
- cash_affiliate   (출납/제휴)
- isa
- market_analysis
- general

출력 형식:
{"domain": "...", "confidence": 0~1, "reason": "..."}

질문: "{query}"
"""
