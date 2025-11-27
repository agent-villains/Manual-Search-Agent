
KEYWORDS = {
    "isa": ["isa", "특약", "계좌전환", "해지", "세액", "납입"],
    "cash_affiliate": ["수표", "입금", "출금", "어음", "제휴"],
    "branch_operations": ["여신", "대출", "업무분장", "계약", "펀드"],
    "market_analysis": ["시장", "리포트", "분석", "자동화"],
}


def classify_domain(query: str):
    query_lower = query.lower()
    scores = {}

    for domain, words in KEYWORDS.items():
        score = sum(1 for w in words if w in query_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return {"domain": "general", "confidence": 0.2}

    best = max(scores, key=scores.get)
    confidence = min(scores[best] * 0.3, 0.9)

    return {"domain": best, "confidence": confidence}
