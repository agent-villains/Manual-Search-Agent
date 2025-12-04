import os
from dotenv import load_dotenv

from . import agent
from . import agent2

load_dotenv()

# 환경 변수로 RAG 버전 선택
# RAG_MODE="routing" 또는 "no_routing" (기본값: "routing")
RAG_MODE = os.getenv("RAG_MODE", "routing").lower()

if RAG_MODE == "no_routing":
    # 라우팅을 사용하지 않는 버전 (ALL 코퍼스만 사용)
    rag_agent = agent2.rag_agent
else:
    # 라우팅을 사용하는 버전 (카테고리별 RAG) - 기본값
    rag_agent = agent.rag_agent

# 참조용: 두 버전 모두 접근 가능
rag_agent_routing = agent.rag_agent
rag_agent_no_routing = agent2.rag_agent

__all__ = ["rag_agent", "rag_agent_routing", "rag_agent_no_routing"]