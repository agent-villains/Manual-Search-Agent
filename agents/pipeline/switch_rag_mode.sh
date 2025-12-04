#!/bin/bash
# Linux/Mac용 RAG 모드 전환 스크립트

if [ -z "$1" ]; then
    echo "현재 RAG 모드를 확인하려면:"
    echo "  ./switch_rag_mode.sh routing      # 라우팅 버전 사용"
    echo "  ./switch_rag_mode.sh no_routing   # 라우팅 없는 버전 사용"
    python switch_rag_mode.py
else
    python switch_rag_mode.py "$1"
fi

