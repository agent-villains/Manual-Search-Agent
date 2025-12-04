"""
RAG 모드 전환 스크립트

사용법:
    python switch_rag_mode.py routing      # 라우팅 버전 사용
    python switch_rag_mode.py no_routing   # 라우팅 없는 버전 사용
    python switch_rag_mode.py              # 현재 모드 확인
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 경로 찾기
def find_env_file():
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        env_file = parent / ".env"
        if env_file.exists():
            return env_file
    return current / ".env"

def get_current_mode():
    """현재 RAG_MODE 확인"""
    load_dotenv()
    mode = os.getenv("RAG_MODE", "routing")
    return mode

def set_mode(mode):
    """RAG_MODE 설정"""
    if mode not in ["routing", "no_routing"]:
        print(f"❌ 잘못된 모드: {mode}")
        print("사용 가능한 모드: 'routing' 또는 'no_routing'")
        return False
    
    env_file = find_env_file()
    
    # .env 파일 읽기
    env_vars = {}
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    
    # RAG_MODE 업데이트
    env_vars["RAG_MODE"] = mode
    
    # .env 파일 쓰기
    with open(env_file, "w", encoding="utf-8") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ RAG_MODE가 '{mode}'로 설정되었습니다.")
    print(f"   파일 위치: {env_file}")
    return True

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        set_mode(mode)
    else:
        current = get_current_mode()
        print(f"현재 RAG_MODE: {current}")
        print("\n사용법:")
        print("  python switch_rag_mode.py routing      # 라우팅 버전 사용")
        print("  python switch_rag_mode.py no_routing   # 라우팅 없는 버전 사용")

if __name__ == "__main__":
    main()

