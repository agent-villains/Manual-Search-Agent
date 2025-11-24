import os
import vertexai
from vertexai import rag
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
CORPUS_NAME=os.getenv("RAG_CORPUS")
vertexai.init(project=PROJECT_ID, location=LOCATION)

files = list(rag.list_files(corpus_name=CORPUS_NAME))

print("총 RAG 파일 개수:", len(files))
print("===== 파일 목록 =====")
for f in files:
    print("- file:", f.name)
    print("uri:", getattr(f, "gcs_source", None) or getattr(f, "file_source", None))
    print()
