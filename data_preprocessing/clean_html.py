import sys
import os
from pathlib import Path
from typing import Tuple, Optional
from bs4 import BeautifulSoup

TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>{page_title}</title>
  <style>
    body {{
      font-family: "맑은 고딕", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
      line-height: 1.6;
      margin: 20px;
    }}
    h1 {{
      font-size: 20px;
      margin-bottom: 10px;
    }}
    .breadcrumb {{
      font-size: 11px;
      color: #555;
      margin-bottom: 16px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 16px;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 4px 6px;
      vertical-align: top;
    }}
    th {{
      background: #f5f5f5;
    }}
  </style>
</head>
<body>
  <h1>{h1_title}</h1>
  <div id="content-placeholder"></div>
</body>
</html>
"""

def load_original_html(path: Path) -> str:
  text = path.read_text(encoding="utf-8", errors="ignore")
  soup = BeautifulSoup(text, "html.parser")

  # td.line-content 안에 있는 HTML 반환
  line_tds = soup.select("td.line-content")
  if line_tds:
    inner_html = "\n".join(td.get_text() for td in line_tds)
    return inner_html
  return text

def extract_main_parts(html_text: str) -> Tuple[Optional[BeautifulSoup], Optional[BeautifulSoup], str]:
  """
  titleWrap, manualViewCont, 제목 텍스트 추출
  """
  
  soup = BeautifulSoup(html_text, "html.parser")

  title_wrap = soup.find("div", class_="titleWrap")
  manual_div = soup.find(id="manualViewCont")

  # 제목 추출
  title_text = "ISA 매뉴얼"
  if title_wrap:
    h2 = title_wrap.find("h2")
    if h2 and h2.get_text(strip=True):
      title_text = h2.get_text(strip=True)

  return title_wrap, manual_div, title_text

def build_clean_html(title_wrap, manual_div, title_text: str) -> str:
  """
  정제된 HTML 생성.
  """
  page_title = f"{title_text} - 본문만" # 원본 파일과의 구분을 위해 - 본문만 파일 이름 추가함
  base_html = TEMPLATE_HTML.format(page_title=page_title, h1_title=title_text)

  out_soup = BeautifulSoup(base_html, "html.parser")
  placeholder = out_soup.find(id="content-placeholder")

  if title_wrap:
    placeholder.append(title_wrap)

  if manual_div:
    placeholder.append(manual_div)

  placeholder.attrs.pop("id", None)

  return out_soup.prettify()


def process_file(path: Path, out_dir: Optional[Path] = None):
  original_html = load_original_html(path)
  title_wrap, manual_div, title_text = extract_main_parts(original_html)

  if not manual_div:
    print("오류")
    return

  clean_html = build_clean_html(title_wrap, manual_div, title_text)

  if out_dir is None:
    out_dir = path.parent

  out_dir.mkdir(parents=True, exist_ok=True)
  out_name = path.stem + "_본문만.html"
  out_path = out_dir / out_name

  out_path.write_text(clean_html, encoding="utf-8")
  print("저장완료",out_path)


def main():
  target = Path(sys.argv[1])
  if target.is_file():
    process_file(target)
  elif target.is_dir():
    html_files = list(target.glob("*.html"))
    if not html_files:
      print("html 파일이 없음")
      return
    for f in html_files:
      process_file(f)

if __name__ == "__main__":
  main()
