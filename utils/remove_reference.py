import re
import fitz  # PyMuPDF
import unicodedata
import tools
from pdfminer.high_level import extract_text
import re
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
import unicodedata

from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



def remove_last_section_from_pdf(file_path: str) -> str:
    """
    PDF 파일에서 조건에 따라 특정 섹션 이후를 제외하고 본문 텍스트만 반환합니다.

    Args:
        file_path (str): PDF 파일 경로.

    Returns:
        str: 특정 섹션 제외된 본문 텍스트.
    """
    # pdfminer를 활용해서 텍스트 추출하기
    full_text = extract_text(file_path)

    # Unicode 정규화
    full_text = unicodedata.normalize("NFKD", full_text)

    # 특정 단어가 있는지 확인
    contains_advancedsciencenews = "www.advancedsciencenews.com" in full_text
    contains_chemelectrochem = "www.chemelectrochem.org" in full_text
    contains_materialsviews = "www.MaterialsViews.com" in full_text

    # 조건에 따라 키워드 설정
    if contains_materialsviews:
        keyword = "Acknowledgements"
    elif contains_advancedsciencenews or contains_chemelectrochem:
        keyword = "Conflict of Interest"
    else:
        keyword = "References"

    # 키워드로 시작하는 부분 중 가장 마지막 부분 찾기
    if keyword == "Conflict of Interest":
        keyword_pattern = r"(?i)c[ o]*n[ f]*l[ i]*c[ t]*[\uFB00]*[ o]*f[ i]*n[ t]*e[ r]*e[ s]*t"
    else:
        keyword_pattern = "(?i)" + keyword.replace(" ", r"\s*")

    matches = list(re.finditer(keyword_pattern, full_text))

    if matches:
        # 마지막 매치의 시작 위치를 기준으로 텍스트를 잘라냄
        last_match = matches[-1]
        full_text = full_text[:last_match.start()]

    return full_text