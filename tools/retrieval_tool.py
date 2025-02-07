import re
from pdfminer.high_level import extract_text
import unicodedata

from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    ## pdfminer를 활용해서 텍스트 추출하기
    full_text = extract_text(file_path)

    ## Unicode 정규화
    full_text = unicodedata.normalize("NFKD", full_text)

    ## 특정 단어가 있는지 확인
    contains_advancedsciencenews = "www.advancedsciencenews.com" in full_text
    contains_chemelectrochem = "www.chemelectrochem.org" in full_text
    contains_materialsviews = "www.MaterialsViews.com" in full_text

    ## 조건에 따라 키워드 설정
    if contains_materialsviews:
        keyword = "Acknowledgements"
    elif contains_advancedsciencenews or contains_chemelectrochem:
        keyword = "Conflict of Interest"
    else:
        keyword = "References"

    ## 키워드로 시작하는 부분 중 가장 마지막 부분 찾기
    if keyword == "Conflict of Interest":
        keyword_pattern = r"(?i)c[ o]*n[ f]*l[ i]*c[ t]*[\uFB00]*[ o]*f[ i]*n[ t]*e[ r]*e[ s]*t"
    else:
        keyword_pattern = "(?i)" + keyword.replace(" ", r"\s*")

    matches = list(re.finditer(keyword_pattern, full_text))

    if matches:
        ## 마지막 매치의 시작 위치를 기준으로 텍스트를 잘라냄
        last_match = matches[-1]
        full_text = full_text[:last_match.start()]

    return full_text


def embedding_file(
    file_folder: str, 
    file_name: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 100, 
    search_k: int = 10
) -> VectorStoreRetriever:
    """PDF 문서를 청크 단위로 분할하고 OpenAI 임베딩 모델을 활용하여 벡터 저장소(Vector Store)에 저장한 후, 
    해당 벡터 저장소를 기반으로 검색할 수 있는 검색기(VectorStoreRetriever)를 생성합니다.

    Args:
        file_folder (str): PDF 문서가 저장된 폴더 경로
        file_name (str): PDF 문서의 파일명 (확장자 제외)
        chunk_size (int, optional): 문서를 분할할 청크 크기 (기본값: 500)
        chunk_overlap (int, optional): 청크 간 겹치는 텍스트 길이 (기본값: 100)
        search_k (int, optional): 검색 시 반환할 상위 유사 문서 개수 (기본값: 10)

    Returns:
        VectorStoreRetriever: 벡터 저장소를 기반으로 문서를 검색할 수 있는 검색기 객체
    """

    ## RecursiveCharacterTextSplitter를 사용하여 문서를 작은 청크로 분할
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,         ## 최대 청크 길이 정의
        chunk_overlap=chunk_overlap,   ## 청크 간 겹침 길이 정의
        separators=["\n\n"]            ## 문단 단위로 분할
    )

    ## PDF 파일 경로 생성
    paper_file_path = f"{file_folder}/{file_name}.pdf"

    ## 참고 문헌(ref) 섹션 제거 후 텍스트 추출
    docs = remove_last_section_from_pdf(file_path=paper_file_path)

    ## 텍스트를 청크 단위로 분할
    docs = splitter.split_text(docs)

    ## OpenAI 임베딩 모델을 사용하여 문서 임베딩 생성
    embeddings = OpenAIEmbeddings()

    ## FAISS 기반 벡터 저장소 생성 및 텍스트 임베딩 저장
    vector_store = FAISS.from_texts(
        texts=docs,       ## 벡터 저장소에 추가할 문서 리스트
        embedding=embeddings  ## 사용할 임베딩 모델
    )

    ## 벡터 저장소를 검색기로 변환
    retriever = vector_store.as_retriever(
        search_type="similarity",    ## 유사도 기반 검색
        search_kwargs={"k": search_k}  ## 상위 k개의 유사 문서 반환
    )

    ## 생성된 검색기 정보 출력
    print(f"##  {file_name} retriever를 생성했습니다.")
    print(f"    - chunk_size    : {chunk_size}")
    print(f"    - chunk_overlap : {chunk_overlap}")
    print(f"    - retrieve_k    : {search_k}")   

    return retriever
