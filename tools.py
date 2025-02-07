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


def embedding_file(
    file_folder: str, 
    file_name: str, 
    # rag_method: str, 
    chunk_size: int=500, 
    chunk_overlap: int=100, 
    search_k: int=10
) -> VectorStoreRetriever:
    """문서를 청크 단위로 분할하고 임베딩 모델(text-embedding-ada-002)을 통해 임베딩하여 vector store에 저장합니다. 이후 vector store를 기반으로 검색하는 객체를 생성합니다.

    Args:
        file (str): pdf 문서 경로

    Returns:
        VectorStoreRetriever: 검색기
    """
    ## 긴 텍스트를 작은 청크로 나누는 데 사용되는 클래스
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,         ## 최대 청크 길이 정의
        chunk_overlap=chunk_overlap,      ## 청크 간 겹침 길이 정의
        separators=["\n\n"]     ## 텍스트를 나눌 때 사용할 구분자를 지정 (문단)
    )
    paper_file_path = f"{file_folder}/{file_name}.pdf"
    
    ## ref 제거 전 코드
    # if rag_method == "multiagent-rag":
    #     loader = PyPDFLoader(paper_file_path)
    #     docs = loader.load_and_split(text_splitter=splitter) 
       
    #     ## Embedding 생성 및 vector store에 저장
    #     embeddings = OpenAIEmbeddings()
    #     vector_store = FAISS.from_documents(
    #         documents=docs,         ## 벡터 저장소에 추가할 문서 리스트
    #         embedding=embeddings    ## 사용할 임베딩 함수
    #     )
    
    ## ref 제거 후 코드
    # elif rag_method == "relevance-rag" or rag_method == "ensemble-rag" or rag_method == "crew-rag":
    docs = remove_last_section_from_pdf(file_path=paper_file_path)
    docs = splitter.split_text(docs)
    
    ## Embedding 생성 및 vector store에 저장        
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        texts=docs,         ## 벡터 저장소에 추가할 문서 리스트
        embedding=embeddings    ## 사용할 임베딩 함수
    )
    
    # ## key error
    # else:
    #     raise KeyError(f"Invalid rag_method: {rag_method}")

    ## 검색기로 변환: 현재 벡터 저장소를 기반으로 VectorStoreRetriever 객체를 생성하는 기능을 제공
    retriever = vector_store.as_retriever(
        search_type="similarity",    ## 어떻게 검색할 것인지? default가 유사도
        search_kwargs={"k": search_k}
    )
    
    print(f"##       {file_name} retriever를 생성했습니다.")
    print(f"##          - chunk_size    :{chunk_size}")
    print(f"##          - chunk_overlap :{chunk_overlap}")
    print(f"##          - retrieve_k    :{search_k}")   

    return retriever


## Save VectorDB Function
# def embedding_file(file_folder: str, file_name: str, db_folder: str) -> VectorStoreRetriever:
#     """문서를 청크 단위로 분할하고 임베딩 모델(text-embedding-ada-002)을 통해 임베딩하여 vector store에 저장합니다. 이후 vector store를 기반으로 검색하는 객체를 생성합니다.

#     Args:
#         file (str): pdf 문서 경로

#     Returns:
#         VectorStoreRetriever: 검색기
#     """

#     ## Vector DB 불러오기
#     if os.path.exists(f"{db_folder}/vectordb-{file_name}.faiss") and os.path.exists(f"{db_folder}/vectordb-{file_name}.pkl"):
#         vector_store = FAISS.load_local(
#             folder_path=db_folder, 
#             index_name=f"vectordb-{file_name}", 
#             embeddings = OpenAIEmbeddings(),
#             allow_dangerous_deserialization=True
#             )
#         print(f"##       vectordb-{file_name}을 불러왔습니다.")

#     else:
#         ## 긴 텍스트를 작은 청크로 나누는 데 사용되는 클래스
#         splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             chunk_size=500,         ## 최대 청크 길이 정의
#             chunk_overlap=100,      ## 청크 간 겹침 길이 정의
#             separators=["\n\n"]     ## 텍스트를 나눌 때 사용할 구분자를 지정 (문단)
#         )

#         ## PDF 파일 불러오기
#         loader = PyPDFLoader(f"{file_folder}/{file_name}.pdf")
#         docs = loader.load_and_split(text_splitter=splitter)

#         ## Embedding 생성 및 vector store에 저장
#         embeddings = OpenAIEmbeddings()
#         vector_store = FAISS.from_documents(
#             documents=docs,         ## 벡터 저장소에 추가할 문서 리스트
#             embedding=embeddings    ## 사용할 임베딩 함수
#         )

#         # output 폴더가 없으면 생성
#         if not os.path.exists(db_folder):
#             os.makedirs(db_folder)
        
#         vector_store.save_local(
#             folder_path=db_folder,
#             index_name=f"vectordb-{file_name}"
#         )
#         print(f"##       vectordb-{file_name}을 {db_folder}/vectordb-{file_name}에 저장했습니다.")
    
#     ## 검색기로 변환: 현재 벡터 저장소를 기반으로 VectorStoreRetriever 객체를 생성하는 기능을 제공
#     retriever = vector_store.as_retriever(
#         search_type="similarity",    ## 어떻게 검색할 것인지? default가 유사도
#         search_kwargs={"k": 10}
#     )

#     return retriever