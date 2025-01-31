import os
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def embedding_file(
    file_folder: str, 
    file_name: str, 
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

    ## PDF 파일 불러오기
    loader = PyPDFLoader(f"{file_folder}/{file_name}.pdf")
    docs = loader.load_and_split(text_splitter=splitter)

    ## Embedding 생성 및 vector store에 저장
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents=docs,         ## 벡터 저장소에 추가할 문서 리스트
        embedding=embeddings    ## 사용할 임베딩 함수
    )

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