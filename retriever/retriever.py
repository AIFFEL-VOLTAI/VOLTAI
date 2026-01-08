from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from utils.remove_reference import remove_last_section_from_pdf


def embedding_file(file_folder: str, file_name: str, chunk_size: int = 500, chunk_overlap: int = 100, search_k: int = 10):
    """문서를 분할하고 벡터 임베딩을 생성하여 검색기를 반환"""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n"])
    docs = remove_last_section_from_pdf(f"{file_folder}/{file_name}.pdf")
    docs = splitter.split_text(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=docs, embedding=embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": search_k})

