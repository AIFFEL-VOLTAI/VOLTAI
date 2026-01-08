from .retriever import embedding_file

def get_retriever(file_folder, file_number, chunk_size, chunk_overlap, search_k):
    """Embedding을 기반으로 retriever를 생성"""

    file_name = f"paper_{file_number:03d}"

    return embedding_file(
        file_folder=file_folder,
        file_name=file_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        search_k=search_k
    )
