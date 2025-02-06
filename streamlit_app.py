import streamlit_authenticator as stauth
import streamlit as st

from crew import Crew
from graph_relevancerag import RelevanceRAG
from graph_ensemblerag import EnsembleRAG
from graph_multiagentrag import MultiAgentRAG


def get_rag_instance(
    rag_method, 
    file_folder, 
    file_number, 
    chunk_size, 
    chunk_overlap,
    search_k,
    system_prompt, 
    model_name, 
    save_graph_png
):
    """
    RAG 클래스를 동적으로 받아서 인스턴스를 생성하는 함수
    
    Params:
        rag_method: RAG 방법 ("relevance-rag", "ensemble-rag", "multiagent-rag")
        file_folder: 논문 파일이 위치한 폴더 경로
        file_number: 처리할 논문 번호
        system_prompt: system prompt
        model_name: LLM 모델 명 ("gpt-4o", "gpt-4o-mini")
        save_graph_png: graph 저장 결정
        
    Return:
        생성된 RAG 모델 인스턴스
    """
    
    # RAG 모델 인스턴스 생성
    if rag_method == "relevance-rag":
        return RelevanceRAG(file_folder, file_number, chunk_size, chunk_overlap, search_k, system_prompt, model_name, save_graph_png)
        
    elif rag_method == "ensemble-rag":
        return EnsembleRAG(file_folder, file_number, chunk_size, chunk_overlap, search_k, system_prompt, model_name, save_graph_png)
        
    elif rag_method == "multiagent-rag":
        return MultiAgentRAG(file_folder, file_number, chunk_size, chunk_overlap, search_k, system_prompt, model_name, save_graph_png)

# def main():
st.set_page_config(
    page_title="안녕하세요!",
)

## 스트림릿 페이지 제목 설정
st.title("Extract Battery Data from Papers🔋")

## 사이드 바: rag_method,  
rag_method = st.sidebar.selectbox("RAG Method", ["Relevance-RAG", "Ensemble-RAG", "Multiagent-RAG"])
open_api_key = st.sidebar.text_input("Input your OpenAI api key🗝️:", type="password")

## pdf 파일들 불러오기
st.subheader('Input Papers📑')
pdf_files = st.file_uploader("Upload your PDFs here:", type="pdf", accept_multiple_files=True)

if pdf_files:
    category_names = ["CAM (Cathode Active Material)", "Electrode (half-cell)", "Morphological Properties", "Cathode Performance"]
    for single_paper in pdf_files:
        st.toggle(f"Extract {single_paper.name}", )
    
    