import warnings
import json
import google.generativeai as genai
warnings.filterwarnings("ignore")

from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser

from typing import Annotated, TypedDict
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents.base import Document
from langchain_teddynote.messages import messages_to_history
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from retriever import get_retriever


# GraphState 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    example: Annotated[dict, "Example"] # 예시
    answer: Annotated[str, "Answer"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)

# Graph 구축
class RelevanceRAG:
    def __init__(
        self, 
        file_folder:str="./data/input_data/", 
        file_number:int=1, 
        # db_folder:str="./vectordb", 
        chunk_size: int=500, 
        chunk_overlap: int=100, 
        search_k: int=10,       
        system_prompt:str = None, 
        model_name:str="gemini-3-pro-preview",
        save_graph_png:bool=False,
    ):
        self.retriever = get_retriever(
            file_folder=file_folder, 
            file_number=file_number, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            search_k=search_k
        )
        self.model_name = model_name

        # =======================================================
        # 🌟 [추가된 코드] generation_config 변수 정의 🌟
        # * NameError 해결 및 RAG 최적화 설정 적용
        # =======================================================
        generation_config = {
            "temperature": 1.0,          
            "max_output_tokens": 2048,     # JSON 출력을 위해 넉넉히 설정
            "thinking_config": {
                "include_thoughts": False, # LangGraph의 JSON 파싱 오류 방지 (필수)
                "thinking_level": "high"    # 속도 개선을 위해 Low Reasoning 레벨로 설정
            }
        }
        # =======================================================

        self.model = ChatGoogleGenerativeAI(model=self.model_name, generation_config=generation_config) # 👈 설정 적용)
        self.relevance_checker = ChatGoogleGenerativeAI(model=self.model_name, temperature=1.0)
        self.llm_answer_prompt = system_prompt["llm_answer_system_prompt"]
        self.relevance_check_template = """
        You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the answer: {answer} \n
        If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant. \n
        
        Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the answer.
        If the retrieved document does not contain the values or information being searched for, and 'None' is provided as the answer, check if the response accurately reflects the absence of the requested information. If the absence is accurate and justified, grade the document as relevant even if the values are 'None'.
        """
        
        # 그래프 생성
        bulider = StateGraph(GraphState)

        # 노드 정의
        bulider.add_node("retrieve", self.retrieve_document)
        bulider.add_node("relevance_check", self.relevance_check)
        bulider.add_node("llm_answer", self.llm_answer)

       # 엣지 정의
        bulider.add_edge("retrieve", "llm_answer")  # _start_ -> 검색 시작
        bulider.add_edge("llm_answer", "relevance_check")  # 답변 생성 -> 관련성 체크

        # 조건부 엣지를 추가합니다.
        bulider.add_conditional_edges(
            "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            self.is_relevant,
            {
                "yes": END,  # 관련성이 있으면 _end_로 이동합니다.
                "no": "retrieve",  # 관련성이 없으면 다시 검색합니다.
            },
        )

        # 그래프 진입점 설정
        bulider.set_entry_point("retrieve")
        
        # 체크포인터 설정
        memory = MemorySaver()

        # 컴파일
        self.graph = bulider.compile(checkpointer=memory) 
        
        if save_graph_png:       
            self.graph.get_graph().draw_mermaid_png(output_file_path="./graph_img/relevancerag_graph.png")

    
    def format_docs(self, docs: list[Document]) -> str:
        """문시 리스트에서 텍스트를 추출하여 하나의 문자로 합치는 기능을 합니다.

        Args:
            docs (list[Document]): 여러 개의 Documnet 객체로 이루어진 리스트

        Returns:
            str: 모든 문서의 텍스트가 하나로 합쳐진 문자열을 반환
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    def retrieve_document(self, state: GraphState) -> GraphState:
        """문서에서 검색하여 질문과 관련성 있는 문서를 찾습니다.

        Args:
            state (GraphState): 질문을 상태에서 가져옵니다.

        Returns:
            GraphState: 검색된 문서를 context 키에 저장한 상태 변수
        """        
        # 질문을 상태에서 가져옵니다.
        latest_question = state["question"]

        # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
        retrieved_docs = self.retriever.invoke(latest_question)

        # 검색된 문서를 형식화합니다.(프롬프트 입력으로 넣어주기 위함)
        retrieved_docs = self.format_docs(retrieved_docs)

        # 검색된 문서를 context 키에 저장합니다.
        return GraphState(context=retrieved_docs)
    
    
    def llm_answer(self, state: GraphState) -> GraphState:
        """프롬프트에 따라 LLM이 질문에 대한 답변을 출력합니다. 

        Args:
            state (GraphState): 질문, 검색된 문서를 상태에서 가져옵니다. 

        Returns:
            GraphState: json 형태로 생성된 답변, (유저의 질문, 답변) 메세지를 저장한 상태 변수
        """        
        # 질문을 상태에서 가져옵니다.
        latest_question = state["question"]

        # 검색된 문서를 상태에서 가져옵니다.
        context = state["context"]
        
        # example 
        example = state["example"]

        # prompt 설정
        prompt = PromptTemplate(
            template=self.llm_answer_prompt,
            input_variables=["example", "context", "question"],
            )

        # 체인 호출
        chain = prompt | self.model | JsonOutputParser()

        response = chain.invoke(
            {
                "question": latest_question,
                "context": context, 
                "example": example,
                "chat_history": messages_to_history(state["messages"]),
            }
        )

        # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.

        if isinstance(response, list):
            final_answer = response
        else:
            final_answer = [response]

        return GraphState(
            answer=final_answer, 
            messages=[
                ("user", latest_question), 
                ("assistant", json.dumps(response, ensure_ascii=False)) 
            ]
        )


    def relevance_check(self, state: GraphState) -> GraphState:
        """답변과 검색 문서 간의 관련성을 평가합니다. 

        Args:
            state (GraphState): 검색된 문서와 답변을 가져옵니다. 

        Returns:
            GraphState: 관련성 점수를 저장한 상태 변수
        """    
        
        class GradeAnswer(BaseModel):
            """Binary scoring to evaluate the appropriateness of answers to retrieval"""

            binary_score: str = Field(
                description="Indicate 'yes' or 'no' whether the answer solves the question"
            )
            
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=self.relevance_check_template,
            input_variables=["context", "answer"],
        )

        # 체인
        structured_relevance_checker = self.relevance_checker.with_structured_output(GradeAnswer)
        relevance_chain = prompt | structured_relevance_checker
        
        # retrieval_answer_relevant = GroundednessChecker(
        #     llm=self.relevance_checker, target="retrieval-answer"
        # ).create()

        # 관련성 체크를 실행("yes" or "no")
        response = relevance_chain.invoke(
            {"context": state["context"], "answer": state["answer"]}
        )

        print(f"        RELEVANCE CHECK : {response.binary_score}")

        # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
        return GraphState(relevance=response.binary_score)


    def is_relevant(self, state: GraphState) -> GraphState:
        """관련성을 체크하는 함수

        Args:
            state (GraphState):

        Returns:
            GraphState: 관련성을 저장한 상태 변수
        """        
        return state["relevance"]