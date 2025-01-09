from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents.base import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_teddynote.messages import messages_to_history
from langchain.prompts import PromptTemplate
from langchain_teddynote.evaluator import GroundednessChecker
from tools import embedding_file


# GraphState 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    answer: Annotated[str, "Answer"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)
    

# Graph 구축
class DataExtractor:
    def __init__(
        self, 
        folder_path:str="./data/input_data/", 
        file_number:int=1
    ):
        self.file_name = folder_path + f"paper_{file_number}.pdf"
        self.retriever = embedding_file(file=self.file_name)
        
        self.model = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        self.relevance_checker = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.llm_answer_prompt = """
        Based on the following document, please provide an answer to the given question.
        Document:
        {context}

        Question:
        {question}

        Answer:
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
        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    
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

        # prompt 설정
        prompt = PromptTemplate(
            template=self.llm_answer_prompt,
            input_variables=["context", "question"],
            )

        # 체인 호출
        chain = prompt | self.model | JsonOutputParser()

        response = chain.invoke(
            {
                "question": latest_question,
                "context": context,
                "chat_history": messages_to_history(state["messages"]),
            }
        )

        # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.
        return GraphState(
            answer=response,
            messages=[("user", latest_question), ("assistant", response)]
        )


    def relevance_check(self, state: GraphState) -> GraphState:
        """답변과 검색 문서 간의 관련성을 평가합니다. 

        Args:
            state (GraphState): 검색된 문서와 답변을 가져옵니다. 

        Returns:
            GraphState: 관련성 점수를 저장한 상태 변수
        """    
        # 관련성 평가기를 생성합니다.
        retrieval_answer_relevant = GroundednessChecker(
            llm=self.relevance_checker, target="retrieval-answer"
        ).create()

        # 관련성 체크를 실행("yes" or "no")
        response = retrieval_answer_relevant.invoke(
            {"context": state["context"], "answer": state["answer"]}
        )

        print("==== [RELEVANCE CHECK] ====")
        print(response.score)

        # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
        return GraphState(relevance=response.score)


    def is_relevant(self, state: GraphState) -> GraphState:
        """관련성을 체크하는 함수

        Args:
            state (GraphState):

        Returns:
            GraphState: 관련성을 저장한 상태 변수
        """        
        return state["relevance"]