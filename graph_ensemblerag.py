import warnings
warnings.filterwarnings("ignore")

from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser

from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents.base import Document
from langchain_teddynote.messages import messages_to_history
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from tools import embedding_file


# GraphState 상태 정의
class GraphStateEnsembleRAG(TypedDict):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    answer1: Annotated[str, "Answer1"]  # 답변
    answer2: Annotated[str, "Answer2"]  # 답변
    answer3: Annotated[str, "Answer3"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)
    discussion: Annotated[str, "Discussion"]

# Graph 구축
class EnsembleRAG:
    def __init__(
        self, 
        file_folder:str="./data/input_data", 
        file_number:int=1, 
        # db_folder:str="./vectordb", 
        chunk_size: int=500, 
        chunk_overlap: int=100, 
        search_k: int=10,       
        system_prompt:str = None, 
        model_name:str="gpt-4o",
        save_graph_png:bool=False,
    ):
        if file_number < 10:
            file_name = f"paper_00{file_number}"
        elif file_number < 100:
            file_name = f"paper_0{file_number}"
        else:
            file_name = f"paper_{file_number}"

        self.retriever = embedding_file(
            file_folder=file_folder, 
            file_name=file_name, 
            rag_method="ensemble-rag", 
            # db_folder=db_folder
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            search_k=search_k
        )
        
        self.model_name = model_name
        self.model_1 = ChatOpenAI(model_name=self.model_name, temperature=0.5)
        self.model_2 = ChatOpenAI(model_name=self.model_name, temperature=0.7)
        self.model_3 = ChatOpenAI(model_name=self.model_name, temperature=0.9)
        self.llm_answer_prompt = system_prompt["llm_answer_system_prompt"]

        self.relevance_checker = ChatOpenAI(model=self.model_name, temperature=0.5)
        self.relevance_check_template1 = """
You are a grader assessing relevance of a retrieved document to a user question. \n 
Here is the retrieved document: \n\n {context} \n\n
Here is the answer: {answer1} \n
If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant. \n

Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the answer.
If the retrieved document does not contain the values or information being searched for, and 'None' is provided as the answer, check if the response accurately reflects the absence of the requested information. If the absence is accurate and justified, grade the document as relevant even if the values are 'None'.
"""
        self.relevance_check_template2 = """
You are a grader assessing relevance of a retrieved document to a user question. \n 
Here is the retrieved document: \n\n {context} \n\n
Here is the answer: {answer2} \n
If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant. \n

Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the answer.
If the retrieved document does not contain the values or information being searched for, and 'None' is provided as the answer, check if the response accurately reflects the absence of the requested information. If the absence is accurate and justified, grade the document as relevant even if the values are 'None'.
"""
        self.relevance_check_template3 = """
You are a grader assessing relevance of a retrieved document to a user question. \n 
Here is the retrieved document: \n\n {context} \n\n
Here is the answer: {answer3} \n
If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant. \n

Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the answer.
If the retrieved document does not contain the values or information being searched for, and 'None' is provided as the answer, check if the response accurately reflects the absence of the requested information. If the absence is accurate and justified, grade the document as relevant even if the values are 'None'.
"""
        
        self.discussion_model = ChatOpenAI(model=self.model_name, temperature=0.1)
        self.discussion_prompt = """
You are an expert in extracting crucial information from battery-related research papers and generating the most accurate and comprehensive answers. Below are the answers provided by multiple LLM models to the same question, along with the retrieved document (context). Based on this information, generate the most reliable and well-rounded answer. Follow these guidelines when formulating your response:

1. Analyze the answers from each LLM model and extract the key information, prioritizing the overlapping points.
2. Evaluate how the retrieved document (context) relates to the answers from the LLM models, and add important content based on its credibility.
3. In case of ambiguity or conflicts between model answers, draw a clear conclusion based on the retrieved document.
4. The final answer should be accurate and detailed, incorporating all relevant information.
5. The output should be in JSON format, clearly separating and organizing the information.

### Input Data
1. Question: {question}
2. LLM Model 1 Answer: {answer1}
3. LLM Model 2 Answer: {answer2}
4. LLM Model 3 Answer: {answer3}
5. Retrieved Document (context): {context}
"""
        
        # 그래프 생성
        bulider = StateGraph(GraphStateEnsembleRAG)

        # 노드 정의
        bulider.add_node("retrieve", self.retrieve_document)
        bulider.add_node("relevance_check1", self.relevance_check1)
        bulider.add_node("relevance_check2", self.relevance_check2)
        bulider.add_node("relevance_check3", self.relevance_check3)
        bulider.add_node("llm_answer1", self.llm_answer1)
        bulider.add_node("llm_answer2", self.llm_answer2)
        bulider.add_node("llm_answer3", self.llm_answer3)
        bulider.add_node("discussion_node", self.discussion_node)

       # 엣지 정의
        bulider.add_edge("retrieve", "llm_answer1")  # _start_ -> 검색 시작
        bulider.add_edge("retrieve", "llm_answer2")  # _start_ -> 검색 시작
        bulider.add_edge("retrieve", "llm_answer3")  # _start_ -> 검색 시작
        bulider.add_edge("llm_answer1", "relevance_check1")  # 답변 생성 -> 관련성 체크
        bulider.add_edge("llm_answer2", "relevance_check2")  # 답변 생성 -> 관련성 체크
        bulider.add_edge("llm_answer3", "relevance_check3")  # 답변 생성 -> 관련성 체크

        # 조건부 엣지를 추가합니다.
        bulider.add_conditional_edges(
            "relevance_check1",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            self.is_relevant1,
            {
                "yes": "discussion_node",  # 관련성이 있으면 _end_로 이동합니다.
                "no": "retrieve",  # 관련성이 없으면 다시 검색합니다.
            },
        )
        bulider.add_conditional_edges(
            "relevance_check2",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            self.is_relevant2,
            {
                "yes": "discussion_node",  # 관련성이 있으면 _end_로 이동합니다.
                "no": "retrieve",  # 관련성이 없으면 다시 검색합니다.
            },
        )
        bulider.add_conditional_edges(
            "relevance_check3",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            self.is_relevant3,
            {
                "yes": "discussion_node",  # 관련성이 있으면 _end_로 이동합니다.
                "no": "retrieve",  # 관련성이 없으면 다시 검색합니다.
            },
        )

        bulider.add_edge("discussion_node", END)
        
        # 그래프 진입점 설정
        bulider.set_entry_point("retrieve")
        
        # 체크포인터 설정
        memory = MemorySaver()

        # 컴파일
        self.graph = bulider.compile(checkpointer=memory)        
        
        if save_graph_png:
            self.graph.get_graph().draw_mermaid_png(output_file_path="./graph_img/ensemblerag_graph.png")

    
    def format_docs(self, docs: list[Document]) -> str:
        """문시 리스트에서 텍스트를 추출하여 하나의 문자로 합치는 기능을 합니다.

        Args:
            docs (list[Document]): 여러 개의 Documnet 객체로 이루어진 리스트

        Returns:
            str: 모든 문서의 텍스트가 하나로 합쳐진 문자열을 반환
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    def retrieve_document(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
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
        return GraphStateEnsembleRAG(context=retrieved_docs)
    
    
    def llm_answer1(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
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
        chain = prompt | self.model_1 | JsonOutputParser()

        response = chain.invoke(
            {
                "question": latest_question,
                "context": context,
                "chat_history": messages_to_history(state["messages"]),
            }
        )

        # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.
        return GraphStateEnsembleRAG(
            answer1=response,
            messages=[("user", latest_question), ("assistant", response)]
        )


    def llm_answer2(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
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
        chain = prompt | self.model_2 | JsonOutputParser()

        response = chain.invoke(
            {
                "question": latest_question,
                "context": context,
                "chat_history": messages_to_history(state["messages"]),
            }
        )

        # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.
        return GraphStateEnsembleRAG(
            answer2=response,
            messages=[("user", latest_question), ("assistant", response)]
        )


    def llm_answer3(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
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
        chain = prompt | self.model_3 | JsonOutputParser()

        response = chain.invoke(
            {
                "question": latest_question,
                "context": context,
                "chat_history": messages_to_history(state["messages"]),
            }
        )

        # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.
        return GraphStateEnsembleRAG(
            answer3=response,
            messages=[("user", latest_question), ("assistant", response)]
        )


    def relevance_check1(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
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
            template=self.relevance_check_template1,
            input_variables=["context", "answer1"],
        )

        # 체인
        structured_relevance_checker = self.relevance_checker.with_structured_output(GradeAnswer)
        relevance_chain = prompt | structured_relevance_checker
        
        # retrieval_answer_relevant = GroundednessChecker(
        #     llm=self.relevance_checker, target="retrieval-answer"
        # ).create()

        # 관련성 체크를 실행("yes" or "no")
        response = relevance_chain.invoke(
            {"context": state["context"], "answer1": state["answer1"]}
        )

        print(f"        RELEVANCE CHECK for ANSWER 1 : {response.binary_score}")

        # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
        return GraphStateEnsembleRAG(relevance1=response.binary_score)


    def relevance_check2(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
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
            template=self.relevance_check_template2,
            input_variables=["context", "answer2"],
        )

        # 체인
        structured_relevance_checker = self.relevance_checker.with_structured_output(GradeAnswer)
        relevance_chain = prompt | structured_relevance_checker
        
        # retrieval_answer_relevant = GroundednessChecker(
        #     llm=self.relevance_checker, target="retrieval-answer"
        # ).create()

        # 관련성 체크를 실행("yes" or "no")
        response = relevance_chain.invoke(
            {"context": state["context"], "answer2": state["answer2"]}
        )

        print(f"        RELEVANCE CHECK for ANSWER 2 : {response.binary_score}")

        # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
        return GraphStateEnsembleRAG(relevance2=response.binary_score)
    
    
    def relevance_check3(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
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
            template=self.relevance_check_template3,
            input_variables=["context", "answer3"],
        )

        # 체인
        structured_relevance_checker = self.relevance_checker.with_structured_output(GradeAnswer)
        relevance_chain = prompt | structured_relevance_checker
        
        # retrieval_answer_relevant = GroundednessChecker(
        #     llm=self.relevance_checker, target="retrieval-answer"
        # ).create()

        # 관련성 체크를 실행("yes" or "no")
        response = relevance_chain.invoke(
            {"context": state["context"], "answer3": state["answer3"]}
        )

        print(f"        RELEVANCE CHECK for ANSWER 3 : {response.binary_score}")

        # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
        return GraphStateEnsembleRAG(relevance3=response.binary_score)


    def is_relevant1(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
        """관련성을 체크하는 함수

        Args:
            state (GraphState):

        Returns:
            GraphState: 관련성을 저장한 상태 변수
        """        
        return state["relevance1"]


    def is_relevant2(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
        """관련성을 체크하는 함수

        Args:
            state (GraphState):

        Returns:
            GraphState: 관련성을 저장한 상태 변수
        """        
        return state["relevance2"]


    def is_relevant3(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
        """관련성을 체크하는 함수

        Args:
            state (GraphState):

        Returns:
            GraphState: 관련성을 저장한 상태 변수
        """        
        return state["relevance3"]
    
    
    def discussion_node(self, state: GraphStateEnsembleRAG) -> GraphStateEnsembleRAG:
        prompt = PromptTemplate(
            template=self.discussion_prompt,
            input_variables=["question", "answer1", "answer2", "answer3", "context"],
        )
        discussion_chain = prompt | self.discussion_model | JsonOutputParser()
        response = discussion_chain.invoke(
            {
                "question": state["question"],
                "answer1": state["answer1"],
                "answer2": state["answer2"],
                "answer3": state["answer3"],
                "context": state["context"] 
            }
        )
        print(f"        Success Discussion!")

        return GraphStateEnsembleRAG(discussion=response)