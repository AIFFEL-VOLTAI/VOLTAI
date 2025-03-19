from retriever.retriever_handler import get_retriever
from utils.model_handler import get_llm
from utils.utils import format_docs
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from pydantic import BaseModel
from typing import Literal

class Agent:
    def __init__(
        self,        
        file_folder:str="./data/raw", 
        file_number:int=1, 
        chunk_size: int=500, 
        chunk_overlap: int=100, 
        search_k: int=10,   
        
        sample_name_searcher_model_name:str="gpt-4o",
        supervisor_model_name:str="gpt-4o",
        researcher_model_name:str="gpt-4o",
        verifier_model_name:str="gpt-4o"        
    ):

        ## retriever 설정
        self.retriever = get_retriever(
            file_folder=file_folder, 
            file_number=file_number,
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            search_k=search_k
        )
        self.retriever_tool = Tool(
            name="retriever",
            func=self.retriever.get_relevant_documents,
            description="Retrieve relevant documents based on a query."
        )     
        
        ## Supervisor 시스템 프롬프트
        supervisor_system_prompt = """
        - 당신은 Researcher와 Verifier를 관리하는 지시자입니다. 
        - 아래의 작업자들 간의 대화를 조정하여 최종적으로 정확한 답변을 도출해야 합니다.

        ## 역할 및 프로세스:
        1. **Sample Name Searcher**로부터 sample name들을 받습니다.
        2. 우리는 질문에 담긴 모든 변수에 대한 정보를 각 sample name에 대해 추출해야 합니다.  
        3. 모든 정보를 한 번에 추출하면 정확도가 떨어질 수 있으므로, 추출할 변수들을 4개의 category로 분할합니다.
        4. **4명의 Researcher 에이전트**에게 각 변수 category에 대한 질문을 생성하고, 그 질문 리스트를 제공합니다.
        5. Researcher들이 정보를 추출한 후, **4명의 Verifier 에이전트**에게 전달하여 검증을 요청합니다.
        6. Verifier의 검증 결과를 종합하여 최종 답변을 추론합니다.

        ## 작업자 관리:
        - 당신은 {members} 간의 대화를 조율합니다.
        - 아래의 사용자 요청에 따라, 다음 작업을 수행할 작업자를 결정하고 지시해야 합니다.
        - 각 작업자는 자신의 작업을 완료하면 결과와 상태를 반환합니다.
        - 모든 과정이 완료되면 `FINISH`로 응답해야 합니다.

        ## 질문 목록:
        List()

        ### Final Answer:
        """
        
        ## Researcher 시스템 프롬프트
        researcher_system_prompt = """
        - Supervisor로부터 받은 질문에 해당되는 변수들들를 논문으로부터 검색해서 찾아야 합니다. 
        - retriever tool을 사용할 경우 받은 질문에 추가적인 설명을 붙여 query를 만들고 검색해야 합니다. 
        - 논문에 나와있지 않는 변수가 있다면 누락하지말고 없다라는 말을 꼭 추가해줘야 합니다.
        """

        ## Verifier 시스템 프롬프트
        verifier_system_prompt = """
        - 당신은 Researcher로부터 받은 답변들에 대해 잘못된 부분이 없는지 확인하는 역할을 하는 에이전트 입니다. 
        - 답변들을 확인할 경우 필수적으로 retriever tool을 사용해서 확인해야 합니다. 
        - 논문에 나와있지 않는 정보가 있다면 누락하지말고 없다라는 말을 꼭 추가해줘야 합니다.
        - 잘못된 부분이 있다면 Researcher에게 피드백해야 합니다. 
        - 만약 잘못된 부분이 없이 모두 잘 추출되었다면 잘 작성된 Researcher의 답변을 Supervisor에게 전달합니다. 
        - Supervisor에게 답변을 전달할 경우 `### Complete Verification` 필수적으로 추가해야 합니다. 
        
        ### Complete Verification:
        """

        ## agent 및 node 생성
        self.sample_name_searcher_chain = self.sample_name_searcher(self.retriever, sample_name_searcher_model_name)
        
        members = [f"Researcher{i}" for i in range(1, 5)] + [f"Verifier{i}" for i in range(1, 5)]
        self.supervisor_agent = self.create_supervisor(supervisor_model_name, members, supervisor_system_prompt)
        self.researcher_agent = self.create_agent(researcher_model_name, [self.retriever_tool], researcher1_system_prompt)
        self.verifier_agent = self.create_agent(verifier_model_name, [self.retriever_tool], verifier_system_prompt)

    
    def sample_name_searcher(self, retriever: object, model_name: str):
        model = get_llm(model_name, temperature=0.1)
        
        sample_name_retriever_prompt = """
        You are an expert assistant specializing in extracting information from research papers related to battery technology. Your role is to carefully analyze the provided document.

        Document:
        {context}

        Question:
        {sample_name_question}
        
        Answer:
        """
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=sample_name_retriever_prompt,
            input_variables=["context", "sample_name_question"],
            partial_variables={"format_instructions": format_instructions},
        )

        sample_name_searcher_chain = (
            {"context": retriever | format_docs, "sample_name_question": RunnablePassthrough()}
            | prompt | model | output_parser
        )
        
        return sample_name_searcher_chain


    def create_agent(self, model_name, tools, system_message: str):
        # 에이전트를 생성합니다.
        functions = [format_tool_to_openai_function(t) for t in tools]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."                                        
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join(
            [tool.name for tool in tools]))
        llm = get_llm(model_name, temperature=0.1)
        
        return prompt | llm.bind_functions(functions)
        

    def create_supervisor(self, model_name, members: list, system_prompt: str=None):            
        options_for_next = ["FINISH"] + members
        
        # 작업자 선택 응답 모델 정의: 다음 작업자를 선택하거나 작업 완료를 나타냄
        class RouteResponse(BaseModel):
            next: Literal[*options_for_next]
    
        # ChatPromptTemplate 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next? "
                    "Or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(options_for_next), members=", ".join(members))

        llm = get_llm(model_name, temperature=0.1)

        return prompt | llm.with_structured_output(RouteResponse)
