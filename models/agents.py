from retriever.retriever_handler import get_retriever
from utils.model_handler import get_llm
from utils.utils import format_docs
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import Tool

from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

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
            1. Sample Name Searcher로부터 sample name들을 받습니다. 
            2. 우리는 질문에 담긴 모든 변수에 대한 정보를 각 sample name에 대해 추출해야 합니다.  
            3. 모든 정보를 한번에 추출하기에는 정확도가 떨어질 수 있으니 추출할 변수들을 4개로 분할합니다.
            4. 4명의 researcher 에이전트에게 찾아야 할 변수들에 대한 질문들을 생성하고 verifier로부터 답변을 받아서 합쳐야 합니다. 
                질문을 생성한 후 모두 List에 담아서 제공해야 합니다. 
            5. 4명의 verifier로부터 답변을 받았다면 최종적으로 종합해서 최종 답변을 추론해야 합니다. 
            6. 최종 답변을 추론할 경우 `### Final Answer`이라는 문구를 필수적으로 추가해야 합니다. 
        
        Question: 
        {question}
        
        Sub-Questions:
        List()
        
        ### Final Answer:
        """
        
        ## Researcher 시스템 프롬프트
        researcher_system_prompt = """
        - Supervisor로부터 받은 질문에 해당되는 변수들들를 논문으로부터 검색해서 찾아야 합니다. 
        - retriever tool을 사용할 경우 받은 질문에 추가적인 설명을 붙여 query를 만들고 검색해야 합니다. 
        - 논문에 나와있지 않는 변수가 있다면 누락하지말고 없다라는 말을 꼭 추가해줘야 합니다.
        
        Sub-Question: {sub_question}
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
        self.supervisor_agent = self.create_agent(supervisor_model_name, None, supervisor_system_prompt)
        self.researcher_agent = self.create_agent(researcher_model_name, [self.retriever_tool], researcher_system_prompt)
        self.verifier_agent = self.create_agent(verifier_model_name, [self.retriever_tool], verifier_system_prompt)

    
    def sample_name_searcher(self, retriever: object, model_name: str):
        model = get_llm(model_name, temperature=0.1)
        sample_name_retriever_prompt = """
        You are an expert assistant specializing in extracting information from research papers related to battery technology. Your role is to carefully analyze the provided document.

        Document:
        {context}

        Question:
        {question}

        Answer:
        """
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=sample_name_retriever_prompt,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": format_instructions},
        )

        sample_name_searcher_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | model | output_parser
        )
        
        return sample_name_searcher_chain


    def create_agent(self, model_name, tools, system_message: str):
        # 에이전트를 생성합니다.
        if tools:
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
        
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_message,
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            llm = get_llm(model_name, temperature=0.1)

            return prompt | llm
