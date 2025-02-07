import json
import warnings
warnings.filterwarnings("ignore")

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.output_parsers import JsonOutputParser

from langchain.tools import Tool

import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

import functools

from tools.retrieval_tool import embedding_file


# 각 에이전트와 도구에 대한 다른 노드를 생성할 것입니다. 이 클래스는 그래프의 각 노드 사이에서 전달되는 객체를 정의합니다.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class MultiAgentRAG:
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
        file_name = f"00{file_number}"[-3:] 

        ## retriever 설정
        self.retriever = embedding_file(
            file_folder=file_folder, 
            file_name=file_name, 
            # rag_method="relevance-rag",  ## "multiagent-rag", 
            # db_folder=db_folder
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            search_k=search_k
        )
        self.retriever_tool = Tool(
            name="retriever",
            func=self.retriever.get_relevant_documents,
            description="Retrieve relevant documents based on a query."
        )

        ## researcher 시스템 프롬프트
        self.system_prompt = system_prompt
        
        ## agent 및 node 생성
        self.model_name = model_name
        llm = ChatOpenAI(model=self.model_name, temperature=0.1)

        # Research agent and node
        self.research_agent = self.create_agent(
            llm,
            [self.retriever_tool],
            system_message=self.system_prompt["researcher"],
        )
        self.research_node = functools.partial(self.agent_node, agent=self.research_agent, name="Researcher")

        # Data_Verifier
        self.verifier_agent = self.create_agent(
            llm,
            [self.retriever_tool],
            system_message=self.system_prompt["data_verifier"],
        )
        self.verifier_node = functools.partial(self.agent_node, agent=self.verifier_agent, name="Data_Verifier")

        # Json_Processor
        self.json_processor_system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", self.system_prompt["json_processor"]
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.json_processor_agent = self.json_processor_system_prompt | ChatOpenAI(model=self.model_name, temperature=0.1) | JsonOutputParser()
        self.json_processor_node = functools.partial(self.json_processor_agent_node, agent=self.json_processor_agent, name="Json_Processor")

        self.tools = [self.retriever_tool]
        self.tool_executor = ToolExecutor(self.tools)
        

        ## graph 구축
        workflow = StateGraph(AgentState)

        workflow.add_node("Researcher", self.research_node)
        workflow.add_node("Data_Verifier", self.verifier_node)
        workflow.add_node("call_tool", self.tool_node)
        workflow.add_node("Json_Processor", self.json_processor_node)

        workflow.add_edge("Json_Processor", END)
        workflow.add_conditional_edges(
            "Researcher",
            self.router,
            {"continue": "Data_Verifier", "call_tool": "call_tool"},
        )
        workflow.add_conditional_edges(
            "Data_Verifier",
            self.router,
            {"continue": "Researcher", "call_tool": "call_tool", "process_output": "Json_Processor"},
        )
        workflow.add_conditional_edges(
            "call_tool",
            lambda x: x["sender"],
            {
                "Researcher": "Researcher",
                "Data_Verifier": "Data_Verifier",
            },
        )

        workflow.set_entry_point("Researcher")
        self.graph = workflow.compile()   
        
        if save_graph_png:        
            self.graph.get_graph().draw_mermaid_png(output_file_path="./graph_img/multiagentrag_graph.png")


    def create_agent(self, llm, tools, system_message: str):
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
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        
        return prompt | llm.bind_functions(functions)
    
    
    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        if isinstance(result, FunctionMessage):
            pass
        else:
            result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            "sender": name,
        }


    def json_processor_agent_node(self, state, agent, name):
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=f"""Convert Final Output in the given response into a JSON format.: {state["messages"][-1].content}""")
                ]
            }
        )
        return {"messages": result, "name": name}


    def tool_node(self, state):
        # 그래프에서 도구를 실행하는 함수입니다.
        # 에이전트 액션을 입력받아 해당 도구를 호출하고 결과를 반환합니다.
        messages = state["messages"]
        
        # 계속 조건에 따라 마지막 메시지가 함수 호출을 포함하고 있음을 알 수 있습니다.
        first_message = messages[0]
        last_message = messages[-1]
        
        # ToolInvocation을 함수 호출로부터 구성합니다.
        tool_input = json.loads(last_message.additional_kwargs["function_call"]["arguments"])
        tool_name = last_message.additional_kwargs["function_call"]["name"]
        
        if tool_name == "retriever":
            base_query = tool_input.get("__arg1", "")  # 기존 query 가져오기
            refined_query = f"Context: {first_message.content} | Query: {base_query}"
            tool_input["__arg1"] = refined_query
        
        # 단일 인자 입력은 값으로 직접 전달할 수 있습니다.
        if len(tool_input) == 1 and "__arg1" in tool_input:
            tool_input = next(iter(tool_input.values()))
        
        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_input,
        )
        
        # 도구 실행자를 호출하고 응답을 받습니다.
        response = self.tool_executor.invoke(action)
        
        # 응답을 사용하여 FunctionMessage를 생성합니다.
        function_message = FunctionMessage(
            content=f"{tool_name} response: {str(response)}", name=action.tool
        )
        
        # 기존 리스트에 추가될 리스트를 반환합니다.
        return {"messages": [function_message]}
    
    
    def router(self, state):
        # 상태 정보를 기반으로 다음 단계를 결정하는 라우터 함수
        messages = state["messages"]
        last_message = messages[-1]
        if "function_call" in last_message.additional_kwargs:
            # 이전 에이전트가 도구를 호출함
            return "call_tool"
        if "Final Output" in last_message.content:
            # 어느 에이전트든 작업이 끝났다고 결정함
            return "process_output"
        return "continue"