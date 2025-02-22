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

from tools import embedding_file


# 각 에이전트와 도구에 대한 다른 노드를 생성할 것입니다. 이 클래스는 그래프의 각 노드 사이에서 전달되는 객체를 정의합니다.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class MultiAgentRAG:
    def __init__(
        self, 
        file_folder:str="./data/raw", 
        file_number:int=1, 
        # db_folder:str="./vectordb", 
        chunk_size: int=500, 
        chunk_overlap: int=100, 
        search_k: int=10,       
        system_prompt:str = None, 
        model_name:str="gpt-4o",
        save_graph_png:bool=False,
    ):
        ## 파일 명 설정
        if file_number < 10:
            file_name = f"paper_00{file_number}"
        elif file_number < 100:
            file_name = f"paper_0{file_number}"
        else:
            file_name = f"paper_{file_number}"

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
        self.researcher_system_prompt = system_prompt["researcher_system_prompt"]

        ## verifier 시스템 프롬프트
        self.verifier_system_prompt = """You are a meticulous verifier agent specializing in the domain of battery technology.
Your primary task is to verify the accuracy of the Researcher's answers by using the search tool to cross-check the extracted information from research papers on batteries, formatted into JSON.  

Your responsibilities include validating the following:  

### Accuracy:  
Extracted values through documents retrieved via the search tool must be verified to ensure they match accurately.

### Completeness:  
Confirm that all fields in the JSON structure are either filled with accurate values from the battery-related sections of the PDF or marked as "None" if not mentioned in the document.  

If any field is missing or only partially extracted, explicitly state:  
- **Which fields are incomplete or missing**  
- **Whether the missing information exists in the PDF but was not extracted, or is genuinely absent**  
- **Suggestions for improvement (e.g., re-extraction, manual verification, or alternative sources if applicable)**  

### Consistency:  
Verify that the JSON structure, format, and data types adhere strictly to the required schema for battery-related research data.  

### Corrections:  
Identify and highlight any errors, including:  
- **Inaccurate values** (i.e., extracted values that do not match the PDF)  
- **Missing data** (i.e., fields left empty when information is available)  
- **Formatting inconsistencies** (i.e., data types or schema mismatches)  

For any issues found, provide a **clear and actionable correction**, including:  
- **The specific field in question**  
- **The nature of the issue (incorrect value, missing data, formatting error, etc.)**  
- **Suggestions or corrections to resolve the issue**  

### Handling Missing Data:  
If certain information is genuinely **not found** in the PDF, specify:  
- **Which fields could not be located**  
- **Confirmation that they are absent from the document**  
- **A recommendation to keep the field as `"None"` or any alternative solutions**  

### Final Output:  
If the JSON is entirely correct, confirm its validity and output the JSON structure exactly as provided.  
Include the phrase `### Final Output` before printing the JSON. This ensures the output is clearly marked and easy to locate.  

### Scope:  
Focus **exclusively** on battery-related content extracted from the PDF.  
Ignore any reference content or information outside the provided document.  
"""
        
        ## agent 및 node 생성
        self.model_name = model_name
        llm = ChatOpenAI(model=self.model_name, temperature=0.1)

        # Research agent and node
        self.research_agent = self.create_agent(
            llm,
            [self.retriever_tool],
            system_message=self.researcher_system_prompt,
        )
        self.research_node = functools.partial(self.agent_node, agent=self.research_agent, name="Researcher")

        # Data_Verifier
        self.verifier_agent = self.create_agent(
            llm,
            [self.retriever_tool],
            system_message=self.verifier_system_prompt,
        )
        self.verifier_node = functools.partial(self.agent_node, agent=self.verifier_agent, name="Data_Verifier")

        # Json_Processor
        self.json_processor_system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a JSON Processor Agent. Your sole responsibility is to process the response generated by an LLM and ensure the accurate extraction of the JSON content within the response. Follow these instructions precisely:

### Instructions:
1. **Extract JSON Only**:
- Identify the ```json``` block within the provided response.
- Extract and output the content within the ```json``` block exactly as it appears.

2. **No Modifications**:
- Do not modify, add, or remove any part of the JSON content.
- Preserve the relevancerag structure, field names, and values without alteration.

3. **No Hallucination**:
- Do not interpret, infer, or generate additional content.

4. **Output Format**:
- Respond with the extracted JSON content only.
- Do not include any explanations, comments, or surrounding text.
- The output must be a clean, valid JSON.

### Your Role:
Ensure the integrity and consistency of the JSON data by strictly adhering to these instructions. Your output should always be concise and compliant with the above rules."""
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
        prompt = prompt.partial(tool_names=", ".join(
            [tool.name for tool in tools]))
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