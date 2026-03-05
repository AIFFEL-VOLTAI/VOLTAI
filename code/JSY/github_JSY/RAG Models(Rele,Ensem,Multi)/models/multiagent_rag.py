import json
import warnings
warnings.filterwarnings("ignore")

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage, # [변경 1] FunctionMessage 대신 ToolMessage 사용
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool

import operator
from typing import Annotated, Sequence, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI # [변경 2] Google GenAI 임포트
from typing_extensions import TypedDict

import functools

from retriever import get_retriever

from langchain_core.runnables import RunnableLambda

# 각 에이전트와 도구에 대한 다른 노드를 생성할 것입니다. 이 클래스는 그래프의 각 노드 사이에서 전달되는 객체를 정의합니다.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class MultiAgentRAG:
    def __init__(
        self, 
        file_folder:str="./data/raw", 
        file_number:int=1, 
        chunk_size: int=500, 
        chunk_overlap: int=100, 
        search_k: int=10,       
        system_prompt:str = None, 
        model_name:str="gemini-3-pro-preview", # [변경 3] 모델 이름 변경
        save_graph_png:bool=False,
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
        
        ## [변경 4] LLM 초기화: ChatGoogleGenerativeAI 사용
        self.model_name = model_name
        # Gemini는 safety settings 등을 설정할 수 있습니다. 필요시 추가하세요.
        llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.0)

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
        # self.verifier_node = functools.partial(self.agent_node, agent=self.verifier_agent, name="Data_Verifier")

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

        self.json_processor_agent = self.json_processor_system_prompt | ChatGoogleGenerativeAI(model=self.model_name, temperature=0.0) | JsonOutputParser()
        self.json_processor_node = functools.partial(self.json_processor_agent_node, agent=self.json_processor_agent, name="Json_Processor")

        self.tools = [self.retriever_tool]
        self.tool_executor = ToolExecutor(self.tools)
        

        ## graph 구축
        workflow = StateGraph(AgentState)

        workflow.add_node("Researcher", self.research_node)
        workflow.add_node("Data_Verifier", self.verifier_node)
        workflow.add_node("call_tool", self.tool_node)
        workflow.add_node("Json_Processor", self.json_processor_node)
        # 🔥 [추가] 회초리 노드 등록
        workflow.add_node("Researcher_Retry", self.researcher_retry_node)

        workflow.add_edge("Json_Processor", END)
        workflow.add_conditional_edges(
            "Researcher",
            self.router,
            {"continue": "Researcher", "call_tool": "call_tool","process_output": "Data_Verifier","retry_researcher": "Researcher_Retry"},
        )
        # 3. [추가] 회초리 맞았으면 -> 다시 Researcher에게 가서 행동하게 함
        workflow.add_edge("Researcher_Retry", "Researcher")
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
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants.\n"
                    "Use the provided tools to progress towards answering the question.\n"
                    "If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off. Execute what you can to make progress.\n"
                    "Avoid repeated tool calls. If uncertain, use null and finalize.\n"
                    "If you or any of the other assistants have the final answer or deliverable, output it.\n"
                    "\n"
                    "*** CRITICAL INSTRUCTION FOR TOOL USAGE ***\n"
                    "Before you call any tool, you MUST write a brief explanation in your response stating EXACTLY what specific information is missing from the current context and WHY you need to search again.\n"
                    "Do NOT call a tool without providing this reasoning first.\n\n"
                    "You have access to the following tools: {tool_names}.\n\n{system_message}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        
        # 🔥 [핵심 추가] LLM에게 전달되기 직전에 과거 메시지들을 "안전한 텍스트"로 세탁합니다.
        def sanitize_messages(input_dict):
            safe_messages = []
            for msg in input_dict["messages"]:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    # 1. 과거의 도구 호출 기록(AIMessage)을 텍스트 기억으로 변환 (API 에러 방지)
                    tool_desc = ", ".join([tc["name"] for tc in msg.tool_calls])
                    content = msg.content if msg.content else ""
                    safe_content = f"{content}\n[System Memory: I successfully called the tool '{tool_desc}']".strip()
                    safe_messages.append(AIMessage(content=safe_content, name=msg.name))
                    
                elif isinstance(msg, ToolMessage):
                    # 2. 도구 결과(ToolMessage)를 일반 HumanMessage로 변환 (API 에러 방지)
                    safe_messages.append(HumanMessage(content=f"[Tool Result from {msg.name}]:\n{msg.content}"))
                    
                else:
                    safe_messages.append(msg)
                    
            # 복사본을 만들어 반환 (원본 LangGraph state는 훼손하지 않음)
            return {"messages": safe_messages, "sender": input_dict.get("sender", "")}

        # 파이프라인의 맨 앞에 세탁기(RunnableLambda)를 끼워 넣습니다.
        return RunnableLambda(sanitize_messages) | prompt | llm.bind_tools(tools)
    
    
    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        
        # [기존 도구 호출 로그]
        if result.tool_calls:
            print(f"\n🚨 [{name}] 도구 호출 감지!")
            if hasattr(result, 'content') and result.content:
                print(f"🔍 [생각의 흐름]: {result.content}")
            for tc in result.tool_calls:
                print(f"🛠️ [호출 도구]: {tc['name']} -> {tc['args']}")
            print("-" * 50)
            
        # 🎙️ [신규 대화 도청 로그] 도구 호출 없이 텍스트만 뱉을 때
        else:
            print(f"\n💬 [{name}] 텍스트 생성 (도구 사용 안 함)")
            # 내용이 너무 길면 보기 힘드니 앞부분 300자만 출력
            print(f"📝 [내용 미리보기]: {result.content[:300]}...")
            print("-" * 50)

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
            
            # [수정 전] 딕셔너리를 그대로 반환해서 에러 발생
            # return {"messages": result, "name": name}
            
            # ✅ [수정 후] 1. JSON을 문자열로 변환 -> 2. AIMessage로 포장 -> 3. 리스트에 담기
            # result는 딕셔너리이므로, 다시 문자열로 바꿔서 메시지에 넣어야 안전합니다.
            import json
            final_content = json.dumps(result, ensure_ascii=False, indent=2)
            
            return {
                "messages": [AIMessage(content=final_content, name=name)], 
                "sender": name
            }


    def tool_node(self, state):
        # [변경 7] Gemini/LangChain 표준에 맞춘 tool_node 로직 수정
        messages = state["messages"]
        last_message = messages[-1]
        
        # tool_calls 속성 확인 (Gemini는 여기에 호출 정보가 담깁니다)
        tool_calls = last_message.tool_calls
        
        results = []
        
        print(f"\n🛠️ [DEBUG] Tool Node 진입! (호출된 도구 개수: {len(tool_calls)})")

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            print(f"➡️ [DEBUG] 호출 도구: {tool_name}")
            print(f"➡️ [DEBUG] 원본 입력(Args): {tool_input}")

            if tool_name == "retriever":
                first_message_content = messages[0].content
                
                # Gemini는 인자를 잘 파싱하지만, 안전장치로 확인
                base_query = tool_input.get("__arg1", "") 
                if not base_query:
                    # 인자가 딕셔너리로 묶여있지 않고 바로 값인 경우 등을 대비
                    if tool_input:
                         base_query = list(tool_input.values())[0]
                
                print(f"🔍 [DEBUG] 추출된 검색어(base_query): {base_query}")

                # [주의] 쿼리가 너무 길어지는지 확인이 필요합니다.
                refined_query = f"Context: {first_message_content[:200]}... | Query: {base_query}"
                print(f"📡 [DEBUG] Retriever로 보내는 최종 쿼리: {refined_query}")
                
                tool_input = refined_query
            
            action = ToolInvocation(
                tool=tool_name,
                tool_input=tool_input,
            )
            
            # 실제 도구 실행 (Retriever 검색 수행)
            response = self.tool_executor.invoke(action)
            
            # --- 결과 확인용 디버그 로그 ---
            response_str = str(response)
            is_empty = False
            
            # 리스트인 경우 개수 확인
            if isinstance(response, list):
                print(f"✅ [DEBUG] 검색 결과 개수: {len(response)}개")
                if len(response) == 0:
                    is_empty = True
            else:
                print(f"✅ [DEBUG] 검색 결과 타입: {type(response)}")

            # 내용 미리보기 (너무 길면 자름)
            print(f"📄 [DEBUG] 검색 결과 내용(앞부분): {response_str[:300]}...")
            
            if is_empty or response_str == "[]":
                print("🚨 [CRITICAL] 검색 결과가 비어있습니다! (Empty Context)")
            # ---------------------------

            # [변경 8] FunctionMessage 대신 ToolMessage 사용 (Gemini 필수)
            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                content=response_str,
                name=tool_name
            )
            results.append(tool_message)
        
        print("🚪 [DEBUG] Tool Node 종료. 결과 반환.\n")
        return {"messages": results}
    
    
    def router(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        sender = state.get("sender", "Unknown") # sender를 미리 가져옵니다.
        
        # 1. 도구 호출 확인 (행동함 -> Tool Node로)
        if getattr(last_message, "tool_calls", None):
            print("🔀 [Router] 도구 호출이 있어서 'call_tool'로 이동합니다.")
            return "call_tool"
            
        # 2. 내용 추출 (List, Dict, String 등 모든 경우의 수 방어)
        content = last_message.content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            content_str = " ".join(text_parts)
        else:
            content_str = str(content)
            
        # 3. JSON 완성 확인 (숙제 끝냄 -> Verifier로)
        # (주의: Verifier가 'Final Output'이라고 말했을 때도 여기서 걸려서 저장 단계로 갑니다)
        if "final output" in content_str.lower() or "```json" in content_str.lower():
            print("🏁 [Router] 'Final Output' 키워드 감지! Json_Processor로 이동하여 종료합니다.")
            return "process_output"
            
        # 🔥 [핵심 추가] 3.5 리서처의 '행동 마비' 감지 및 재시도
        # 리서처가 도구(Step 1)도 안 썼고, JSON(Step 3)도 안 썼는데 말만 한다? -> 다시 해!
        if sender == "Researcher":
            print(f"🔄 [Router] 리서처가 도구/JSON 없이 텍스트만 출력했습니다. 다시 시도하게 합니다. (Retry)")
            return "retry_researcher" 
            
        # 4. 그 외 (Verifier의 피드백 등) -> 계속 핑퐁
        next_agent = "Data_Verifier" if sender == "Researcher" else "Researcher"
        print(f"➡️ [Router] 합의 안 됨. {sender}가 {next_agent}에게 다시 검토를 넘깁니다. (continue)")
        
        return "continue"
    
    # [신규 추가] 리서처가 행동 안 하고 말만 할 때, 정신 차리게 하는 노드
    def researcher_retry_node(self, state):
        messages = state["messages"]
        
        # 지금까지 시스템이 몇 번 혼냈는지 계산합니다.
        enforcer_count = sum(1 for m in messages if getattr(m, "name", "") == "System_Enforcer")
        
        print(f"⚡ [System] 리서처 회초리 방 진입 (누적 경고: {enforcer_count + 1}회)")

        if enforcer_count >= 2:
            # 3번째 경고부터는 극약 처방: "도구 쓰지 마! 억지로라도 JSON 만들어!"
            warning_msg = (
                "[CRITICAL OVERRIDE] YOU ARE STUCK IN AN INFINITE LOOP. "
                "DO NOT USE ANY TOOLS. DO NOT EXPLAIN YOUR THOUGHTS. "
                "YOU MUST IMMEDIATELY OUTPUT THE DATA USING ```json FORMAT. "
                "IF DATA IS MISSING, FILL IT WITH null. "
                "START YOUR RESPONSE WITH ```json."
            )
        else:
            # 처음 1~2번은 원래대로 기회를 줌
            warning_msg = (
                "[SYSTEM ERROR] You outputted raw text without calling a tool or providing JSON. \n"
                "STOP explaining what you will do. \n"
                "IMMEDIATELY call a tool (retriever) to find information OR output the final JSON with ```json."
            )

        from langchain_core.messages import HumanMessage
        return {
            "messages": [HumanMessage(content=warning_msg, name="System_Enforcer")],
            "sender": "System_Enforcer"
        }
    
    # [신규 추가] Data_Verifier 전용 커스텀 노드 (강제 승인 기능 탑재)
    def verifier_node(self, state):
        from langchain_core.messages import AIMessage, HumanMessage # 상단에 없다면 추가
        
        agent = self.verifier_agent
        name = "Data_Verifier"
        messages = state["messages"]
        
        # 🔥 [완벽한 카운팅] 대화 기록에서 Data_Verifier가 말한 횟수를 직접 셉니다.
        # getattr를 써서 name 속성이 없는 메시지에서 에러가 나는 것을 방지합니다.
        verifier_count = sum(1 for m in messages if getattr(m, "name", "") == name)
        
        print(f"📊 [DEBUG] Data_Verifier 이전 개입 횟수: {verifier_count}회")
        
        # 1회 이상 개입했다면 (즉, 리서처가 한 번 고쳐왔는데 또 반려하려는 상황이면) 바로 강제 통과!
        # 횟수를 2로 하고 싶으시면 verifier_count >= 2 로 수정하시면 됩니다.
        if verifier_count >= 2: 
            print(f"🛑 [System] 핑퐁 제한 도달! 강제 승인 명령 주입!")
            force_message = HumanMessage(
                content=(
                    "[SYSTEM INSTRUCTION] STOP! Do NOT ask for any more revisions. "
                    "You MUST APPROVE the data NOW. "
                    "Immediately output the phrase '### Final Output' and print the exact JSON from the Researcher."
                )
            )
            # 기존 메시지 리스트에 강제 명령 추가
            messages = list(messages) + [force_message]

        # 에이전트 실행
        result = agent.invoke({"messages": messages})
        
        # 🔥 [핵심 안전장치] 다음 카운팅을 위해 반환되는 메시지에 명시적으로 이름을 달아줍니다.
        final_result = AIMessage(content=result.content, name=name)
        
        return {
            "messages": [final_result],
            "sender": name,
        }


