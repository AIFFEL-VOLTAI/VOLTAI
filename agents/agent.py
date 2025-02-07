from langchain_core.documents.base import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

class Agent:
    def __init__(self, config_path: str = "./config"):    
        self.prompt_path = f"{config_path}/prompts"
        self.example_path = f"{config_path}/examples"
        
        
    def format_docs(self, docs: list[Document]) -> str:
        """문시 리스트에서 텍스트를 추출하여 하나의 문자로 합치는 기능을 합니다.

        Args:
            docs (list[Document]): 여러 개의 Documnet 객체로 이루어진 리스트

        Returns:
            str: 모든 문서의 텍스트가 하나로 합쳐진 문자열을 반환
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
                     
    def sample_name_searcher(self, retriever, llm):
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()

        ## prompt 설정
        prompt = PromptTemplate(
            template="""
        You are an expert assistant specializing in extracting information from research papers related to battery technology. Your role is to carefully analyze the provided document.

        Document:
        {context}

        Question:
        {question}

        Answer:""",
            input_variables=["context", "question"],
            partial_variables={"format_instructions": format_instructions},
        )

        ## 체인 호출
        sample_name_searcher_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt | llm | output_parser
        )
        
        return sample_name_searcher_chain
    
    
    def llm_answer(self, system_prompt, llm):
        ## prompt 설정
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["example", "context", "question"],
        )

        ## 체인 호출
        llm_answer_chain = prompt | llm | JsonOutputParser()

        return llm_answer_chain
    
    
    def relevance_checker(self, system_prompt, llm):
        class GradeAnswer(BaseModel):
            """Binary scoring to evaluate the appropriateness of answers to retrieval"""

            binary_score: str = Field(
                description="Indicate 'yes' or 'no' whether the answer solves the question"
            )
            
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "answer"],
        )

        ## 체인
        structured_relevance_checker = llm.with_structured_output(GradeAnswer)
        relevance_checker_chain = prompt | structured_relevance_checker
        
        return relevance_checker_chain


    def discussion_model(self, system_prompt, input_variables, llm):
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=input_variables,
        )
        discussion_chain = prompt | llm | JsonOutputParser()
        
        return discussion_chain
        
    
    def researcher(self, system_prompt, retriever, llm):
        
        