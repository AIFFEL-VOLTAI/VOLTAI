from tools import embedding_file
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough


class Crew:
    def __init__(
        self, 
        file_folder:str="./data/input_data", 
        file_number:int=22, 
        rag_method:str="crew-rag", 
        chunk_size:int=1000, 
        chunk_overlap:int=100, 
        search_k:int=10, 
        model_name:str="gpt-4o", 
    ):
        file_name = "paper_" + f"00{file_number}"[-3:]
        self.retriever = embedding_file(
            file_folder=file_folder, 
            file_name=file_name, 
            # rag_method=rag_method, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            search_k=search_k            
        )
        
        self.model_name = model_name
        self.model = ChatOpenAI(model_name=self.model_name, temperature=0)
        
        ## Sample Name Retriever LLM Prompt
        self.sample_name_retriever_prompt = """
        You are an expert assistant specializing in extracting information from research papers related to battery technology. Your role is to carefully analyze the provided document.

        Document:
        {context}

        Question:
        {question}

        Answer:
        """
    

    def format_docs(self, docs: list[Document]) -> str:
        """문시 리스트에서 텍스트를 추출하여 하나의 문자로 합치는 기능을 합니다.

        Args:
            docs (list[Document]): 여러 개의 Documnet 객체로 이루어진 리스트

        Returns:
            str: 모든 문서의 텍스트가 하나로 합쳐진 문자열을 반환
        """
        return "\n\n".join(doc.page_content for doc in docs)

        
    def sample_name_searcher(self):
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()

        # prompt 설정
        prompt = PromptTemplate(
            template=self.sample_name_retriever_prompt,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": format_instructions},
            )

        # 체인 호출
        sample_name_searcher_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt | self.model | output_parser)
        
        return sample_name_searcher_chain