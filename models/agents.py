from retriever.retriever_handler import get_retriever
from utils.model_handler import get_llm
from utils.utils import format_docs
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough

def sample_name_searcher(file_folder, file_number, chunk_size, chunk_overlap, search_k, model_name):
    retriever = get_retriever(file_folder, file_number, chunk_size, chunk_overlap, search_k)
    model = get_llm(model_name)
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