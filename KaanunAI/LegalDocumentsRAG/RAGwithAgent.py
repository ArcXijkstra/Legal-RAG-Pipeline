from RAGwithoutAgent import RAGPipeline
from crewai import Agent, Task, Crew
from langchain.tools import tool
from typing import Optional, List
import os
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from crewai import LLM



# llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")
# llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")

# llm = LLM(model = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768"))

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
GEMINI_API_KEY=os.getenv("GOOGLE_API_KEY")



llm = LLM(model= 'groq/llama-3.3-70b-versatile')
# llm = LLM(
#     model="gemini/gemini-1.5-pro-latest",
#     temperature=0.7
# )

# Advanced configuration with detailed parameters


def format_documents(docs: list) -> str:
        """Convert list of Documents to formatted string with metadata"""
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Section: {doc.metadata.get('section_title', 'N/A')}\n"
            f"Content: {doc.page_content}..."
            for doc in docs
        )


# [Keep all your existing imports and environment setup]

class LegalRAGAgents:
    def __init__(self):
        self.vector_db = None
        self.processed_docs = None
        
    

    @tool("Legal Context Retriever")
    def retrieve_legal_context(self, query: str) -> List[Document]:
        """Retrieve relevant legal context using hybrid search"""
        expanded_query = RAGPipeline().add_query_expansion(query)
        retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 6}),
                BM25Retriever.from_documents(self.processed_docs)
            ],
            weights=[0.7, 0.3]
        )
        return retriever.invoke(expanded_query)

    @tool("Legal Analysis Expert")
    def analyze_legal_question(self, context: List[Document], question: str) -> str:
        """Analyze legal questions with proper citations and interpretations"""
        formatted_context = format_documents(context)
        prompt = ChatPromptTemplate.from_template(
            """Legal Expert Analysis Protocol:
            1. Identify relevant documents and sections
            2. Cross-reference related provisions
            3. Consider jurisdictional specifics (Nepali law)
            4. Provide structured response with sources
            
            Context: {context}
            
            Question: {question}
            
            Required Format:
            - [Document Title], Section [Number]: [Excerpt]
            - Legal Interpretation: [Analysis]
            - Potential Applications: [Use Cases]
            - Related Provisions: [Cross-references]"""
        )
        
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke({"context": formatted_context, "question": question})

# Initialize CrewAI agents
def create_crew():
    # Agent Definitions
    

    research_agent = Agent(
        role="Legal Research Expert",
        goal="Retrieve relevant legal context using hybrid search techniques",
        backstory="Specialist in legal information retrieval with multi-modal search capabilities",
        tools=[LegalRAGAgents().retrieve_legal_context],
        verbose=True,
        llm = "groq/llama-3.3-70b-versatile"
    )

    analysis_agent = Agent(
        role="Legal Analysis Expert",
        goal="Provide accurate legal interpretations with proper citations",
        backstory="Experienced legal professional with expertise in Nepali law",
        tools=[LegalRAGAgents().analyze_legal_question],
        verbose=True,
        llm = "groq/llama-3.3-70b-versatile"
    )

    # Task Definitions

    research_task = Task(
        description="Retrieve relevant legal context for: {question}",
        agent=research_agent,
        expected_output="List of relevant legal provisions with metadata",
        context=[load_task]
    )

    analysis_task = Task(
        description="Analyze the legal question: {question}",
        agent=analysis_agent,
        expected_output="Structured legal analysis with citations and interpretations",
        context=[research_task]
    )

    return Crew(
        agents=[data_loader, research_agent, analysis_agent],
        tasks=[load_task, research_task, analysis_task],
        verbose=True,
    )

# Usage
if __name__ == "__main__":
    legal_crew = create_crew()
    result = legal_crew.kickoff(inputs={"question": "What are the cyberbullying laws in Nepal?"})
    print("\n\nFinal Legal Analysis:")
    print(result)

