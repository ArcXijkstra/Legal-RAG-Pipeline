# First install required packages:
# pip install langchain-core langchain-community groq ollama faiss-cpu python-dotenv

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()
current_dir = os.path.dirname(os.path.realpath(__file__))
# Initialize components with latest LangChain syntax
def initialize_components():
    # MXBAI embeddings via Ollama
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Load FAISS vector store
    vector_store = FAISS.load_local(
        os.path.join(current_dir,"faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True,
        
    )
    
    # Groq LLM (using OpenAI-compatible API)
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")
    
    return vector_store, llm

# Custom prompt template
prompt_template = PromptTemplate.from_template("""
You are a legal expert specialized in Nepali Cyber Crime Law. Use only the provided context.

Analyze and provide:
1. Exact legal provisions from context
2. Document titles and section numbers
3. Clear interpretation

Context: {context}

Question: {question}

Format:
[Document Title], Section [Number] - [Exact Provision from Context if Available, or Summary specifying 'Summary:']
Interpretation: [Analysis]
---
Confidence: [High/Medium/Low]""")

# Create RAG chain
def create_rag_chain():
    vector_store, llm = initialize_components()
    
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        # search_kwargs={
        #     "k": 4,
        #     "score_threshold": 0.68,
        #     # "filter": {"source": "cyber_laws"}
        # }
    )
    
    return create_retrieval_chain(retriever, document_chain)

# Query handler
def legal_query(question: str):
    rag_chain = create_rag_chain()
    
    try:
        response = rag_chain.invoke({"input": question})
        
        if not response["context"]:
            return "No relevant provisions found. Consult legal texts directly."
            
        return response["answer"]
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Example usage
if __name__ == "__main__":
    query = "What are the penalties for online fraud under Nepali law?"
    result = legal_query(query)
    print(result)