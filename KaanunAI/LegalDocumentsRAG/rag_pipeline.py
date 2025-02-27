
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore


load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_template(
#     """You are a legal expert and knows every legal nuances and language, now, Answer the following question based on only the provided context.
#     Think very very carefully before providing a detailed answer. And also site the Reference from the meta data of the document. For example:
#     Chpater 1, Section 2 of "Constitution of Nepal"
#     <context>
#     {context}
#     </context>
#     Question: {input}
#     """
# )

# chain = create_stuff_documents_chain(llm, prompt)

try:
    qdrant_client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
    print("Connected to Qdrant successfully!")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
    exit(1)
    
from langchain_ollama import OllamaEmbeddings


# embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# def retrieve_content(query: str, k: int = 5):
#     """
#     Retrieve relevant content from Qdrant based on the query.
#     """
#     search_results = qdrant_client.query_points(
#         collection_name="legal_documents",
#         query_vector=embeddings.embed_query(query),
#         limit=k,
#         query_filter=models.Filter(
#             must=[
#                 models.FieldCondition(
#                     key="type",
#                     match=models.MatchValue(value="content")
#                 )
#             ]
#         )
#     )
#     return search_results



# def generate_response(query: str, retrieved_content: list):
#     """
#     Generate a response using the LLM based on the retrieved content.
#     """
#     # Combine the retrieved content into a single context string
#     context = "\n\n".join([result.payload["content"] for result in retrieved_content])
    
#     # Create a prompt for the LLM
#     prompt = f"""
#     You are a legal expert and knows every legal nuances and language, now, Answer the following question based on only the provided context.
#     Think very very carefully before providing a detailed answer. And also site the Reference from the meta data of the document. For example:
#     Chpater 1, Section 2 of "Constitution of Nepal
#     Question: {query}

#     Context: {context}

#     Answer:"""

#     # Generate the response using the LLM
#     response = llm.invoke(prompt)
#     return response

# def query_rag_pipeline(query: str):
#     """
#     Query the RAG pipeline and return the response.
#     """
#     # Step 1: Retrieve relevant content from Qdrant
#     retrieved_content = retrieve_content(query)
    
#     # Step 2: Generate a response using the LLM
#     response = generate_response(query, retrieved_content)
#     return response

# if __name__ == "__main__":
#     query = "What is the law for cyber bullying of Children in Nepal"
#     response = query_rag_pipeline(query)
#     print("Response from RAG pipeline:")
#     print(response)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = QdrantVectorStore.from_existing_collection(
    client=qdrant_client,
    collection_name="legal_documents",
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "filter": models.Filter(
            must=[models.FieldCondition(
                key="type",
                match=models.MatchValue(value="content")
            )]
        )
    }
)

prompt = ChatPromptTemplate.from_template(
    """You are a legal expert and knows every legal nuances and language, now, Answer the following question based on only the provided context.
    Think very very carefully before providing a detailed answer. And also site the Reference from the meta data of the document. For example:
    Chpater 1, Section 2 of "Constitution of Nepal"
    <context>
    {context}
    </context>
    Question: {input}
    """
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def query_rag(question: str) -> str:
    return rag_chain.invoke(question)

# Usage
if __name__ == "__main__":
    print(query_rag("What is the law for cyber bullying of Children in Nepal"))