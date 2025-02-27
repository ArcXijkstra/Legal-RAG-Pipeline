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


prompttemplate1 = '''
You are a legal expert and knows every legal nuances and language, now, Answer the following question based on only the provided context.
    Think very very carefully before providing a detailed answer. And also site the Reference from the meta data of the document. For example:
    Chpater 1, Section 2 of "Constitution of Nepal"
     <context>
     {context}
     </context>
     Question: {question}
'''

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)
llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")



def format_documents(docs: list) -> str:
        """Convert list of Documents to formatted string with metadata"""
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Section: {doc.metadata.get('section_title', 'N/A')}\n"
            f"Content: {doc.page_content[:500]}..."
            for doc in docs
        )
        

class RAGPipeline:
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.llm = llm
        self.chunk_dir = os.path.abspath(os.path.join("Data", "Chunked"))
        self.prompt = ChatPromptTemplate.from_template(
            '''
            You are a legal expert specializing in Nepali law. Analyze the context thoroughly and provide:
            1. Specific legal provisions from official documents
            2. Relevant section numbers and document titles
            3. Interpretation:
            If the query is question about information about some actions, also mention the legal penalties or consequences.
            If multiple laws apply, list them all. If uncertain, state "Based on available information:" 
            followed by the most relevant provisions.
            
            Context: {context}
            
            Question: {question}
            
            Answer in this format:
            [Document Title], [Section Number] - [Provision Summary]
            [Interpretation/Application]
            '''
        )
        # Custom prompt template with context guidance
        # self.prompt = ChatPromptTemplate.from_template(
        #     """Answer based on context only:
        #     Context: {context}
            
        #     Question: {question}
            
        #     If unsure, say "I don't know". Keep answers under 3 sentences."""
        # )

    # def load_chunked_data(self, chunk_dir: str):
    #     """Load pre-chunked JSON data using metadata-aware loader"""
    #     loader = JSONLoader(
    #         # current_dir = os.path.dirname(os.path.realpath(__file__))
    #         file_path=os.path.join(chunk_dir, "*.json"),
    #         jq_schema='.[] | {section_title: .section_title, content: .content | join(" ")}',
    #         text_content=False
    #     )
    #     return loader.load()
    
    # def load_chunked_data(self):
    #     """Load chunked JSON files from Data/Chunked directory"""
    #     current_dir = os.path.dirname(os.path.realpath(__file__))
    #     loader = JSONLoader(
    #         file_path=os.path.join(current_dir,"/Data/Chunked/*.json"),
    #         jq_schema='.[] | {section_title: .section_title, content: .content | join(" ")}',
    #         text_content=False
    #     )
    #     return loader.load()
    
    def load_chunked_data(self):
        """Handle Windows path formatting and multiple files"""
        # Verify directory exists
        if not os.path.exists(self.chunk_dir):
            raise FileNotFoundError(f"Chunk directory not found at: {self.chunk_dir}")

        # Get all JSON files
        json_files = [os.path.join(self.chunk_dir, f) 
                     for f in os.listdir(self.chunk_dir) 
                     if f.endswith(".json")]

        if not json_files:
            raise ValueError(f"No JSON files found in {self.chunk_dir}")

        # Load documents from all files
        documents = []
        for file_path in json_files:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.[] | {section_title: .section_title, content: .content | join(" ")}',
                text_content=False
            )
            documents.extend(loader.load())

        print(f"Loaded {len(documents)} chunks from {len(json_files)} files")
        return documents

    def process_documents(self, documents):
        """Optional text refinement and splitting"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)
        # return documents

    def initialize_vectorstore(self, documents):
        """Create ChromaDB vector store with metadata preservation"""
        
        # embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        # self.vector_store = Chroma.from_documents(
        #     documents=documents,
        #     embedding=embeddings,
        #     persist_directory="./chroma_db"
        # )
        # self.retriever = self.vector_store.as_retriever(
        #     search_type="mmr",  # Maximal Marginal Relevance
        #     search_kwargs={"k": 5}
        # )
        
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        self.vector_db = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        # self.retriever = self.vector_db.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k": 6}
        # )
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 2  # Number of keyword matches to retrieve
        
        # Configure vector retriever
        vector_retriever = self.vector_db.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": 8, "lambda_mult": 0.9}
        )
        
        # Create ensemble retriever
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.8, 0.2]
        )
    
    def add_query_expansion(self, query: str) -> str:
        """Expand legal queries with common terminology variants"""
        expansion_rules = {
        # Core cyberbullying terms
        "cyberbullying": [
            "online harassment", "digital defamation", "internet abuse",
            "social media bullying", "cyber stalking", "trolling",
            "doxing", "online shaming", "साइबर उत्पीडन"
        ],
        
        # Legal framework terms
        "laws": [
            "acts", "regulations", "provisions", "directives",
            "नियमावली", "कानून", "विधान"
        ],
        
        # Nepal-specific references
        "Nepal": [
            "Nepalese", "Federal Democratic Republic of Nepal",
            "नेपाल सरकार", "नेपाली कानून", "नेपाली दण्ड संहिता"
        ],
        
        # Related cyber laws
        "cyber crime": [
            "Electronic Transactions Act 2063", "IT Policy 2072",
            "Data Protection Bill", "Privacy Act",
            "साइबर कानून", "इलेक्ट्रोनिक लेनदेन ऐन"
        ],
        
        # Legal procedures
        "prosecution": [
            "penalties", "punishments", "fines", "imprisonment",
            "FIR registration", "cyber crime investigation",
            "दण्ड", "सजाय", "अनुसन्धान"
        ],
        
        # Government bodies
        "authorities": [
            "Nepal Police Cyber Bureau", "Ministry of Communication",
            "Supreme Court of Nepal", "नेपाल प्रहरी",
            "साइबर ब्युरो"
        ],
        
        # Common legal sections
        "Section 47": [
            "ETA Section 47", "Electronic Transactions Act Section 47",
            "cyber defamation provisions", "सिट्टा ४७"
        ],
        
        # International frameworks
        "convention": [
            "Budapest Convention", "ICCPR",
            "SAARC Cyber Law Framework"
        ]
    }

    # Add multilingual expansion
        for term, expansions in expansion_rules.items():
            if term in query.lower():
                query += " " + " ".join(expansions)
        return query
        
    def save_vectorstore(self, path="./faiss_index"):
        """Save FAISS index to disk"""
        self.vector_db.save_local(path)

    def load_vectorstore(self, path="./faiss_index"):
        """Load existing FAISS index"""
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vector_db = FAISS.load_local(
            folder_path=path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_db.as_retriever()

    
    def build_chain(self):
        """LCEL composition with metadata handling"""
        return (
            {
                "context": RunnableLambda(lambda x: self.retriever.get_relevant_documents(x)) 
                           | RunnableLambda(format_documents),
                "question": RunnableLambda(self.add_query_expansion)
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )


# Usage pipeline
# if __name__ == "__main__":
#     rag = RAGPipeline()
    
#     # 1. Load pre-chunked data
#     documents = rag.load_chunked_data("C:/genAI/genai-api-v1/LegalDocumentsRAG/Data/Chunked")
    
#     # 2. Optional secondary processing
#     processed_docs = rag.process_documents(documents)
    
#     # 3. Initialize vector store
#     rag.initialize_vectorstore(processed_docs)
    
#     # 4. Build and test chain
#     rag_chain = rag.build_chain()
#     print(rag_chain.invoke("What are the laws for cyberbullying in Nepal?"))



if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join("Data", "Chunked"))
    print(f"Expected chunk directory: {base_path}")
    print(f"Directory exists: {os.path.exists(base_path)}")
    
    if os.path.exists(base_path):
        print(f"Files found: {len(os.listdir(base_path))}")
    rag = RAGPipeline()
    
    
    # Load chunked data
    documents = rag.load_chunked_data()
    
    # Optional processing
    processed_docs = rag.process_documents(documents)
    
    # Initialize FAISS (either create new or load existing)
    if not os.path.exists("./faiss_index"):
        rag.initialize_vectorstore(processed_docs)
        rag.save_vectorstore()
    else:
        rag.load_vectorstore()
    
    # Build and test the chain
    rag_chain = rag.build_chain()
    print(rag_chain.invoke("dowry system"))