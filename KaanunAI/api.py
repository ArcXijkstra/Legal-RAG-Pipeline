from fastapi import FastAPI
from pydantic import BaseModel
from LegalDocumentsRAG.RAGwithoutAgent import RAGPipeline  # Assuming your RAG code is in rag_pipeline.py
import os
from fastapi.middleware.cors import CORSMiddleware
from LegalDocumentsRAG.SRS.SRSanalyzer import process_srs, process_srstext
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  # Frontend running on localhost:3000
    "https://myfrontend.com",  # Example of a production frontend
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Request model for questions
class QuestionRequest(BaseModel):
    question: str

# Initialize RAG pipeline once during startup
rag = RAGPipeline()

# Load documents and initialize vector store
try:
    print("Initializing RAG pipeline...")
    documents = rag.load_chunked_data()
    processed_docs = rag.process_documents(documents)
    
    # Check for existing vector store
    if not os.path.exists("C:/genAI/genai-api-v1/LegalDocumentsRAG/faiss_index"):
        print("Wrong path to vector store")

        # print("Creating new vector store...")
        # rag.initialize_vectorstore(processed_docs)
        # rag.save_vectorstore()
    else:
        print("Loading existing vector store...")
        rag.load_vectorstore()
    
    # Build the conversation chain
    rag_chain = rag.build_chain()
    print("API ready to receive questions")
except Exception as e:
    print(f"Initialization error: {str(e)}")
    raise RuntimeError("Failed to initialize RAG pipeline") from e

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Endpoint to handle legal questions about Nepali Cyber Crime law
    """
    print(request)
    try:
        result = rag_chain.invoke(str(request.question))
        return {"answer": result}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

@app.get("/health")
def health_check():
    """Endpoint for service health verification"""
    return {"status": "healthy"}

@app.post("/srsanalyser")
def srs_analyser(request: QuestionRequest):
    """
    Endpoint to process SRS PDF file
    """
    try:
        result = process_srs(request.question)
        return {"answer": result}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

# @app.post("/srsfileupload")
# async def process_srs_file(file: UploadFile = File(...)):
#     """
#     Endpoint to process SRS PDF file
#     """
import shutil

@app.post("/srsfileupload")
async def process_srs_file(file: UploadFile = File(...)):
    """
    Endpoint to process SRS PDF file
    """
    try:
        # Save the file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # Call your process_srs function with the file path
        result = process_srs(temp_file_path)

        # Clean up the temporary file after processing
        os.remove(temp_file_path)
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    # try:
        # print("File:", file)
        # print("File name:", file.filename)
        
        # with open("uploadedfile.pdf", "wb+") as f:
        #     f.write(await file.read())
        # # Read file contents
        # # return process_srs("uploadedfile.pdf")
        # with open("uploadedfile.pdf", "rb") as f:
        #     contents = f.read()
        #     result = process_srstext(contents.decode("utf-8"))
        #     return {"result": result}
    #     return {"answer": str(process_srs(file.filename))}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/upload-srs/")
def upload_SRS(request: QuestionRequest):
    if request.question == True:
        return {"answer": str(process_srs('SRS_template.pdf'))}


    # try:
    #     # Save the uploaded file temporarily
    #     temp_dir = Path("temp_files")
    #     temp_dir.mkdir(exist_ok=True)
    #     contents = file.file.read()
        
    #     file_path = f"{temp_dir}/{file.filename}"
        

    #     # with open(file_path, "wb") as f:
    #     #     f.write(await file.read())

    #     # Pass the saved file path to the process function
    #     result = process_srstext(contents.decode("utf-8"))
    #     # Clean up temporary file
    #     file_path.unlink()

    #     return {"result": result}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    
# To run locally: uvicorn api:app --reload

class TextRequest(BaseModel):
    text: str

@app.post("/process-text/")
async def process_text_endpoint(request: TextRequest):
    try:
        # Call the processing function
        result = process_srstext(request.text)
        return {"markdown_output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")