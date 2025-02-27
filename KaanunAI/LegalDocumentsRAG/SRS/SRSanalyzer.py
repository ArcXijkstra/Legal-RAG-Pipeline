from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent
import os
from dotenv import load_dotenv

load_dotenv()

# Load the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY2"))

# Feature Extraction Agent
def extract_features(pdf_text):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract all key features and requirements from the following text. "
            "List them clearly and concisely:\n\n{text}"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(pdf_text)

# Compliance Analysis Agent
def analyze_compliance(features):
    prompt = PromptTemplate(
        input_variables=["features"],
        template=(
            "Analyze the following features for compliance with Nepalese laws. "
            "For each feature, specify if it is legal, illegal, or potentially risky. "
            "If there are risks, explain why and mention relevant laws and the sections in Nepal:\n\n{features}"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(features)

# Recommendation Agent
def recommend_alternatives(compliance_analysis):
    prompt = PromptTemplate(
        input_variables=["analysis"],
        template=(
            "Based on the following compliance analysis, provide alternative suggestions "
            "to address any illegal or risky features. Ensure the recommendations are compliant "
            "with Nepalese laws:\n\n{analysis}"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(compliance_analysis)

def convert_to_markdown(output_dict):
    """
    Converts the output of the process_srs function to Markdown format.
    """
    md_content = "# Software Requirement Specification (SRS) Analysis\n\n"
    md_content += "## Extracted Features\n"
    md_content += f"{output_dict['features']}\n\n"
    md_content += "## Compliance Analysis\n"
    md_content += f"{output_dict['analysis']}\n\n"
    md_content += "## Recommendations\n"
    md_content += f"{output_dict['recommendations']}\n"
    return md_content

def process_srstext(text):
    # Step 1: Extract Features
    extracted_features = extract_features(text)
    print("Extracted Features:\n", extracted_features)

    # Step 2: Analyze Compliance
    compliance_analysis = analyze_compliance(extracted_features)
    print("Compliance Analysis:\n", compliance_analysis)

    # Step 3: Recommend Alternatives
    recommendations = recommend_alternatives(compliance_analysis)
    print("Recommendations:\n", recommendations)

    output = {
        "features": extracted_features,
        "analysis": compliance_analysis,
        "recommendations": recommendations,
    }

    markdown_output = convert_to_markdown(output)
    return markdown_output

# Main Workflow
def process_srs(pdf_path):
    # Step 1: Load the SRS PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    pdf_text = "\n".join(page.page_content for page in pages)
    
    # Step 2: Extract Features
    extracted_features = extract_features(pdf_text)
    print("Extracted Features:\n", extracted_features)

    # Step 3: Analyze Compliance
    compliance_analysis = analyze_compliance(extracted_features)
    print("Compliance Analysis:\n", compliance_analysis)

    # Step 4: Recommend Alternatives
    recommendations = recommend_alternatives(compliance_analysis)
    print("Recommendations:\n", recommendations)

    output =  {
        "features": extracted_features,
        "analysis": compliance_analysis,
        "recommendations": recommendations,
    }
    
    markdown_output = convert_to_markdown(output)
    return markdown_output

# Example Usage
# if __name__ == "__main__":
#     pdf_path = "SRS_template.pdf"  # Replace with the path to your PDF
#     result = process_srs(pdf_path)
