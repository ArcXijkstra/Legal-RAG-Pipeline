from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent
import os
from dotenv import load_dotenv

load_dotenv()

# Load the LLM with legal specialization
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.2,  # Slightly higher for legal interpretation
    google_api_key=os.getenv("GOOGLE_API_KEY2"),
    system="You are a legal compliance expert specializing in Nepalese law"
)

# Key Nepal Legal Framework Integration :cite[1]:cite[7]:cite[8]
NEPAL_LAWS = """
Relevant Nepalese Legislation:
1. Companies Act 2063 (Corporate Governance)
2. Consumer Protection Act (Fair Terms)
3. Data Privacy Guidelines (Data Handling)
4. Labor Act 2074 (Employment Clauses)
5. Electronic Transactions Act (Digital Agreements)
6. Foreign Investment Act (Cross-border Terms)
7. Banking and Financial Institutions Act (Payment Terms)
8. Competition Promotion Act (Anti-monopoly)
"""

# Clause Extraction Agent
def extract_clauses(pdf_text):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Identify and list all contractual clauses from the document. "
            "Categorize them as: Payment, Termination, Liability, Data Handling, "
            "Dispute Resolution, Intellectual Property, etc.:\n\n{text}"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(pdf_text)

# Legal Compliance Analyzer
def analyze_legal_compliance(clauses):
    prompt = PromptTemplate(
        input_variables=["clauses", "laws"],
        template=(
            "Analyze these clauses for compliance with Nepalese law:\n{clauses}\n"
            "Legal Framework:\n{laws}\n"
            "For each clause: \n"
            "1. Determine legal status (Compliant/Risky/Non-compliant)\n"
            "2. Identify specific law sections violated (if any)\n"
            "3. Flag potential loopholes\n"
            "4. Highlight enforcement vulnerabilities"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"clauses": clauses, "laws": NEPAL_LAWS})

# Loophole Detection Agent
def detect_loopholes(analysis):
    prompt = PromptTemplate(
        input_variables=["analysis"],
        template=(
            "Identify exploitable loopholes in these analyzed clauses:\n{analysis}\n"
            "Consider:\n"
            "- Ambiguous terminology\n"
            "- Unfair termination clauses\n"
            "- Excessive liability limitations\n"
            "- Non-compliant dispute resolution\n"
            "- Data localization violations\n"
            "- Hidden fee structures"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(analysis)

# Remediation Agent
def generate_remediation(issues):
    prompt = PromptTemplate(
        input_variables=["issues"],
        template=(
            "Create legally compliant alternatives for these issues:\n{issues}\n"
            "Ensure solutions:\n"
            "- Align with Companies Act 2063 :cite[8]\n"
            "- Follow Nepal Rastra Bank guidelines :cite[1]\n"
            "- Meet Labor Act 2074 requirements :cite[8]\n"
            "- Incorporate SEBON regulations where applicable :cite[1]"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(issues)

def convert_to_markdown(output):
    md = "# Terms & Conditions Legal Analysis\n\n"
    md += "## Extracted Clauses\n" + output['clauses'] + "\n\n"
    md += "## Compliance Analysis\n" + output['analysis'] + "\n\n"
    md += "## Loophole Detection\n" + output['loopholes'] + "\n\n"
    md += "## Legal Remediation\n" + output['remediation'] + "\n"
    return md

# Enhanced Workflow
def analyze_terms(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join(p.page_content for p in pages)
    
    clauses = extract_clauses(text)
    analysis = analyze_legal_compliance(clauses)
    loopholes = detect_loopholes(analysis)
    remediation = generate_remediation(loopholes)
    
    return convert_to_markdown({
        "clauses": clauses,
        "analysis": analysis,
        "loopholes": loopholes,
        "remediation": remediation
    })
    
if __name__ == "__main__":
    pdf_path = "sample_terms.pdf"
    output = analyze_terms(pdf_path)
    print(output)