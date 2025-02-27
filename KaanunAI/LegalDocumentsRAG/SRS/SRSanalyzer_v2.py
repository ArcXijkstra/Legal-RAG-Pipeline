from typing import TypedDict, List, Optional, Annotated
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
# import { ChatGoogleGenerativeAI } from "@langchain/google-genai"

load_dotenv()

# Set your API keys


import os
os.environ["OPENAI_API_KEY"] = "dummy-key"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY2)

# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="groq/llama-3.3-70b-versatile")
# ======================
# 1. Define State Schema
# ======================
class AgentState(TypedDict):
    document_path: str
    extracted_features: Annotated[List[dict], "extracted_features"]
    legal_analysis: Annotated[List[dict], "legal_analysis"]
    recommendations: Annotated[List[dict], "recommendations"]

# ======================
# 2. Define Tools/Nodes
# ======================


# Nepal Legal Compliance Database (Simplified)
NEPAL_LEGAL_RULES = {
    "cryptocurrency": {
        "status": "illegal",
        "reference": "Nepal Rastra Bank Notice 2022, Penal Code §262A",
        "penalty": "Up to 5 years imprisonment and fines",
        "alternatives": ["eSewa", "Khalti", "Bank Transfer"]
    },
    "paypal": {
        "status": "unapproved",
        "reference": "Foreign Exchange Regulation Act 2019",
        "risk": "Potential currency violation",
        "alternatives": ["ConnectIPS", "FonePay"]
    }
}

def extract_features(state: AgentState):
    """Agent 1: Feature Extraction from PDF"""
    loader = PyPDFLoader(state["document_path"])
    pages = loader.load_and_split()
    
    # Extract features using LLM
    feature_prompt = ChatPromptTemplate.from_template("""
    Analyze this SRS document and list all technical features/requirements.
    Focus on payment systems, data handling, and user authentication.
    Format: {{"feature": "...", "category": "...", "criticality": "high/medium/low"}}
    
    Document Content:
    {content}
    """)
    
    feature_chain = feature_prompt | llm | JsonOutputParser()
    features = []
    for page in pages:
        features.extend(feature_chain.invoke({"content": page.page_content}))
    
    return {"extracted_features": features}

def legal_analysis(state: AgentState):
    """Agent 2: Legal Compliance Checker"""
    analysis = []
    
    for feature in state["extracted_features"]:
        # Rule-based check
        legal_status = {"status": "compliant", "issues": []}
        
        # Check against known legal rules
        if "crypto" in feature["feature"].lower():
            legal_status = {
                "status": "non-compliant",
                "issues": [NEPAL_LEGAL_RULES["cryptocurrency"]],
                "suggested_action": "Remove or replace this feature"
            }
        elif "paypal" in feature["feature"].lower():
            legal_status = {
                "status": "risky",
                "issues": [NEPAL_LEGAL_RULES["paypal"]],
                "suggested_action": "Seek NRB approval or use alternatives"
            }
        
        # LLM-based nuanced check
        nuance_prompt = ChatPromptTemplate.from_template("""
        Could this feature violate any Nepal laws beyond payment regulations?
        Consider privacy, cybersecurity, and business laws.
        
        Feature: {feature}
        Current analysis: {analysis}
        
        Respond with {{"additional_risks": "...", "severity": "low/medium/high"}}
        """)
        
        nuance_analysis = nuance_prompt | llm | JsonOutputParser()
        extra_analysis = nuance_analysis.invoke({
            "feature": feature,
            "analysis": legal_status
        })
        
        analysis.append({
            **feature,
            "legal_status": {**legal_status, **extra_analysis}
        })
    
    return {"legal_analysis": analysis}

def generate_recommendations(state: AgentState):
    """Agent 3: Recommendation Generator"""
    recommendations = []
    
    for item in state["legal_analysis"]:
        if item["legal_status"]["status"] != "compliant":
            rec_prompt = ChatPromptTemplate.from_template("""
            For this non-compliant feature in Nepal, suggest technical alternatives and modifications.
            Legal issues: {issues}
            
            Current feature: {feature}
            Category: {category}
            
            Provide:
            - Alternative implementation
            - Required technical changes
            - Compliance steps
            """)
            
            rec_chain = rec_prompt | llm | JsonOutputParser()
            recommendation = rec_chain.invoke({
                "issues": item["legal_status"]["issues"],
                "feature": item["feature"],
                "category": item["category"]
            })
            
            recommendations.append({
                "original_feature": item["feature"],
                **recommendation
            })
    
    return {"recommendations": recommendations}

# ======================
# 3. Build Workflow Graph
# ======================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("feature_extractor", extract_features)
workflow.add_node("legal_analyst", legal_analysis)
workflow.add_node("recommendation_engine", generate_recommendations)

# Set edges
workflow.set_entry_point("feature_extractor")
workflow.add_edge("feature_extractor", "legal_analyst")
workflow.add_edge("legal_analyst", "recommendation_engine")
workflow.add_edge("recommendation_engine", END)

# ======================
# 4. Compile and Execute
# ======================
agent = workflow.compile()

# Run with sample PDF
result = agent.invoke({
    "document_path": "SRS_template.pdf",
    "extracted_features": [],
    "legal_analysis": [],
    "recommendations": []
})

# ======================
# 5. Output Formatter
# ======================
def format_report(result):
    report = []
    for feature, analysis, recommendation in zip(
        result["extracted_features"],
        result["legal_analysis"],
        result["recommendations"]
    ):
        report.append(f"""
        Feature: {feature['feature']}
        Category: {feature['category']}
        Criticality: {feature['criticality']}
        
        Legal Status: {analysis['legal_status']['status']}
        Issues: {[issue['reference'] for issue in analysis['legal_status']['issues']]}
        Risks: {analysis['legal_status'].get('additional_risks', 'None identified')}
        
        Recommendation: {recommendation['Alternative implementation']}
        Required Changes: {recommendation['Required technical changes']}
        Compliance Steps: {recommendation['Compliance steps']}
        """)
    return "\n".join(report)

print(format_report(result))