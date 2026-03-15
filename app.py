import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- MODERN LANGCHAIN IMPORTS ---
try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.react.agent import create_react_agent

from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# --- CUSTOM TOOLS (Extracted from your notebook) ---

def ocr_tool(image_file):
    """Extracts text from a payment screenshot."""
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

def expense_tool(text):
    """Detects amount and category from transaction text."""
    amount_match = re.search(r'\d+', text)
    amount = amount_match.group() if amount_match else "Unknown"
    
    text = text.lower()
    if "swiggy" in text or "zomato" in text:
        category = "Food"
    elif "uber" in text or "ola" in text:
        category = "Transport"
    elif "amazon" in text or "flipkart" in text:
        category = "Shopping"
    else:
        category = "Others"
        
    return f"Amount: ₹{amount}, Category: {category}"

def advice_tool(category):
    """Provides financial advice based on the detected category."""
    advice = {
        "Food": "Reduce food delivery spending and cook more at home.",
        "Transport": "Consider public transport to save money.",
        "Shopping": "Avoid impulse buying and track your purchases.",
        "Others": "Review your expenses regularly."
    }
    return advice.get(category, "Track your spending regularly.")

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="AI Finance Agent", page_icon="💰")
st.title("💰 AI Finance Agent")
st.write("Upload a payment screenshot to analyze your spending.")

# Secure API Key Input
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to the sidebar to continue.", icon="🗝️")
else:
    # Initialize the Modern LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    # Define Tools for the Agent
    tools = [
        Tool(
            name="OCR_Tool", 
            func=ocr_tool, 
            description="Use this first to extract text from a payment screenshot image."
        ),
        Tool(
            name="Expense_Analyzer", 
            func=expense_tool, 
            description="Use this to detect the amount and category from extracted text."
        ),
        Tool(
            name="Financial_Advisor", 
            func=advice_tool, 
            description="Use this to provide financial advice based on a category."
        )
    ]

    # Initialize Modern Agent Logic
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    # File Upload Widget
    uploaded_file = st.file_uploader("Upload a screenshot (JPG, PNG, JFIF)", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Screenshot', use_container_width=True)
        
        if st.button("Analyze Expense"):
            with st.spinner('Agent is processing your receipt...'):
                try:
                    # Pass the file object to the agent
                    response = agent_executor.invoke({
                        "input": f"Analyze this transaction image: {uploaded_file}. Tell me the amount, category, and advice."
                    })
                    st.success("### Analysis Result")
                    st.write(response["output"])
                except Exception as e:
                    st.error(f"Error: {e}")
