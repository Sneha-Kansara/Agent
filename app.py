import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- MODERN 2026 IMPORTS ---
import langchainhub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

# --- TOOLS ---
def ocr_tool(image_file):
    img = Image.open(image_file)
    return pytesseract.image_to_string(img)

def expense_tool(text):
    amount = re.search(r'\d+', text).group() if re.search(r'\d+', text) else "Unknown"
    return f"Amount: ₹{amount}, Category: Analyzed"

def advice_tool(category):
    return "Great tracking! Small savings today lead to big wealth tomorrow."

# --- UI SETUP ---
st.set_page_config(page_title="AI Finance Agent", page_icon="💰")
st.title("💰 AI Finance Agent")

gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# Pre-initialize agent_executor
agent_executor = None

if not gemini_key:
    st.info("Please enter your Gemini API Key in the sidebar.", icon="🗝️")
else:
    try:
        # 1. Initialize Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=gemini_key,
            temperature=0
        )
        
        # 2. Define Tools
        tools = [
            Tool(name="OCR", func=ocr_tool, description="Reads text from images."),
            Tool(name="Analyzer", func=expense_tool, description="Finds amount and category."),
            Tool(name="Advisor", func=advice_tool, description="Provides financial advice.")
        ]

        # 3. Pull Prompt (Check Indentation Here)
        prompt = langchainhub.pull("hwchase17/react")
        
        # 4. Create Agent
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
        
    except Exception as e:
        st.error(f"Initialization Error: {e}")

# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload screenshot (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file and agent_executor:
    st.image(uploaded_file, use_container_width=True)
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):
            try:
                response = agent_executor.invoke({"input": f"Analyze this file: {uploaded_file}"})
                st.success(response["output"])
            except Exception as e:
                st.error(f"Execution Error: {e}")
