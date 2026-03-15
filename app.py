import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- STABLE 2026 IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent

# The definitive import for 'pull' in v1.0+ environments
from langchain_classic import hub
# Usage later will be: hub.pull("hwchase17/react")

from langchain_core.tools import Tool

# Robust Hub Import for Streamlit Cloud
try:
    from langchainhub import pull
except ImportError:
    from langchain.hub import pull

# --- CUSTOM FINANCE TOOLS ---

def ocr_tool(image_file):
    """Extracts text from the uploaded screenshot."""
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

def expense_tool(text):
    """Parses text for amount and category."""
    amount_match = re.search(r'\d+', text)
    amount = amount_match.group() if amount_match else "Unknown"
    
    text_lower = text.lower()
    if any(k in text_lower for k in ["swiggy", "zomato", "food", "blinkit"]):
        category = "Food/Groceries"
    elif any(k in text_lower for k in ["uber", "ola", "rapido", "petrol"]):
        category = "Transport"
    elif any(k in text_lower for k in ["amazon", "flipkart", "myntra"]):
        category = "Shopping"
    else:
        category = "Miscellaneous"
        
    return f"Amount: ₹{amount}, Category: {category}"

def advice_tool(category):
    """Provides financial advice based on the category."""
    advice_map = {
        "Food/Groceries": "Try to plan meals in advance to avoid last-minute delivery costs.",
        "Transport": "Check if a daily pass or carpooling reduces your monthly spend.",
        "Shopping": "Consider the 30-day rule: wait a month before buying non-essentials.",
        "Miscellaneous": "Review these small leaks; they often add up to big savings."
    }
    return advice_map.get(category, "Tracking is the first step to financial freedom!")

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Free AI Finance Agent", page_icon="💰")
st.title("💰 AI Finance Agent")
st.markdown("Analyze your payment screenshots using **Google Gemini**.")

# Sidebar for API Key
gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# Pre-initialize the executor to prevent NameError
agent_executor = None

if not gemini_key:
    st.info("Please enter your free Gemini API Key in the sidebar to start.", icon="🗝️")
else:
    try:
        # 1. Initialize Gemini Model (Free Tier)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=gemini_key,
            temperature=0
        )

        # 2. Define Tools
        tools = [
            Tool(name="OCR_Tool", func=ocr_tool, description="Reads text from images."),
            Tool(name="Expense_Analyzer", func=expense_tool, description="Identifies amount and category."),
            Tool(name="Financial_Advisor", func=advice_tool, description="Gives advice for a category.")
        ]

        # 3. Pull ReAct Prompt and Build Agent
        prompt = pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
    except Exception as e:
        st.error(f"Initialization Error: {e}")

# --- MAIN APP LOGIC ---

uploaded_file = st.file_uploader("Upload screenshot (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Target Screenshot", use_container_width=True)
    
    # Only show button if agent_executor was successfully created
    if agent_executor is not None:
        if st.button("Analyze Spending"):
            with st.spinner("Gemini is analyzing your receipt..."):
                try:
                    result = agent_executor.invoke({
                        "input": f"Analyze this image: {uploaded_file}. Tell me the spend amount, category, and specific advice."
                    })
                    st.success("### Analysis Result")
                    st.write(result["output"])
                except Exception as e:
                    st.error(f"Execution Error: {e}")
    else:
        st.warning("Agent is not initialized. Check your API key and connection.")
