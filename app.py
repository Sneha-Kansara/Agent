import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- MODERN 2026 IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

# Fix for the 'pull' error: Import the function directly
try:
    from langchainhub import pull
except ImportError:
    from langchain.hub import pull

# --- CUSTOM FINANCE TOOLS ---

def ocr_tool(image_file):
    """Step 1: Extract text using Tesseract."""
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"OCR Error: Ensure tesseract-ocr is in packages.txt. Details: {e}"

def expense_tool(text):
    """Step 2: Parse amount and category."""
    amount_match = re.search(r'\d+', text)
    amount = amount_match.group() if amount_match else "Unknown"
    
    text_lower = text.lower()
    if any(k in text_lower for k in ["swiggy", "zomato", "food", "blinkit"]):
        category = "Food/Groceries"
    elif any(k in text_lower for k in ["uber", "ola", "rapido", "petrol"]):
        category = "Transport"
    else:
        category = "Miscellaneous"
        
    return f"Amount: ₹{amount}, Category: {category}"

def advice_tool(category):
    """Step 3: Provide financial advice."""
    advice_map = {
        "Food/Groceries": "Planning meals ahead can save up to 30% on food costs!",
        "Transport": "Consider carpooling or public transport for this route.",
        "Miscellaneous": "Small recurring expenses often hide the biggest savings."
    }
    return advice_map.get(category, "Review your daily spending to find saving opportunities.")

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Finance Agent", page_icon="💰")
st.title("💰 AI Finance Agent")
st.markdown("Upload your payment screenshots for instant analysis.")

# Sidebar for Gemini Key
gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# Pre-initialize variables
agent_executor = None

if not gemini_key:
    st.info("Please enter your Gemini API Key in the sidebar.", icon="🗝️")
else:
    try:
        # 1. Initialize Gemini Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=gemini_key,
            temperature=0
        )

        # 2. Define Tools
        tools = [
            Tool(name="OCR_Tool", func=ocr_tool, description="Reads text from images."),
            Tool(name="Expense_Analyzer", func=expense_tool, description="Finds amount and category."),
            Tool(name="Financial_Advisor", func=advice_tool, description="Provides advice based on category.")
        ]

        # 3. Pull ReAct Prompt and Build Agent
        # We imported 'pull' directly, so we call it directly here
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

uploaded_file = st.file_uploader("Upload screenshot", type=["jpg", "png", "jpeg"])

if uploaded_file and agent_executor:
    st.image(uploaded_file, caption="Target Screenshot", use_container_width=True)
    
    if st.button("Analyze Spending"):
        with st.spinner("Agent is processing..."):
            try:
                # Invoke the agent
                result = agent_executor.invoke({
                    "input": f"Analyze this image: {uploaded_file}. Tell me the spend amount, category, and advice."
                })
                st.success("### Analysis Result")
                st.write(result["output"])
            except Exception as e:
                st.error(f"Execution Error: {e}")
elif uploaded_file and not agent_executor:
    st.warning("Please provide a valid API key to enable the agent.")
