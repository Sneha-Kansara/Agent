import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- STABLE PRODUCTION IMPORTS (2026) ---
# We use 'langchain_classic' to guarantee the AgentExecutor works on Streamlit Cloud
from langchain_classic.agents import AgentExecutor, create_react_agent
import langchainhub as hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# --- CUSTOM FINANCE TOOLS ---

def ocr_tool(image_file):
    """Step 1: Extract text from the uploaded screenshot."""
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

def expense_tool(text):
    """Step 2: Parse text for amount and category."""
    amount_match = re.search(r'\d+', text)
    amount = amount_match.group() if amount_match else "Unknown"
    
    text_lower = text.lower()
    if any(k in text_lower for k in ["swiggy", "zomato", "food", "restaurant"]):
        category = "Food"
    elif any(k in text_lower for k in ["uber", "ola", "transport", "ride"]):
        category = "Transport"
    elif any(k in text_lower for k in ["amazon", "flipkart", "shopping", "order"]):
        category = "Shopping"
    else:
        category = "Others"
        
    return f"Amount: ₹{amount}, Category: {category}"

def advice_tool(category):
    """Step 3: Provide advice based on the category."""
    advice_map = {
        "Food": "Try to reduce dining out to save 20% more this month.",
        "Transport": "Check if public transport or carpooling is an option.",
        "Shopping": "Wait 24 hours before making non-essential purchases.",
        "Others": "Keep tracking these 'Others' to find hidden leaks in your budget."
    }
    return advice_map.get(category, "Review this expense to see if it's necessary.")

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Finance Agent", page_icon="🏦")
st.title("🏦 AI Finance Agent")
st.markdown("Upload your payment screenshots (UPI, Bank, etc.) for instant analysis.")

# Secure API Key Entry
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to start.", icon="🗝️")
else:
    # 1. Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    # 2. Define Tools
    tools = [
        Tool(name="OCR_Tool", func=ocr_tool, description="Extracts text from images."),
        Tool(name="Expense_Analyzer", func=expense_tool, description="Finds amount and category from text."),
        Tool(name="Financial_Advisor", func=advice_tool, description="Provides advice for a category.")
    ]

    # 3. Setup Agent
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # 4. App Logic
    uploaded_file = st.file_uploader("Upload screenshot", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Target Image", use_container_width=True)
        
        if st.button("Analyze Now"):
            with st.spinner("Agent is working..."):
                try:
                    # We invoke the agent with the file object
                    result = agent_executor.invoke({
                        "input": f"Use your tools to analyze this image: {uploaded_file}. Give me a summary of spend, category, and advice."
                    })
                    st.success("### Analysis Complete")
                    st.write(result["output"])
                except Exception as e:
                    st.error(f"Error processing agent: {e}")
