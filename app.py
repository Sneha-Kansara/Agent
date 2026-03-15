import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- STABLE PRODUCTION IMPORTS (2026) ---
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchainhub import pull  # Standard for March 2026
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# --- CUSTOM FINANCE TOOLS (Optimized from your Notebook) ---

def ocr_tool(image_file):
    """Step 1: Extract text from the payment screenshot."""
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
    """Step 3: Provide advice based on the transaction category."""
    advice_map = {
        "Food": "Consider cooking at home more often to save on delivery fees.",
        "Transport": "Look into monthly passes if you travel this route daily.",
        "Shopping": "Check if this was a need or a want before your next purchase.",
        "Others": "Keep an eye on these miscellaneous costs."
    }
    return advice_map.get(category, "Regularly review your spending habits.")

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Finance Agent", page_icon="🏦")
st.title("🏦 AI Finance Agent")
st.markdown("Analyze your payment screenshots and get instant financial advice.")

# Secure API Key Entry in Sidebar
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to start.", icon="🗝️")
else:
    # 1. Initialize Modern LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    # 2. Define Tools using Notebook Logic
    tools = [
        Tool(name="OCR_Tool", func=ocr_tool, description="Use this to extract text from images."),
        Tool(name="Expense_Analyzer", func=expense_tool, description="Detects amount and category."),
        Tool(name="Financial_Advisor", func=advice_tool, description="Provides advice for a category.")
    ]

    # 3. Setup Agent (Using pull and create_react_agent)
    try:
        prompt = pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
    except Exception as e:
        st.error(f"Failed to initialize Agent: {e}")

    # 4. File Upload and Execution
    uploaded_file = st.file_uploader("Upload screenshot (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run Financial Analysis"):
            with st.spinner("Agent is analyzing..."):
                try:
                    # Pass the file object directly to the agent's chain
                    result = agent_executor.invoke({
                        "input": f"Analyze this image: {uploaded_file}. Tell me the spend amount, category, and advice."
                    })
                    st.success("### Analysis Result")
                    st.write(result["output"])
                except Exception as e:
                    st.error(f"Execution Error: {e}")
