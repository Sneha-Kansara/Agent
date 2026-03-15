import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- GOOGLE GEMINI IMPORTS (2026) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
import langchainhub as hub 
from langchain_core.tools import Tool

# --- CUSTOM FINANCE TOOLS ---

def ocr_tool(image_file):
    """Step 1: Extract text from the image."""
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
        "Food": "Cook at home to save on delivery fees this week!",
        "Transport": "Check if a monthly pass or carpool is cheaper.",
        "Shopping": "Wait 24 hours before buying non-essentials.",
        "Others": "Review these small costs; they add up fast!"
    }
    return advice_map.get(category, "Always review your expenses weekly.")

# --- STREAMLIT UI ---

st.set_page_config(page_title="Free AI Finance Agent", page_icon="🏦")
st.title("🏦 AI Finance Agent (Powered by Gemini)")
st.markdown("Analyze your payment screenshots for **free** using Google Gemini.")

# Sidebar for Gemini Key
gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not gemini_key:
    st.info("Get your free key at [Google AI Studio](https://aistudio.google.com/)", icon="🗝️")
else:
    # 1. Initialize Gemini Model
    # 'gemini-1.5-flash' is fast and free
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=gemini_key,
        temperature=0
    )

    # 2. Define Tools
    tools = [
        Tool(name="OCR_Tool", func=ocr_tool, description="Extracts text from images."),
        Tool(name="Expense_Analyzer", func=expense_tool, description="Finds amount and category from text."),
        Tool(name="Financial_Advisor", func=advice_tool, description="Provides advice for a category.")
    ]

    # 3. Setup Agent
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
        st.error(f"Error loading Agent: {e}")

    # 4. App Logic
    uploaded_file = st.file_uploader("Upload screenshot", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Target Screenshot", use_container_width=True)
        
        if st.button("Analyze Spend"):
            with st.spinner("Gemini is thinking..."):
                try:
                    # Agent invoke
                    result = agent_executor.invoke({
                        "input": f"Analyze this image: {uploaded_file}. Summarize spend, category, and advice."
                    })
                    st.success("### Analysis Complete")
                    st.write(result["output"])
                except Exception as e:
                    st.error(f"Execution Error: {e}")
