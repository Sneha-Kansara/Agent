import streamlit as st
import pytesseract
from PIL import Image
import re
import os

# --- STABLE 2026 IMPORTS ---
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

# --- FINANCE TOOLS ---

def ocr_tool(image_file):
    """Extracts text from the image using Tesseract."""
    img = Image.open(image_file)
    return pytesseract.image_to_string(img)

def expense_tool(text):
    """Parses text for amount and category."""
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
    """Provides financial advice."""
    advice_map = {
        "Food/Groceries": "Planning meals ahead can save you a lot on delivery fees!",
        "Transport": "Check for monthly passes to save on commute costs.",
        "Miscellaneous": "Small spends add up—track these for a week."
    }
    return advice_map.get(category, "Review your spending to find saving gaps.")

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Finance Agent", page_icon="💰")
st.title("💰 AI Finance Agent")
st.markdown("Powered by **Google Gemini**.")

# Sidebar for API Key
gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# Pre-initialize agent_executor
agent_executor = None

if not gemini_key:
    st.info("Please enter your Gemini API Key in the sidebar.", icon="🗝️")
else:
    try:
        # 1. Initialize Gemini Model 
        # FIX: Using the string name directly usually resolves the v1beta 404
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

        # 3. Pull Prompt (Using LangSmith client)
        client = Client()
        prompt = client.pull_prompt("hwchase17/react")
        
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

# --- MAIN APP LOGIC ---

uploaded_file = st.file_uploader("Upload payment screenshot", type=["jpg", "png", "jpeg"])

if uploaded_file and agent_executor:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Run Financial Analysis"):
        with st.spinner("Analyzing..."):
            try:
                # Process the file via the Agent
                # We send the OCR text result to help the agent context
                raw_text = ocr_tool(uploaded_file)
                result = agent_executor.invoke({
                    "input": f"Analyze this text from a receipt: {raw_text}. Identify amount, category, and give advice."
                })
                st.success("### Analysis Result")
                st.write(result["output"])
            except Exception as e:
                st.error(f"Execution Error: {e}")
