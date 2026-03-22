import streamlit as st
import pytesseract
from PIL import Image
import re

# --- MODERN 2026 IMPORTS ---
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

# --- FINANCE TOOLS ---
def ocr_tool(image_file):
    img = Image.open(image_file)
    return pytesseract.image_to_string(img)

def expense_tool(text):
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
    return "Great tracking! Small savings today lead to big wealth tomorrow."

# --- UI SETUP ---
st.set_page_config(page_title="AI Finance Agent", page_icon="💰")
st.title("💰 AI Finance Agent")

gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
agent_executor = None

if not gemini_key:
    st.info("Please enter your Gemini API Key in the sidebar.", icon="🗝️")
else:
    try:
        # 1. Initialize Gemini Flash-Lite (Faster & higher limits)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=gemini_key,
            temperature=0,
            max_retries=6,  # AUTO-RETRY ON 429 ERRORS
            timeout=60
        )

        # 2. Define Tools
        tools = [
            Tool(name="OCR", func=ocr_tool, description="Extracts text from images."),
            Tool(name="Analyzer", func=expense_tool, description="Finds amount and category."),
            Tool(name="Advisor", func=advice_tool, description="Provides advice.")
        ]

        # 3. Setup Agent
        client = Client()
        prompt = client.pull_prompt("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=5  # Prevent runaway loops that waste quota
        )
    except Exception as e:
        st.error(f"Initialization Error: {e}")

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload screenshot", type=["jpg", "png", "jpeg"])

if uploaded_file and agent_executor:
    st.image(uploaded_file, use_container_width=True)
    
    if st.button("Analyze Now"):
        with st.spinner("Processing... (Waiting for quota if needed)"):
            try:
                # Pre-run OCR to save one agent loop/request
                extracted_text = ocr_tool(uploaded_file)
                
                result = agent_executor.invoke({
                    "input": f"Analyze this receipt text: '{extracted_text}'. Get amount, category, and advice."
                })
                st.success("### Analysis Result")
                st.write(result["output"])
            except Exception as e:
                st.error(f"Execution Error: {e}")
