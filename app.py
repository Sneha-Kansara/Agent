import streamlit as st
import pytesseract
from PIL import Image
import re
import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# 1. Define your custom tools (directly from your notebook)
def ocr_tool(image):
    # In Streamlit, 'image' will be a file-like object
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return text

def expense_tool(text):
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
    advice = {
        "Food": "Reduce food delivery spending and cook more at home.",
        "Transport": "Consider public transport to save money.",
        "Shopping": "Avoid impulse buying and track your purchases.",
        "Others": "Review your expenses regularly."
    }
    return advice.get(category, "Track your spending regularly.")

# 2. Streamlit UI Setup
st.title("💰 AI Finance Agent")
st.write("Upload a payment screenshot to analyze your spending.")

# Securely handle API Key
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 3. Initialize Agent Tools
    tools = [
        Tool(name="OCR Tool", func=ocr_tool, description="Extract text from payment screenshot"),
        Tool(name="Expense Analyzer", func=expense_tool, description="Detect amount and category from transaction text"),
        Tool(name="Financial Advisor", func=advice_tool, description="Provide financial advice based on category")
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # 4. File Upload and Processing
    uploaded_file = st.file_file_uploader("Choose a screenshot...", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Screenshot', use_column_width=True)
        if st.button("Analyze Expense"):
            with st.spinner('Analyzing...'):
                # Note: We pass the file object directly to our ocr_tool
                response = agent.run(f"Analyze this image: {uploaded_file}")
                st.success(response)
else:
    st.warning("Please enter your OpenAI API key in the sidebar to begin.")
