import streamlit as st
import pytesseract
from PIL import Image
import re
import pandas as pd

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

# --------- NEW FEATURES ---------

def budgeting_tool(total):
    total = int(total)

    if total > 10000:
        return "⚠️ You are overspending!"
    elif total > 5000:
        return "⚠️ You are near your budget limit."
    else:
        return "✅ Your spending is under control."

def guru_advice_tool(category):
    guru = {
        "Food": "Warren Buffett: Save before you spend.",
        "Shopping": "Ramit Sethi: Spend consciously.",
        "Transport": "Avoid unnecessary expenses.",
        "Others": "Track every rupee."
    }
    return guru.get(category, "Invest wisely.")

# --- UI SETUP ---
st.set_page_config(page_title="AI Finance Agent", page_icon="💰")
st.title("💰 AI Finance Agent")

gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
agent_executor = None

if not gemini_key:
    st.info("Please enter your Gemini API Key in the sidebar.", icon="🗝️")
else:
    try:
        # 1. Initialize Gemini
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
            Tool(name="Advisor", func=advice_tool, description="Provides advice."),
            Tool(name="Budget Tool", func=budgeting_tool, description="Gives budgeting advice"),
            Tool(name="Guru Advice", func=guru_advice_tool, description="Financial guru advice")
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
            max_iterations=5
        )

    except Exception as e:
        st.error(f"Initialization Error: {e}")

# ---------------- MAIN UI ----------------

option = st.selectbox(
    "Select Input Type",
    ["Screenshot", "Manual Entry", "CSV Upload"]
)

# -------- SCREENSHOT --------
if option == "Screenshot":
    uploaded_file = st.file_uploader("Upload Screenshot", type=["jpg", "png", "jpeg"])

    if uploaded_file and agent_executor:
        st.image(uploaded_file, use_container_width=True)

        if st.button("Analyze Now"):
            with st.spinner("Processing..."):
                try:
                    extracted_text = ocr_tool(uploaded_file)

                    result = agent_executor.invoke({
                        "input": f"Analyze this receipt text: '{extracted_text}'. Get amount, category, and advice."
                    })

                    st.success("### Analysis Result")
                    st.write(result["output"])

                except Exception as e:
                    st.error(f"Execution Error: {e}")

# -------- MANUAL ENTRY --------
elif option == "Manual Entry":
    user_input = st.text_input("Enter expense (e.g., Paid ₹300 to Swiggy)")

    if user_input and agent_executor:
        try:
            result = agent_executor.invoke({
                "input": f"Analyze this text: '{user_input}'"
            })
            st.write(result["output"])
        except Exception as e:
            st.error(f"Error: {e}")

# -------- CSV --------
elif option == "CSV Upload":
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("Uploaded Data:", df)

        if "Amount" in df.columns:
            total = df["Amount"].sum()
            st.write("Total Spending:", total)
            st.write("Budget Advice:", budgeting_tool(total))

# -------- CHART --------
sample_data = pd.DataFrame({
    "Category": ["Food", "Shopping", "Transport"],
    "Amount": [2000, 1500, 1000]
})

st.subheader("📊 Spending Overview")
st.bar_chart(sample_data.set_index("Category"))
