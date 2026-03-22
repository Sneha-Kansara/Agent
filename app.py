import streamlit as st
import pytesseract
from PIL import Image
import re
import os
import pandas as pd

# LangChain imports
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- API KEY ----------------
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # change if needed based on your API
    temperature=0
)

# ---------------- TOOLS ----------------

def expense_tool(text):
    amount_match = re.search(r'\d+', text)
    amount = int(amount_match.group()) if amount_match else 0

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
        "Food": "Reduce online food orders to save money.",
        "Transport": "Use public transport when possible.",
        "Shopping": "Avoid impulse buying.",
        "Others": "Track your expenses regularly."
    }
    return advice.get(category, "Track your spending.")


def budgeting_tool(total):
    total = int(total)

    if total > 10000:
        return "⚠️ You are overspending!"
    elif total > 5000:
        return "⚠️ You are close to budget limit."
    else:
        return "✅ Your spending is under control."


def guru_advice_tool(category):
    guru = {
        "Food": "Warren Buffett: Save first, spend later.",
        "Shopping": "Ramit Sethi: Spend consciously.",
        "Transport": "Avoid unnecessary lifestyle inflation.",
        "Others": "Track every rupee you spend."
    }
    return guru.get(category, "Invest wisely.")

# ---------------- TOOLS LIST ----------------

tools = [
    Tool(name="Expense Analyzer", func=expense_tool,
         description="Extract amount and category"),
    Tool(name="Financial Advisor", func=advice_tool,
         description="Provide advice"),
    Tool(name="Budget Tool", func=budgeting_tool,
         description="Budget advice"),
    Tool(name="Guru Advice", func=guru_advice_tool,
         description="Financial guru advice")
]

# ---------------- AGENT ----------------

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# ---------------- STREAMLIT UI ----------------

st.title("💰 AI Financial Advisor & Expense Manager")

# Input type selection
option = st.selectbox(
    "Select Input Type",
    ["Upload Screenshot", "Manual Entry", "CSV Upload"]
)

expenses = []
categories = []

# ---------------- SCREENSHOT ----------------
if option == "Upload Screenshot":

    uploaded_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        text = pytesseract.image_to_string(image)

        st.subheader("Extracted Text")
        st.write(text)

        result = agent.run(text)
        st.subheader("AI Analysis")
        st.write(result)

# ---------------- MANUAL ENTRY ----------------
elif option == "Manual Entry":

    user_input = st.text_input("Enter expense (e.g., Paid ₹300 to Swiggy)")

    if user_input:
        result = agent.run(user_input)
        st.write(result)

        amount_match = re.search(r'\d+', user_input)
        amount = int(amount_match.group()) if amount_match else 0

        expenses.append(amount)

        if "swiggy" in user_input.lower():
            categories.append("Food")
        else:
            categories.append("Others")

# ---------------- CSV ----------------
elif option == "CSV Upload":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write(df)

        if "Amount" in df.columns:
            total = df["Amount"].sum()
            st.write("Total Spending:", total)

# ---------------- DASHBOARD ----------------

if expenses:

    st.subheader("📊 Spending Analysis")

    df = pd.DataFrame({
        "Category": categories,
        "Amount": expenses
    })

    summary = df.groupby("Category").sum()

    st.bar_chart(summary)

    total_spending = sum(expenses)

    st.subheader("💡 Budget Status")
    st.write(budgeting_tool(total_spending))
