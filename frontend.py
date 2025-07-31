import streamlit as st
from regulatory_agent import RAGAgent, ResponseAgent, OrchestratorAgent, FormatRouter  # adjust imports
from streamlit_chat import message

@st.cache_resource  # persists across reruns
def load_orchestrator():
    pdfs = [
        "FDA_Policy_Device_Software_Functions.pdf",
        "WHO_Medical_Device_Regulations.pdf",
        "FDA_Design_Control_Guidance.pdf",
    ]
    rag = RAGAgent(pdf_filenames=pdfs, db_path="./chroma_db", )  # ensure it loads existing DB, does NOT rebuild each call
    responder = ResponseAgent()   # temperature=0 in your class
    router = FormatRouter(responder.llm)  # your generic format router
    return OrchestratorAgent(rag, responder, router)

orchestrator = load_orchestrator()

def process_input(user_input: str):
    return orchestrator.handle(user_input)
st.header("Regulatory Chatbot")
chat_container = st.container()

with st.form("chat", clear_on_submit=True):
    user_input = st.text_input("Type your message and press Enter to send.")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    answer = process_input(user_input.strip())
    st.session_state.setdefault("past", []).append(user_input.strip())
    st.session_state.setdefault("generated", []).append(answer)

if "generated" in st.session_state:
    with chat_container:
        for i, (u, a) in enumerate(zip(st.session_state["past"], st.session_state["generated"])):
            message(u, is_user=True, key=f"u{i}")
            message(a, key=f"a{i}")