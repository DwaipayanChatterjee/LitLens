# =========================
# Imports
# =========================
import streamlit as st
from pypdf import PdfReader
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.arxiv import ArxivTools


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="LitLens · AI Research Paper Labs",
    page_icon="🔍",
    layout="centered",
)


# =========================
# Styling
# =========================
st.markdown("""
<style>
.chat {
    background-color: #020617;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    border: 1px solid #1e293b;
}
.agent {
    background-color: #0f172a;
}
.citation {
    font-size: 0.9rem;
    color: #93c5fd;
}
.compare-box {
    background-color: #020617;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def extract_pdf_text(uploaded_files, max_chars=12000):
    """Extract and truncate text from uploaded PDFs"""
    text = ""
    for idx, pdf in enumerate(uploaded_files):
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += f"\n[PDF-{idx+1}] {extracted}"
        if len(text) > max_chars:
            break
    return text[:max_chars]


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 🔍 LitLens")
    st.markdown("### AI Research Paper Labs")
    st.markdown("*Focus your research. Cite with clarity.*")

    st.markdown("---")
    openai_key = st.text_input("🔐 OpenAI API Key", type="password")

    st.markdown("---")
    st.subheader("📄 Upload PDFs")
    uploaded_pdfs = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("""
    **LitLens Labs Features**
    - Chat with arXiv papers
    - Upload and analyze PDFs
    - Inline citations
    - Side-by-side paper comparison
    """)


# =========================
# Session State
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []


# =========================
# Header
# =========================
st.title("🔍 LitLens")
st.caption("AI Research Paper Labs — search, upload, compare, and review academic literature with AI")


# =========================
# Tabs
# =========================
chat_tab, compare_tab = st.tabs(["💬 Chat & Review", "⚖️ Paper Comparison"])


# =========================
# CHAT TAB
# =========================
with chat_tab:

    for msg in st.session_state.chat:
        st.markdown(
            f"<div class='chat agent'>{msg}</div>",
            unsafe_allow_html=True,
        )

    query = st.chat_input("Ask LitLens about a research topic or uploaded PDFs...")

    if query:
        if not openai_key:
            st.warning("Please enter your OpenAI API key.")
            st.stop()

        pdf_context = extract_pdf_text(uploaded_pdfs) if uploaded_pdfs else ""

        with st.spinner("🔎 LitLens Labs is reviewing the literature..."):

            research_agent = Agent(
                model=OpenAIChat(
                    id="gpt-4o",
                    temperature=0.6,
                    max_tokens=1400,
                    api_key=openai_key,
                ),
                tools=[ArxivTools()],
                instructions=f"""
                You are LitLens, an AI research assistant.

                Use the following uploaded PDF content when relevant:
                {pdf_context}

                When answering:
                - Use inline citations like [PDF-1], [arXiv-1]
                - Prefer PDFs over arXiv if both are relevant
                - Be concise, technical, and structured
                """
            )

            response = research_agent.run(query, stream=False)

            citation_agent = Agent(
                model=OpenAIChat(
                    id="gpt-4o",
                    temperature=0.3,
                    max_tokens=800,
                    api_key=openai_key,
                ),
                tools=[ArxivTools()],
                instructions="""
                List all referenced papers clearly.

                Format:
                [PDF-1] Uploaded paper
                [arXiv-1] Title (Year) – link
                """
            )

            citations = citation_agent.run(
                f"Provide references for: {query}", stream=False
            )

        st.session_state.chat.append(response.content)

        st.markdown(
            f"<div class='chat agent'>{response.content}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 📑 References")
        st.markdown(
            f"<div class='citation'>{citations.content}</div>",
            unsafe_allow_html=True,
        )


# =========================
# COMPARISON TAB
# =========================
with compare_tab:

    st.subheader("⚖️ Compare Two Papers with LitLens Labs")

    col1, col2 = st.columns(2)

    with col1:
        paper_a = st.text_input("Paper A (title / arXiv link / topic)")

    with col2:
        paper_b = st.text_input("Paper B (title / arXiv link / topic)")

    if st.button("Compare Papers"):
        if not openai_key:
            st.warning("Please enter your OpenAI API key.")
            st.stop()

        if paper_a and paper_b:
            with st.spinner("⚖️ LitLens Labs is comparing the papers..."):

                comparator_agent = Agent(
                    model=OpenAIChat(
                        id="gpt-4o",
                        temperature=0.5,
                        max_tokens=1600,
                        api_key=openai_key,
                    ),
                    tools=[ArxivTools()],
                    instructions="""
                    Compare two research papers.

                    Structure:
                    - Problem Statement
                    - Methodology
                    - Results
                    - Strengths
                    - Weaknesses
                    - When to use which
                    """
                )

                comparison = comparator_agent.run(
                    f"Compare {paper_a} vs {paper_b}", stream=False
                )

            st.markdown(
                f"<div class='compare-box'>{comparison.content}</div>",
                unsafe_allow_html=True,
            )