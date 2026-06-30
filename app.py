import streamlit as st
import streamlit.components.v1 as components

from neo4j import GraphDatabase
from dotenv import load_dotenv

from crawler import crawl
from nlp import run_nlp_pipeline
from graph_pipeline import run_graph_pipeline
from visualization import generate_visualization
from qa_graph import answer_question

import os

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Graphitti",
    page_icon="🕸️",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.main-title{
    font-size:42px;
    font-weight:700;
    color:#3DDC84;
}

.sub-title{
    color:#A0AEC0;
    font-size:18px;
}

.progress-row{
    background:#111827;
    padding:10px;
    border-radius:8px;
    margin-bottom:8px;
}

.check{
    color:#3DDC84;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph_loaded" not in st.session_state:
    st.session_state.graph_loaded = False

# --------------------------------------------------
# Neo4j Connection
# --------------------------------------------------
driver = None
connected = False

try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    connected = True

except Exception:
    connected = False

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.title("⚙️ Graphitti")

    if connected:
        st.success("Neo4j Connected")
    else:
        st.error("Neo4j Not Connected")

    if st.button("🗑 Clear Chat"):
     st.session_state.messages = []

    if st.button("♻️ Reset Pipeline"):
     st.session_state.pipeline_done = False
     st.session_state.messages = []

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🕸️ Graphitti</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Graph Native Web Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# URL Input
# --------------------------------------------------
urls = st.text_area(
    "Enter one URL per line",
    height=180,
    placeholder="""https://en.wikipedia.org/wiki/Spider-Man

https://en.wikipedia.org/wiki/Marvel_Comics"""
)

run_pipeline = st.button(
    "🚀 Run Pipeline",
    use_container_width=True
)

# --------------------------------------------------
# Run Pipeline
# --------------------------------------------------

if run_pipeline:

    url_list = [u.strip() for u in urls.splitlines() if u.strip()]

    if len(url_list) == 0:
        st.warning("Please enter at least one URL.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # -------------------------
    # STEP 1 : Crawl
    # -------------------------
    status.info("🌐 Crawling webpages...")

    documents = crawl(url_list)

    if not documents:
        st.error("No webpages could be crawled.")
        st.stop()

    progress.progress(25)

    # -------------------------
    # STEP 2 : NLP
    # -------------------------
    status.info("🧠 Extracting Entities & Relationships...")

    nlp_results = run_nlp_pipeline(documents)

    if not nlp_results:
        st.error("No entities or relationships extracted.")
        st.stop()

    progress.progress(50)

    # -------------------------
    # STEP 3 : Neo4j
    # -------------------------
    status.info("🗄 Building Knowledge Graph...")

    run_graph_pipeline(nlp_results)

    progress.progress(75)

    # -------------------------
    # STEP 4 : Visualization
    # -------------------------
    status.info("📊 Generating Graph Visualization...")

    generate_visualization()

    progress.progress(100)

    status.success("✅ Knowledge Graph Generated Successfully!")

    # Save state
    st.session_state.pipeline_done = True

# --------------------------------------------------
# Display Graph
# --------------------------------------------------

if st.session_state.pipeline_done:

    st.divider()

    st.subheader("📊 Interactive Knowledge Graph")

    try:

        with open("graph.html", "r", encoding="utf-8") as f:
            graph_html = f.read()

        components.html(
            graph_html,
            height=700,
            scrolling=True
        )

        st.download_button(
            label="📥 Download Graph",
            data=graph_html,
            file_name="graph.html",
            mime="text/html"
        )

    except FileNotFoundError:

        st.warning("graph.html not found.")

# --------------------------------------------------
# Graphitti AI Assistant
# --------------------------------------------------

if st.session_state.pipeline_done:

    st.divider()

    st.subheader("🤖 Graphitti AI Assistant")

    st.caption("Ask questions about the generated Knowledge Graph.")

    # Display previous chat messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input(
        "Ask anything about the Knowledge Graph..."
    )

    if question:

        # -----------------------------
        # Show User Message
        # -----------------------------
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # -----------------------------
        # Get Answer
        # -----------------------------
        with st.chat_message("assistant"):

            with st.spinner("Searching Knowledge Graph..."):

                try:

                    answer = answer_question(question)

                except Exception as e:

                    answer = f"Error: {e}"

                st.markdown(answer)

        # Save Assistant Response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )