from neo4j import GraphDatabase
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ── Connections ───────────────────────────────────────────────────────────────
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Step 1: Extract keyword from question ─────────────────────────────────────
def extract_keyword(question):
    """
    Asks Groq to pull the most important entity/keyword
    from the user's question — this is what we search in the graph.
    Example: "Who founded Apple?" → "Apple"
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the single most important keyword or entity name "
                    "from the user's question. Return ONLY that word or phrase, "
                    "nothing else. No explanation, no punctuation."
                )
            },
            {"role": "user", "content": question}
        ],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()


# ── Step 2: Search graph for that keyword ─────────────────────────────────────
def search_graph(tx, keyword):
    """
    Searches Neo4j for nodes whose name contains the keyword,
    then fetches all their direct connections.
    Returns a list of subject → relation → object facts.
    """
    result = tx.run("""
        MATCH (a)-[r]->(b)
        WHERE toLower(a.name) CONTAINS toLower($keyword)
           OR toLower(b.name) CONTAINS toLower($keyword)
        RETURN a.name AS subject, type(r) AS relation, b.name AS object
        LIMIT 20
    """, keyword=keyword)
    return [record.data() for record in result]


# ── Step 3: Format graph facts into readable context ──────────────────────────
def format_context(facts):
    """
    Turns raw graph records into readable sentences
    that the LLM can understand and reason over.
    Example: "Steve Jobs --[FOUND]--> Apple Inc. in 1976"
    """
    if not facts:
        return "No relevant facts found in the knowledge graph."
    lines = []
    for f in facts:
        lines.append(f"{f['subject']} --[{f['relation']}]--> {f['object']}")
    return "\n".join(lines)


# ── Step 4: Ask LLM to answer using graph context ─────────────────────────────
def ask_llm(question, context):
    """
    Sends the user's question + graph facts to Groq LLM.
    LLM answers ONLY based on the graph facts provided —
    not from its own training data.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph assistant. "
                    "Answer the user's question ONLY using the graph facts provided below. "
                    "If the facts don't contain enough information, say "
                    "'I could not find enough information in the knowledge graph.' "
                    "Keep your answer concise and clear.\n\n"
                    f"Knowledge Graph Facts:\n{context}"
                )
            },
            {"role": "user", "content": question}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


# ── Full Q&A Pipeline ─────────────────────────────────────────────────────────
def answer_question(question):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print('='*50)

    # Step 1: Extract keyword
    keyword = extract_keyword(question)
    print(f"Keyword extracted: {keyword}")

    # Step 2: Search graph
    with neo4j_driver.session() as session:
        facts = session.execute_read(search_graph, keyword)
    print(f"Graph facts found: {len(facts)}")

    # Step 3: Format context
    context = format_context(facts)
    print(f"\nGraph Context:\n{context}")

    # Step 4: Get LLM answer
    answer = ask_llm(question, context)
    print(f"\nAnswer: {answer}")
    print('='*50)
    return answer


# ── Interactive mode ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Knowledge Graph Q&A System")
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        answer_question(question)

    neo4j_driver.close()