from crawler import crawl
from nlp import run_nlp_pipeline
from graph_pipeline import run_graph_pipeline
from visualization import generate_visualization
from langgraph_pipeline import graph
from embeddings import EmbeddingGenerator
from chroma import ChromaManager


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def store_documents_in_chroma(documents: list) -> None:
    print("\nStoring document chunks in ChromaDB...")

    embedding_generator = EmbeddingGenerator()
    chroma_manager = ChromaManager(embedding_generator)

    all_chunks = []
    all_metadata = []

    for document in documents:
        chunks = chunk_text(document["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"url": document["url"]})

    if not all_chunks:
        print("No text chunks were produced; nothing stored in ChromaDB.")
        return

    chroma_manager.add_documents(all_chunks, all_metadata)
    print(f"Stored {len(all_chunks)} chunk(s) in ChromaDB.")


def get_url_count() -> int:
    while True:
        try:
            n = int(input("\nHow many URLs do you want to crawl? "))
            if n > 0:
                return n
            print("Please enter a number greater than 0.")
        except ValueError:
            print("Please enter a valid number.")


def get_urls(n: int) -> list:
    urls = []

    for i in range(n):
        while True:
            url = input(f"Enter URL {i + 1}: ").strip()
            if url.startswith("http://") or url.startswith("https://"):
                urls.append(url)
                break
            print("Invalid URL. Please include http:// or https://")

    return urls


def run_qa_loop():
    print("\n" + "=" * 60)
    print("Knowledge Graph Q&A")
    print("=" * 60)
    print("Ask questions about the generated knowledge graph.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = graph.invoke({"question": question})
            answer = result.get("answer", "No answer generated.")
        except Exception as e:
            answer = f"An error occurred while generating the answer: {e}"

        print(f"\nAnswer: {answer}\n")


def main():
    print("=" * 60)
    print("GRAPHITTI - GRAPH NATIVE WEB INTELLIGENCE")
    print("=" * 60)

    n = get_url_count()
    urls = get_urls(n)

    print("\n" + "=" * 60)
    print("Starting Graphitti Pipeline")
    print("=" * 60)

    documents = crawl(urls)

    if not documents:
        print("\nNo webpages could be crawled.")
        return

    nlp_results = run_nlp_pipeline(documents)

    if not nlp_results:
        print("\nNo entities or triplets were extracted.")
        return

    store_documents_in_chroma(documents)

    run_graph_pipeline(nlp_results)

    print("\nGenerating graph visualization...")
    generate_visualization()

    print("\n" + "=" * 60)
    print("Knowledge Graph Generated Successfully!")
    print("Neo4j database updated successfully.")
    print("ChromaDB updated successfully.")
    print("Visualization saved as 'graph.html'")
    print("=" * 60)

    run_qa_loop()


if __name__ == "__main__":
    main()