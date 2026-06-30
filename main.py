from crawler import crawl
from nlp_pipeline import run_nlp_pipeline
from graph_pipeline import run_graph_pipeline
from visualization import generate_visualization
from qa_graph import answer_question


def main():

    print("=" * 60)
    print("GRAPHITTI - GRAPH NATIVE WEB INTELLIGENCE")
    print("=" * 60)

    # Ask user for number of URLs
    while True:
        try:
            n = int(input("\nHow many URLs do you want to crawl? "))

            if n > 0:
                break

            print("Please enter a number greater than 0.")

        except ValueError:
            print("Please enter a valid number.")

    # Read URLs
    urls = []

    for i in range(n):

        while True:

            url = input(f"Enter URL {i + 1}: ").strip()

            if url.startswith("http://") or url.startswith("https://"):
                urls.append(url)
                break

            print("Invalid URL. Please include http:// or https://")

    print("\n" + "=" * 60)
    print("Starting Graphitti Pipeline")
    print("=" * 60)

    # STEP 1 - Crawl
    documents = crawl(urls)

    if not documents:
        print("\nNo webpages could be crawled.")
        return

    # STEP 2 - NLP
    nlp_results = run_nlp_pipeline(documents)

    if not nlp_results:
        print("\nNo entities or triplets were extracted.")
        return

    # STEP 3 - Build Neo4j Graph
    run_graph_pipeline(nlp_results)

    # STEP 4 - Generate Visualization
    print("\nGenerating graph visualization...")
    generate_visualization()

    print("\n" + "=" * 60)
    print("Knowledge Graph Generated Successfully!")
    print("Neo4j database updated successfully.")
    print("Visualization saved as 'graph.html'")
    print("=" * 60)

    # STEP 5 - Knowledge Graph Q&A
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

    answer = answer_question(question)

    print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()