import requests
from bs4 import BeautifulSoup
import re

# -----------------------------
# Function to clean webpage text
# -----------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# Crawl a single URL
# -----------------------------
def crawl_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"❌ Could not access {url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        text = clean_text(text)

        return {
            "url": url,
            "text": text
        }

    except Exception as e:
        print(f"Error while crawling {url}")
        print(e)
        return None


# -----------------------------
# Crawl multiple URLs
# -----------------------------
def crawl(urls):

    documents = []

    print("\nStarting Crawling...\n")

    for url in urls:

        print(f"Crawling: {url}")

        document = crawl_url(url)

        if document:
            documents.append(document)
            print("✔ Success\n")

        else:
            print("✖ Failed\n")

    print("Crawling Completed!")
    print(f"Total Documents: {len(documents)}\n")

    return documents


# -----------------------------
# Testing
# -----------------------------
if __name__ == "__main__":

    n = int(input("How many URLs do you want to crawl? "))

    urls = []

    for i in range(n):
        url = input(f"Enter URL {i+1}: ")
        urls.append(url)

    docs = crawl(urls)

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Document {i}")
        print("URL:", doc["url"])
        print()
        print(doc["text"][:1000])   # Print first 1000 characters
        print()