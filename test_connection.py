from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

try:
    driver.verify_connectivity()
    print("✅ Neo4j Connected Successfully!")
except Exception as e:
    print("❌ Connection Failed")
    print(e)

driver.close()