from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


class KnowledgeGraph:

    def __init__(
        self,
        uri=NEO4J_URI,
        user=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    def close(self):
        self.driver.close()

    @staticmethod
    def create_triplet(tx, subject, relation, obj):
        import re

        safe_relation = re.sub(r"[^A-Za-z0-9_]", "_", relation.upper())

        query = f"""
        MERGE (a:Entity {{name:$subject}})
        MERGE (b:Entity {{name:$object}})
        MERGE (a)-[:{safe_relation}]->(b)
        """

        tx.run(
            query,
            subject=subject,
            object=obj
        )

    def build_graph(self, results):
        with self.driver.session() as session:
            for document in results:
                print(f"Processing : {document['url']}")

                for triple in document["triplets"]:
                    subject = triple["subject"]
                    relation = triple["relation"]
                    obj = triple["object"]

                    session.execute_write(
                        self.create_triplet,
                        subject,
                        relation,
                        obj
                    )

        print("\nKnowledge Graph Created Successfully!")


def run_graph_pipeline(results):
    kg = KnowledgeGraph()
    kg.build_graph(results)
    kg.close()

    import webbrowser
    print("Opening Neo4j Browser...")
    webbrowser.open("https://console.neo4j.io/")


if __name__ == "__main__":

    sample = [
        {
            "url": "sample",
            "triplets": [
                {
                    "subject": "Google",
                    "relation": "acquired",
                    "object": "YouTube"
                },
                {
                    "subject": "Sundar Pichai",
                    "relation": "works_at",
                    "object": "Google"
                }
            ]
        }
    ]

    run_graph_pipeline(sample)