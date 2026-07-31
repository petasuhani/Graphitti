import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver
from neo4j.graph import Node, Relationship
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GraphRetriever:

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self.database = database

        try:
            self.driver: Driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                liveness_check_timeout=30,
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", uri)
        except Exception as error:
            raise RuntimeError(f"Failed to connect to Neo4j: {error}") from error

        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = ChatOllama(model="llama3", temperature=0)
            except Exception as error:
                raise RuntimeError(f"Failed to initialize default LLM: {error}") from error

        self.schema_text = self._fetch_schema()
        logger.info("Fetched graph schema:\n%s", self.schema_text)

        self.cypher_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You translate natural-language questions into Cypher "
                    "queries for a Neo4j knowledge graph built from Wikipedia "
                    "entities and relationships. Nodes generally represent "
                    "entities and have a 'name' property. Relationships "
                    "represent facts connecting entities.\n\n"
                    "Here is the ACTUAL schema of this database. You MUST "
                    "only use node labels and relationship types that "
                    "appear below — never invent new ones, even if they "
                    "seem like a natural fit for the question:\n\n"
                    f"{self.schema_text}\n\n"
                    "Rules:\n"
                    "- Return ONLY the Cypher query, no explanation, no "
                    "markdown code fences.\n"
                    "- Only use node labels and relationship types listed "
                    "in the schema above.\n"
                    "- Entity names in the database may be long or include "
                    "extra words (e.g. 'the Indian Premier League' instead "
                    "of just 'IPL'). To match names flexibly, do NOT use "
                    "operators like =~ or CONTAINS inside the curly-brace "
                    "{{}} property map — that is invalid Cypher syntax. "
                    "Instead, match the node with no property filter and "
                    "use a separate WHERE clause with CONTAINS, for "
                    "example:\n"
                    "  MATCH (n) WHERE toLower(n.name) CONTAINS "
                    "toLower(\"IPL\") "
                    "OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 25\n"
                    "- If the question doesn't clearly map to a specific "
                    "relationship type in the schema, return any/all "
                    "relationships of the matched node(s), rather than "
                    "guessing a specific relationship type that may not "
                    "exist.\n"
                    "- Always include a LIMIT clause (25 unless the question "
                    "implies otherwise).\n"
                    "- Use MATCH patterns that return whole nodes and "
                    "relationships (e.g. 'RETURN n, r, m') so both node and "
                    "relationship data are available.",
                ),
                ("human", "{question}"),
            ]
        )

    def _fetch_schema(self) -> str:
        try:
            with self.driver.session(database=self.database) as session:
                labels_result = session.run("CALL db.labels() YIELD label RETURN label")
                labels = sorted(record["label"] for record in labels_result)

                rel_types_result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType "
                    "RETURN relationshipType"
                )
                relationship_types = sorted(
                    record["relationshipType"] for record in rel_types_result
                )

            if not labels and not relationship_types:
                return (
                    "No labels or relationship types found. The database "
                    "may be empty."
                )

            lines = []
            if labels:
                lines.append("Node labels: " + ", ".join(labels))
            else:
                lines.append("Node labels: (none found)")

            if relationship_types:
                lines.append("Relationship types: " + ", ".join(relationship_types))
            else:
                lines.append("Relationship types: (none found)")

            return "\n".join(lines)

        except Exception as error:
            logger.warning("Failed to fetch graph schema: %s", error)
            return (
                "Schema could not be retrieved. Use generic patterns like "
                "MATCH (n {name: \"X\"})-[r]-(m) RETURN n, r, m rather than "
                "guessing specific relationship types."
            )

    def close(self) -> None:
        try:
            self.driver.close()
            logger.info("Neo4j connection closed.")
        except Exception as error:
            raise RuntimeError(f"Failed to close Neo4j driver: {error}") from error

    def _generate_cypher(self, question: str) -> str:
        try:
            chain = self.cypher_prompt | self.llm
            response = chain.invoke({"question": question})
            cypher_query = response.content.strip()

            if cypher_query.startswith("```"):
                cypher_query = cypher_query.strip("`")
                cypher_query = cypher_query.replace("cypher\n", "", 1).strip()

            logger.info("Generated Cypher query: %s", cypher_query)
            return cypher_query
        except Exception as error:
            raise RuntimeError(f"Failed to generate Cypher query: {error}") from error

    @staticmethod
    def _run_read_query(tx, cypher_query: str) -> List[Dict[str, Any]]:
        result = tx.run(cypher_query)
        return [dict(record) for record in result]

    def _execute_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        try:
            with self.driver.session(database=self.database) as session:
                records = session.execute_read(self._run_read_query, cypher_query)
                logger.info("Query returned %d record(s).", len(records))
                return records
        except Exception as error:
            raise RuntimeError(f"Failed to execute Cypher query: {error}") from error

    @staticmethod
    def _serialize_node(node: Node) -> Dict[str, Any]:
        return {
            "id": node.element_id,
            "labels": list(node.labels),
            "properties": dict(node),
        }

    @staticmethod
    def _serialize_relationship(relationship: Relationship) -> Dict[str, Any]:
        return {
            "id": relationship.element_id,
            "type": relationship.type,
            "start_node_id": relationship.start_node.element_id if relationship.start_node else None,
            "end_node_id": relationship.end_node.element_id if relationship.end_node else None,
            "properties": dict(relationship),
        }

    def _extract_nodes_and_relationships(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        nodes_by_id: Dict[str, Dict[str, Any]] = {}
        relationships_by_id: Dict[str, Dict[str, Any]] = {}

        for record in records:
            for value in record.values():
                if isinstance(value, Node):
                    serialized = self._serialize_node(value)
                    nodes_by_id[serialized["id"]] = serialized
                elif isinstance(value, Relationship):
                    serialized = self._serialize_relationship(value)
                    relationships_by_id[serialized["id"]] = serialized
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Node):
                            serialized = self._serialize_node(item)
                            nodes_by_id[serialized["id"]] = serialized
                        elif isinstance(item, Relationship):
                            serialized = self._serialize_relationship(item)
                            relationships_by_id[serialized["id"]] = serialized

        return {
            "nodes": list(nodes_by_id.values()),
            "relationships": list(relationships_by_id.values()),
        }

    def _repair_cypher(self, question: str, failed_query: str, error_message: str) -> str:
        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You write Cypher queries for a Neo4j knowledge graph. "
                    "A previously generated query failed. Fix it based on "
                    "the error message. Return ONLY the corrected Cypher "
                    "query, no explanation, no markdown code fences.\n\n"
                    f"Schema:\n{self.schema_text}\n\n"
                    "Reminder: operators like CONTAINS or =~ can only be "
                    "used in a WHERE clause, never inside a {{}} property "
                    "map.",
                ),
                (
                    "human",
                    "Original question: {question}\n\n"
                    "Failed query:\n{failed_query}\n\n"
                    "Error message:\n{error_message}\n\n"
                    "Corrected query:",
                ),
            ]
        )

        try:
            chain = repair_prompt | self.llm
            response = chain.invoke(
                {
                    "question": question,
                    "failed_query": failed_query,
                    "error_message": error_message,
                }
            )
            corrected_query = response.content.strip()

            if corrected_query.startswith("```"):
                corrected_query = corrected_query.strip("`")
                corrected_query = corrected_query.replace("cypher\n", "", 1).strip()

            logger.info("Repaired Cypher query: %s", corrected_query)
            return corrected_query
        except Exception as error:
            raise RuntimeError(f"Failed to repair Cypher query: {error}") from error

    def retrieve(self, question: str) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        cypher_query = self._generate_cypher(question)

        try:
            records = self._execute_query(cypher_query)
        except RuntimeError as error:
            logger.warning(
                "Cypher query failed, attempting one repair: %s", error
            )
            try:
                cypher_query = self._repair_cypher(question, cypher_query, str(error))
                records = self._execute_query(cypher_query)
            except Exception as repair_error:
                raise RuntimeError(
                    f"Cypher query failed and repair attempt also failed: {repair_error}"
                ) from repair_error

        if not records:
            logger.info("No results found for question: %s", question)
            return {
                "question": question,
                "cypher_query": cypher_query,
                "nodes": [],
                "relationships": [],
                "result_count": 0,
            }

        extracted = self._extract_nodes_and_relationships(records)

        return {
            "question": question,
            "cypher_query": cypher_query,
            "nodes": extracted["nodes"],
            "relationships": extracted["relationships"],
            "result_count": len(records),
        }