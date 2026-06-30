from neo4j import GraphDatabase
from pyvis.network import Network
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))
def fetch_graph(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN
        n.name AS source,
        labels(n)[0] AS source_label,
        type(r) AS relationship,
        m.name AS target,
        labels(m)[0] AS target_label
    """

    result = tx.run(query)
    return [record.data() for record in result]


# -----------------------------
# Node Colors
# -----------------------------
def get_color(label):
    colors = {
        "PERSON": "#FF6B6B",
        "ORG": "#4D96FF",
        "MISC": "#6BCB77",
        "LOC": "#FFD93D",
        "Entity": "#9D4EDD",
        "CONCEPT": "#00B894"
    }

    return colors.get(label, "#A0AEC0")


# -----------------------------
# Node Shapes
# -----------------------------
def get_shape(label):
    shapes = {
        "PERSON": "diamond",
        "ORG": "box",
        "MISC": "ellipse",
        "LOC": "triangle",
        "Entity": "dot",
        "CONCEPT": "star"
    }

    return shapes.get(label, "dot")


# -----------------------------
# Build Visualization
# -----------------------------
def build_visualization(records):

    net = Network(
        height="850px",
        width="100%",
        bgcolor="#0A1016",
        font_color="white",  # Changed to white
        directed=True,
        notebook=False
    )

    added_nodes = set()

    for record in records:

        source = record["source"]
        target = record["target"]

        if source not in added_nodes:
            net.add_node(
                source,
                label=source,
                title=f"{source}<br>Type : {record['source_label']}",
                color=get_color(record["source_label"]),
                shape=get_shape(record["source_label"]),
                size=40,  # Increased node size
                borderWidth=2
            )
            added_nodes.add(source)

        if target not in added_nodes:
            net.add_node(
                target,
                label=target,
                title=f"{target}<br>Type : {record['target_label']}",
                color=get_color(record["target_label"]),
                shape=get_shape(record["target_label"]),
                size=40,  # Increased node size
                borderWidth=2
            )
            added_nodes.add(target)

        net.add_edge(
         source,
         target,
         label=record["relationship"],
         title=record["relationship"],
        font={
        "size": 16,
        "color": "white",
        "strokeWidth": 2
    },
        width=3,
        color="#34BA42",
        arrows="to",
        smooth=False
)
        

    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 18,
          "face": "Arial",
          "color": "white"
        }
      },

      "edges": {
        "font": {
          "size": 12,
          "align": "middle",
          "color": "white"
        },

        "smooth": {
          "enabled": true,
          "type": "dynamic"
        }
      },

      "physics": {
        "enabled": true,

        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.25,
          "springLength": 350,
          "springConstant": 0.04,
          "damping": 0.09
        },

        "minVelocity": 0.75
      },

      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)

    net.show("graph.html", notebook=False)

    print("\\nGraph Generated Successfully!")
    print("Saved as graph.html")

def generate_visualization():

    with driver.session() as session:
        records = session.execute_read(fetch_graph)

    driver.close()

    if not records:
        print("No graph data found in Neo4j.")
    else:
        build_visualization(records)