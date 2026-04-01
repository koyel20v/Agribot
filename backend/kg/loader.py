import pandas as pd
import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "1234llmk")

# Load the same model used in app.py — embeddings must be the same dim (384)
print("Loading SentenceTransformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded")


class KGLoader:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    # ── Entity creation ────────────────────────────────────────────────────────

    def create_entity(self, tx, entity: str, entity_type: str):
        """MERGE the entity node — creates it if it doesn't exist."""
        query = f"""
        MERGE (e:{entity_type} {{name: $entity}})
        """
        tx.run(query, entity=entity)

    def set_embedding(self, tx, entity: str, entity_type: str, embedding: list):
        """
        Store the embedding vector directly on the node as a property.
        This is what app.py queries at startup:
            MATCH (n) WHERE n.embedding IS NOT NULL ...
        """
        query = f"""
        MATCH (e:{entity_type} {{name: $entity}})
        SET e.embedding = $embedding
        """
        tx.run(query, entity=entity, embedding=embedding)

    def create_relationship(self, tx, head_entity, head_type,
                            relation, tail_entity, tail_type):
        query = f"""
        MATCH (h:{head_type} {{name: $head_entity}})
        MATCH (t:{tail_type} {{name: $tail_entity}})
        MERGE (h)-[:{relation}]->(t)
        """
        tx.run(query, head_entity=head_entity, tail_entity=tail_entity)

    # ── Main loader ────────────────────────────────────────────────────────────

    def load_data(self, file_path: str):
        data = pd.read_csv(file_path)
        total = len(data)
        print(f"Loading {total} rows from {file_path}...")

        # Track which entities have been embedded this run to avoid
        # re-embedding the same name if it appears in multiple rows
        embedded: set[str] = set()

        with self.driver.session() as session:
            for idx, (_, row) in enumerate(data.iterrows(), start=1):
                try:
                    head = str(row["head_entity"]).strip()
                    head_type = str(row["head_entity_type"]).strip()
                    tail = str(row["tail_entity"]).strip()
                    tail_type = str(row["tail_entity_type"]).strip()
                    relation = str(row["relation"]).strip()

                    # ── Create nodes ──────────────────────────────────────────
                    session.execute_write(self.create_entity, head, head_type)
                    session.execute_write(self.create_entity, tail, tail_type)

                    # ── Embed and store on node (only once per unique name) ───
                    if head not in embedded:
                        vec = embedder.encode(head).tolist()  # list[float] for Neo4j
                        session.execute_write(self.set_embedding, head, head_type, vec)
                        embedded.add(head)

                    if tail not in embedded:
                        vec = embedder.encode(tail).tolist()
                        session.execute_write(self.set_embedding, tail, tail_type, vec)
                        embedded.add(tail)

                    # ── Create relationship ───────────────────────────────────
                    session.execute_write(
                        self.create_relationship,
                        head, head_type, relation, tail, tail_type
                    )

                    if idx % 50 == 0 or idx == total:
                        print(f"  [{idx}/{total}] processed — {len(embedded)} nodes embedded so far")

                except Exception as e:
                    print(f"❌ Error on row {idx}: {row.to_dict()}")
                    print(f"   {e}")

        print(f"\n✅ KG loaded — {total} rows, {len(embedded)} unique nodes embedded")


# ── Entry point ────────────────────────────────────────────────────────────────

def load_kg_data():
    loader = KGLoader()

    # Path resolution — works whether you run from project root or backend/
    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, "kg", "triples", "extracted_kg_triples.csv")

    if not os.path.exists(file_path):
        # Fallback: try one level up (original loader's path logic)
        BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(BASE_DIR, "backend", "kg", "triples", "extracted_kg_triples.csv")

    if not os.path.exists(file_path):
        print(f"❌ CSV not found. Tried:\n   {file_path}")
        print("   Pass the path manually: loader.load_data('path/to/file.csv')")
        loader.close()
        return

    print(f"📂 Using file: {file_path}")
    loader.load_data(file_path)
    loader.close()


if __name__ == "__main__":
    load_kg_data()