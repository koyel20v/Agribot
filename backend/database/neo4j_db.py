from neo4j import GraphDatabase

class Neo4jDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_session(self):
        return self.driver.session()

    def close(self):
        self.driver.close()


# Create instance
neo4j_db = Neo4jDB("bolt://localhost:7687", "neo4j", "1234llmk")