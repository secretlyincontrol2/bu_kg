import time
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from config import settings

logger = logging.getLogger("kg_backend.database")

class Neo4jHandler:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, 
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        self.database = settings.neo4j_database

    def close(self):
        self.driver.close()

    def run_query_with_retry(self, query: str, parameters: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Executes a Cypher query with exponential backoff for transient Aura errors."""
        parameters = parameters or {}
        
        for i in range(max_retries):
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(query, parameters)
                    return [dict(record) for record in result]
            except (ServiceUnavailable, SessionExpired) as e:
                logger.warning(f"Neo4j connection error: {e}. Retrying ({i+1}/{max_retries})...")
                time.sleep(2 + i)
                if i == max_retries - 1:
                    logger.error("Max retries reached. Database unavailable.")
                    raise e
            except Exception as e:
                logger.error(f"Unexpected Cypher query error: {e}")
                raise e

db = Neo4jHandler()
