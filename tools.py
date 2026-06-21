from langchain_core.tools import tool
from qdrant_client import models
from rag import qdrant_client, embeddings_model
from config import COLLECTION_NAME
from database import get_connection
from psycopg2.extras import DictCursor
import psycopg2

@tool
def search_university_knowledge_base(query: str) -> str:
    """Searches university guidelines and deadlines using the 4-bit vector array."""
    query_vector = embeddings_model.embed_query(query)
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2
    ).points

    if search_results:
        combined_context = "\n".join([f"- {res.payload['content']}" for res in search_results])
        return f"Retrieved Context Chunks:\n{combined_context}"
    return "No matching policy or guideline documents discovered."

@tool
def check_admission_status(applicant_id: str) -> str:
    """Queries the live PostgreSQL database to fetch an applicant's registration profile and approval status."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("SELECT * FROM applicants WHERE applicant_id = %s", (applicant_id,))
                record = cursor.fetchone()
        
        if record:
            return f"ID: {record['applicant_id']} | Name: {record['name']} | Status: {record['status']} | GPA: {record['gpa']}"
        return f"No applicant record found for ID: {applicant_id}"
    except Exception as e:
        return f"Error connecting to database or querying record: {str(e)}"

@tool
def update_admission_status(applicant_id: str, new_status: str) -> str:
    """Modifies the live relational PostgreSQL database to change an applicant's status."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if student exists before performing update
                cursor.execute("SELECT name FROM applicants WHERE applicant_id = %s", (applicant_id,))
                if not cursor.fetchone():
                    return f"Error: Cannot update. Applicant {applicant_id} does not exist."

                # Perform the safe, row-level concurrency update
                cursor.execute(
                    "UPDATE applicants SET status = %s WHERE applicant_id = %s", 
                    (new_status, applicant_id)
                )
                conn.commit()
        return f"Database successfully updated. Applicant {applicant_id} status changed to '{new_status}'."
    except Exception as e:
        return f"Error connecting to database or updating record: {str(e)}"

