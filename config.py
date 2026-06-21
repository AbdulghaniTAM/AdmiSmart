import os

# Check HF_TOKEN
if "HF_TOKEN" not in os.environ:
    print("Warning: HF_TOKEN environment variable not found. Gated models may fail to load.")

# Central Configuration Constants
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8b")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", 1024))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "university_knowledge_turbo")
PDF_FILENAME = os.environ.get("PDF_FILENAME", "knowledge_base.pdf")

# Relational Database Configuration
# Fall back to a default PostgreSQL URL if not set
DB_URL = os.environ.get("DB_URL", "postgresql://postgres:postgres@localhost:5432/admissions")
