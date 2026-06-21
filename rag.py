import os
import fitz  # PyMuPDF
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from config import PDF_FILENAME, EMBEDDING_MODEL_NAME, VECTOR_SIZE, COLLECTION_NAME

def create_mock_pdf(filename):
    """Generates the actual physical PDF if it's missing."""
    doc = fitz.open()
    page = doc.new_page()

    kb_data = [
        "Application Deadline: The final deadline to apply for the Agentic AI Bootcamp is May 30, 2026.",
        "Required Documents: Applicants must submit a CV, academic transcript, personal statement, valid email, and phone number.",
        "General Questions: General questions about deadlines, required documents, program format, and application steps can be answered automatically.",
        "Transfer Credit: Transfer credit requests require manual review by the admission office.",
        "Non-standard Transcript: Applicants with non-standard grading systems may need equivalency verification, so their case should be reviewed by staff.",
        "Visa Sponsorship: Visa sponsorship cases must be escalated to senior admission officers.",
        "Rejected Application Appeal: Rejected applicants who want to appeal the decision must be escalated to senior admission officers.",
        "Critical Cases: Complaints, legal issues, scholarship exceptions, and urgent cases should be treated as critical and escalated."
    ]

    y_offset = 50
    for item in kb_data:
        page.insert_text((50, y_offset), item, fontsize=11)
        y_offset += 40
    doc.save(filename)
    doc.close()

# Prepare PDF File
if not os.path.exists(PDF_FILENAME):
    print(f"Creating mock PDF document: {PDF_FILENAME}...")
    create_mock_pdf(PDF_FILENAME)

print(f"Parsing local file: {PDF_FILENAME}")
doc = fitz.open(PDF_FILENAME)
raw_text = "".join([page.get_text() + "\n" for page in doc])
doc.close()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)

# Embeddings Model Loading
print(f"Loading local BGE embedding model ({EMBEDDING_MODEL_NAME})...")
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Qdrant client setup (in-memory)
print("Initializing Qdrant Vector database (in-memory)...")
qdrant_client = QdrantClient(":memory:")

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    quantization_config=models.TurboQuantization(
        turbo=models.TurboQuantQuantizationConfig(
            bits=models.TurboQuantBitSize.BITS4,
            always_ram=True,
        )
    )
)

print("Upserting vectorized chunks...")
for idx, chunk_text in enumerate(chunks):
    if not chunk_text.strip():
        continue
    vector = embeddings_model.embed_query(chunk_text)
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[models.PointStruct(id=idx, vector=vector, payload={"content": chunk_text})]
    )

print("[SUCCESS] Vector storage system generated.")
