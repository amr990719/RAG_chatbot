"""
Ingestion pipeline: reads data/<department>/*, chunks it, embeds it, and
persists it into a local Chroma vector store.

Run this once up front, and again any time files under data/ change.
No Groq API key is required for this step -- embeddings run locally.
"""

from pathlib import Path
import shutil

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "finsolve_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MD_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_markdown_file(path: Path, department: str) -> list[Document]:
    """Split one .md file into self-contained, department-tagged chunks."""
    raw_text = path.read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MD_HEADERS_TO_SPLIT_ON)
    header_sections = header_splitter.split_text(raw_text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks: list[Document] = []
    for section in header_sections:
        breadcrumb = " > ".join(
            value for key, value in section.metadata.items() if key.startswith("Header")
        )
        content = f"{breadcrumb}\n\n{section.page_content}" if breadcrumb else section.page_content

        for piece in char_splitter.split_text(content):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={"department": department, "source": path.name},
                )
            )
    return chunks


def load_hr_csv(path: Path, department: str) -> list[Document]:
    """Turn each employee row into one natural-language Document."""
    df = pd.read_csv(path)

    docs = []
    for _, row in df.iterrows():
        sentence = (
            f"{row['full_name']} (employee ID {row['employee_id']}) works as a "
            f"{row['role']} in the {row['department']} department, based in {row['location']}. "
            f"Email: {row['email']}. Date of birth: {row['date_of_birth']}. "
            f"Joined on {row['date_of_joining']}, reporting to manager ID {row['manager_id']}. "
            f"Annual salary: {row['salary']}. "
            f"Leave balance: {row['leave_balance']} days ({row['leaves_taken']} taken this cycle). "
            f"Attendance: {row['attendance_pct']}%. "
            f"Performance rating: {row['performance_rating']} "
            f"(last reviewed {row['last_review_date']})."
        )
        docs.append(
            Document(
                page_content=sentence,
                metadata={
                    "department": department,
                    "source": path.name,
                    "employee_id": row["employee_id"],
                },
            )
        )
    return docs


def load_all_documents() -> list[Document]:
    all_docs: list[Document] = []

    for department_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        department = department_dir.name
        dept_doc_count = 0

        for file_path in sorted(department_dir.iterdir()):
            if file_path.suffix == ".md":
                docs = chunk_markdown_file(file_path, department)
            elif file_path.suffix == ".csv":
                docs = load_hr_csv(file_path, department)
            else:
                continue

            all_docs.extend(docs)
            dept_doc_count += len(docs)

        print(f"  {department:<12} -> {dept_doc_count} chunks")

    return all_docs


def main():
    print("Loading and chunking documents from data/ ...")
    documents = load_all_documents()
    print(f"Total chunks: {len(documents)}")

    if PERSIST_DIR.exists():
        print(f"Removing existing vector store at {PERSIST_DIR} ...")
        shutil.rmtree(PERSIST_DIR)

    print(f"Embedding with '{EMBEDDING_MODEL}' (first run downloads the model, ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Persisting to {PERSIST_DIR} ...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        # Cosine similarity on normalized embeddings gives clean, comparable
        # 0-1 relevance scores -- used later by the out-of-scope guardrail.
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("Done.")


if __name__ == "__main__":
    main()
