from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from ingestion.embeddings import get_embedding_model
from ingestion.vector_store import create_vector_store


PDF_PATH = "data/data.pdf"


def main():
    print("Loading PDF...")
    documents = load_pdf(PDF_PATH)

    print("Splitting documents...")
    chunks = split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS vector store...")
    create_vector_store(chunks, embedding_model)

    print("Vector DB saved successfully!")


if __name__ == "__main__":
    main()