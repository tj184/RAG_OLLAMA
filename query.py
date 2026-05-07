from ingestion.embeddings import get_embedding_model
from ingestion.vector_store import load_vector_store
from retrieval.retriever import get_retriever
from generation.rag_chain import ask_question


def main():
    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Loading vector store...")
    vector_store = load_vector_store(embedding_model)

    retriever = get_retriever(vector_store)

    while True:
        question = input("\nAsk Question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer = ask_question(retriever, question)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()