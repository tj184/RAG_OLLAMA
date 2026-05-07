from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model, save_path="db/faiss_index"):
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    vector_store.save_local(save_path)

    return vector_store


def load_vector_store(embedding_model, load_path="db/faiss_index"):
    vector_store = FAISS.load_local(
        load_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store