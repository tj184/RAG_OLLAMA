import ollama


SYSTEM_PROMPT = """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}
"""


def format_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def ask_question(retriever, question, model="gemma4:e2b"):
    retrieved_docs = retriever.invoke(question)

    context = format_context(retrieved_docs)

    prompt = SYSTEM_PROMPT.format(context=context)

    final_prompt = f"""
{prompt}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ]
    )

    return response["message"]["content"]