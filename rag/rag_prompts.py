class RAGPromptBuilder:
    """
    Builds prompts specifically for RAG querying
    """

    def __init__(self):
        pass  # self.prompt = []

    def build(self, query, results):
        """
        Builds the prompts for the LLM
        """
        context_parts = []
        for result in results:
            document = result["document"]
            score = result["score"]

            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page")

            context_parts.append(f"""
SOURCE: {source}
PAGE: {page}
SIMILARITY: {score:.3f}

CONTENT:
{document.text}
""")

        context = "\n---\n".join(context_parts)
        prompt = f"""
SYSTEM:
You are a knowledge-based assistant.

Answer the user's question using the provided
context.

If the context does not contain enough information
to answer the question, say that you do not have
enough information.

Do not invent information.

CONTEXT:
{context}

USER QUESTION:
{query}
"""


#             context_parts.append({"text": document.text, "source": source})

#         prompt = f"""
# SYSTEM PROMPT: Answer using the provided context:

# context: {context}
# USER QUESTION: {query}
# """
        return prompt
