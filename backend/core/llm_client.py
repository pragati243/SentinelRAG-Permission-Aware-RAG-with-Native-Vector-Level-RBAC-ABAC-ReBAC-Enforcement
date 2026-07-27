import os
from typing import List, Dict, Any

class LLMClient:
    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], user_role: str, user_dept: str) -> str:
        """
        Generates an answer strictly grounded in context_chunks.
        If context_chunks is empty, returns refusing prompt.
        """
        if not context_chunks:
            return "I don't have access to information that answers this."

        context_str = "\n\n".join([f"[{c['doc_id']} - {c['section']}]: {c['text']}" for c in context_chunks])

        # Attempt OpenAI API if available
        if self.openai_api_key and self.provider in ["auto", "openai"]:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key)
                prompt = (
                    f"You are SentinelRAG, an enterprise AI assistant.\n"
                    f"User Role: {user_role}, Dept: {user_dept}\n"
                    f"Answer the query using ONLY the following permitted context chunks.\n"
                    f"Do not mention missing or withheld information.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Query: {query}\nAnswer:"
                )
                res = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return res.choices[0].message.content.strip()
            except Exception:
                pass

        # Fallback local grounded response synthesizer
        return self._local_grounded_synthesis(query, context_chunks, user_role)

    def _local_grounded_synthesis(self, query: str, chunks: List[Dict[str, Any]], user_role: str) -> str:
        """
        Smart local synthesizer that builds a concise, truthful answer directly from permitted text snippets.
        """
        snippet_summary = []
        for c in chunks[:3]:
            text = c["text"]
            # Clean up linebreaks
            clean_text = " ".join(text.split())
            snippet_summary.append(clean_text)

        combined_info = " ".join(snippet_summary)
        return f"Based on permitted internal documentation ({user_role} view): {combined_info}"

llm_client = LLMClient()
