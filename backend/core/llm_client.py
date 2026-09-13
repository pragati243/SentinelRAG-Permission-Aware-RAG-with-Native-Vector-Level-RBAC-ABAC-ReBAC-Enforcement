import os
from typing import List, Dict, Any

from backend.config import settings
from backend.core.groq_client import call_groq_text

class LLMClient:
    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        # Which path actually produced the last answer -- read by callers
        # (e.g. rag_engine.py's tracing span) instead of re-deriving the same
        # provider-selection branching a second time.
        self.last_model_used = "none"

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], user_role: str, user_dept: str) -> str:
        """
        Generates an answer strictly grounded in context_chunks.
        If context_chunks is empty, returns refusing prompt.
        Provider precedence: OpenAI (if configured) -> Groq (if configured) ->
        local template synthesizer as the last resort so the app always answers.
        """
        if not context_chunks:
            self.last_model_used = "none"
            return "I don't have access to information that answers this."

        context_str = "\n\n".join([f"[{c['doc_id']} - {c['section']}]: {c['text']}" for c in context_chunks])
        system_prompt = (
            f"You are SentinelRAG, an enterprise AI assistant.\n"
            f"User Role: {user_role}, Dept: {user_dept}\n"
            f"Answer the query using ONLY the following permitted context chunks.\n"
            f"Do not mention missing or withheld information."
        )
        user_prompt = f"Context:\n{context_str}\n\nQuery: {query}\nAnswer:"

        # Attempt OpenAI API if available
        if self.openai_api_key and self.provider in ["auto", "openai"]:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key)
                res = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                self.last_model_used = "gpt-3.5-turbo"
                return res.choices[0].message.content.strip()
            except Exception:
                pass

        # Attempt Groq if available (this is what actually generates answers
        # in this project's demo deployment, since it ships without an OpenAI key)
        if settings.GROQ_API_KEY and self.provider in ["auto", "groq"]:
            text = call_groq_text(settings.GROQ_GENERATION_MODEL, system_prompt, user_prompt)
            if text:
                self.last_model_used = settings.GROQ_GENERATION_MODEL
                return text.strip()

        # Fallback local grounded response synthesizer
        self.last_model_used = "local-grounded-synthesizer"
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
