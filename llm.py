"""LLM integration: Groq-hosted inference for answer generation."""

from __future__ import annotations

import json
import os

from groq import Groq


class GroqSummarizer:
    """Answers a query from retrieved context using a Groq-hosted LLM."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def summarize(self, context: str, query: str, max_tokens: int = 1024) -> dict:
        prompt = f"""
You are an expert document analyst.

Task:
Analyze the provided context and answer the user query by creating a concise, accurate, and factual summary.

Instructions:
- Use ONLY information present in the context.
- Do not invent facts.
- Focus on information relevant to the query.
- Remove duplication.
- Preserve important numbers, dates, thresholds, and requirements.
- If information is missing, explicitly state: "I don't have information on this"
- Produce a structured summary.

User Query:
{query}

Context:
{context}

Output Format: Produce output as a valid json

{{
    "response": "summary"
}}
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert document analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        try:
            return json.loads(result)
        except Exception:
            return {"response": result}
