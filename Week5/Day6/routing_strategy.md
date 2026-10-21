# Routing Strategy

## Local Model (Ollama - Qwen2.5:3B)

Use for:

- Greetings
- FAQs
- Document summarization
- Code explanation
- Local PDF Q&A
- ChromaDB Retrieval

Advantages:

- Zero API cost
- Private
- Fast

---

## Cloud API

Use for:

- Very complex reasoning
- Large context (>30K tokens)
- Vision
- Multi-step planning
- High accuracy tasks

---

Decision Rules

Simple Question
        ↓
Local LLM

Medium Question
        ↓
Local LLM + ChromaDB

Complex Question
        ↓
Cloud API

Large Documents
        ↓
Cloud API