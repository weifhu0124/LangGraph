# Projects
## [Adaptive Self RAG for Amazon Agentic AI](https://github.com/weifhu0124/LangGraph/tree/main/agentic_rag)

An adaptive, self RAG with a vector store to answer Amazon Agentic AI offerings (Bedrock agentcore, Amazon Q Business and Amazon Q Developers). 
- It uses an agent to self evaluate whether the answer contains hallucination (based on the retrieved documents) and whether the answer addresses the question.
- It uses a router to direct questions not related to Amazon Agentic AI to perform web search.

| Component | Technology | Description |
|-----------|------------|-------------|
| 🧠 **AI Framework** | LangGraph 🦜🕸️ | Orchestrates the AI pipeline |
| 🤖 **LLM** | OpenAI GPT | Powers the conversation generation |
| 🌐 **Web Search** | Tavily | Enhanced discovery |
| 🔍 **Vector Store** | Chroma | Vector database |
| 📊 **Monitoring** | LangSmith | Optional tracing and debugging |
| 🐍 **Backend** | Python 3.12+ | Core application logic |

<img src="https://github.com/weifhu0124/LangGraph/blob/c5ca1da1ebfe6e3e8f89765df266c52b6ddf8181/agentic_rag/rag.png" width=300 height=500>

Output:
- When asked about "Compare and contrast Amazon Q Business and Amazon Q developer":
<code>{'question': 'Compare and contrast Amazon Q Business and Amazon Q developer', 'generation': 'Amazon Q Business is tailored for businesses to connect company data and systems to solve problems and generate content, while Amazon Q Developer is designed to accelerate software development. Both tools leverage generative AI to assist users in different ways, with Amazon Q Business focusing on workplace productivity and Amazon Q Developer targeting software development needs.', 'web_search': True, 'documents': [...]}</code>

- When asked about "How to make a pizza":
<code>{'question': 'How to make a pizza', 'generation': 'To make a pizza, pour 160 ml of water into a bowl, add yeast, and mix with 250g of white bread flour. Shape and bake the pizza dough at 500°F on a preheated stone until just cooked through. Use a pizza peel or baking sheet to transfer the pizza dough and parchment paper onto your pizza stone, pan, or baking sheet.', 'documents': [...]}</code>

Detailed std output: https://github.com/weifhu0124/LangGraph/blob/c5ca1da1ebfe6e3e8f89765df266c52b6ddf8181/agentic_rag/result.txt

## [LinkedIn Summary Reflection Agent](https://github.com/weifhu0124/LangGraph/tree/main/reflection_agent)

A generation and a reflection agent to iteratively generate and critique LinkedIn About section.

| Component | Technology | Description |
|-----------|------------|-------------|
| 🧠 **AI Framework** | LangGraph 🦜🕸️ | Orchestrates the AI pipeline |
| 🤖 **LLM** | OpenAI GPT | Powers the conversation generation |
| 📊 **Monitoring** | LangSmith | Optional tracing and debugging |
| 🐍 **Backend** | Python 3.12+ | Core application logic |

<img src="https://github.com/weifhu0124/LangGraph/blob/685d46ed33f288db82ee47bbea96d0fcb6e60f3a/reflection_agent/reflection.png" width=300 height=500>

## [Stock Strategies Essay Reflexion Agent](https://github.com/weifhu0124/LangGraph/tree/main/reflexion_agent)

A reflexion agent that uses Tavily Search to write an essay and self-critque on the draft. We use a stock picking strategy as an example and showcased the result after two revise iterations: [📑 Stock Strategy Essay](https://github.com/weifhu0124/LangGraph/blob/8dc95a0ebdb424e354128abe6ca1342c28fe83bf/reflexion_agent/result.txt).

| Component | Technology | Description |
|-----------|------------|-------------|
| 🧠 **AI Framework** | LangGraph 🦜🕸️ | Orchestrates the AI pipeline |
| 🌐 **Web Search** | Tavily | Enhanced discovery |
| 🤖 **LLM** | OpenAI GPT | Powers the conversation generation |
| 📊 **Monitoring** | LangSmith | Optional tracing and debugging |
| 🐍 **Backend** | Python 3.12+ | Core application logic |

<img src="https://github.com/weifhu0124/LangGraph/blob/8dc95a0ebdb424e354128abe6ca1342c28fe83bf/reflexion_agent/reflexion.png" width=300 height=500>

## [Simple ReAct Function Call](https://github.com/weifhu0124/LangGraph/tree/main/react_function_call)

A simple ReAct agent that uses web search and Python function as tools.

| Component | Technology | Description |
|-----------|------------|-------------|
| 🧠 **AI Framework** | LangGraph 🦜🕸️ | Orchestrates the AI pipeline |
| 🌐 **Web Search** | Tavily | Enhanced discovery |
| 🤖 **LLM** | OpenAI GPT | Powers the conversation generation |
| 📊 **Monitoring** | LangSmith | Optional tracing and debugging |
| 🐍 **Backend** | Python 3.12+ | Core application logic |

<img src="https://github.com/weifhu0124/LangGraph/blob/6417ecea3a7a9378045c656272e67ff85be3c41d/react_function_call/flow.png" width=300 height=500>
