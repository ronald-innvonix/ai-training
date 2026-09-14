# Week 5 — LangChain: Tools, Tool Calling & ReAct Agents

This folder contains the `tool_agent.py` demo for Week 5.

Setup

1. Create a virtual environment in this folder (from the repository root):

```bash
cd week-5
python3 -m venv .venv
source .venv/bin/activate
```

2. Copy the environment sample and set your API keys:

```bash
cp .env.sample .env
# Edit .env and add values for OPENAI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

-------------------OR----------------------

From the repository root you can run the demo script:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
python week-5/tool_agent.py --task all
```

You can run a specific task or change the LLM provider at runtime:

```bash
# run only task1 (search demo)
python week-5/tool_agent.py --task task1

# run all tasks with OpenAI provider
python week-5/tool_agent.py --task all --provider openai
```

Notes

- The project provides `pyproject.toml` for metadata; `requirements.txt` is included for easy pip installs inside a venv.
- If you prefer a dependency-managed install (pip + `pyproject.toml`), use your preferred tool (pip, poetry, etc.).
- `tool_agent.py` is ready to run as-is; copy `.env.sample` to `.env` and populate any needed API keys before running.
