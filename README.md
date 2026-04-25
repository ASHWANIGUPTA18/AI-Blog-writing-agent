# AI Blog Writing Agent

An end-to-end AI blog writing system built with **LangGraph** and **OpenAI**. Given a topic, it automatically decides whether to research the web, creates a structured plan, writes all sections in parallel, optionally generates images, and delivers a complete Markdown blog post.

---

## Table of Contents

1. [What is LangGraph?](#what-is-langgraph)
2. [Core LangGraph Concepts Used](#core-langgraph-concepts-used)
3. [Project Architecture](#project-architecture)
4. [How the Agent Thinks](#how-the-agent-thinks)
5. [File-by-File Walkthrough](#file-by-file-walkthrough)
6. [Setup & Installation](#setup--installation)
7. [Running the Project](#running-the-project)
8. [Environment Variables](#environment-variables)
9. [Models Used](#models-used)

---

## What is LangGraph?

LangGraph is a framework for building **stateful, multi-step AI agents** as directed graphs.

Think of it like this:

```
Traditional LLM call:   Input → LLM → Output  (one shot, no memory, no branching)

LangGraph agent:        Input → Node A → Node B → Node C → Output
                                          ↓
                                       Node D (if condition)
                                          ↓
                                       Node C
```

Each **node** is a Python function that reads shared state and writes updates back to it. The **graph** defines which node runs next — either always (a fixed edge) or conditionally (based on what the previous node returned).

### Why LangGraph over plain LLM calls?

| Problem | LangGraph Solution |
|---|---|
| Complex logic needs multiple LLM calls | Model it as a graph of nodes |
| Need branching (if research needed, else skip) | Conditional edges |
| Want to run things in parallel | Fan-out with the `Send` API |
| Need shared memory across steps | Typed `State` dict passed through all nodes |
| Want to reuse a flow inside another flow | Compile a subgraph and use it as a node |

---

## Core LangGraph Concepts Used

### 1. StateGraph and State

The `State` is a Python `TypedDict` — a shared dictionary that every node can read from and write to.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    topic: str
    plan: dict
    sections: list
    final: str

graph = StateGraph(State)
```

Every node receives the full current `State` and returns only the keys it wants to update.

---

### 2. Nodes

A node is just a Python function. It receives `state` and returns a `dict` of updates.

```python
def orchestrator(state: State) -> dict:
    plan = llm.invoke(...)         # call the LLM
    return {"plan": plan}          # only update the "plan" key
```

You register it with:

```python
graph.add_node("orchestrator", orchestrator)
```

---

### 3. Edges (Fixed and Conditional)

**Fixed edge** — always go from A to B:

```python
graph.add_edge("research", "orchestrator")
```

**Conditional edge** — decide at runtime based on state:

```python
def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

graph.add_conditional_edges("router", route_next, {
    "research": "research",
    "orchestrator": "orchestrator"
})
```

---

### 4. Fan-out with Send (Parallel Workers)

The most powerful pattern in this project. Instead of writing sections one by one, the orchestrator creates one `Send` message per section, and LangGraph runs all workers **in parallel**.

```python
from langgraph.types import Send

def fanout(state: State):
    return [
        Send("worker", {"task": task, "topic": state["topic"]})
        for task in state["plan"].tasks     # one Send per section
    ]

graph.add_conditional_edges("orchestrator", fanout, ["worker"])
```

Each `Send("worker", payload)` launches a separate worker node with that payload. All workers run at the same time.

---

### 5. Annotated Reducer (Auto-merge Parallel Results)

When workers run in parallel and all write to the same key, you need to tell LangGraph how to combine their results. Using `Annotated` with a reducer function:

```python
from typing import Annotated
import operator

class State(TypedDict):
    sections: Annotated[list, operator.add]   # automatically concatenates all worker outputs
```

Each worker returns `{"sections": [(task_id, markdown)]}` and LangGraph automatically appends them all together.

---

### 6. Subgraphs

A compiled LangGraph app can be used as a node inside another graph. This project uses a **reducer subgraph** for the image pipeline:

```python
# Build a mini-graph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_images", generate_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_images")
reducer_subgraph = reducer_graph.compile()

# Use it as a single node in the main graph
main_graph.add_node("reducer", reducer_subgraph)
```

---

### 7. compile() and invoke()

After defining all nodes and edges, you compile the graph into a runnable app:

```python
app = graph.compile()

# Run it
output = app.invoke({"topic": "Self-Attention in Transformers", "sections": []})
```

You can also stream progress step by step:

```python
for step in app.stream(inputs, stream_mode="updates"):
    print(step)   # see each node's output as it completes
```

---

## Project Architecture

### Full Graph (Production — `bwa_backend.py`)

```
START
  │
  ▼
┌─────────┐
│  Router │  ← Decides: does this topic need web research?
└─────────┘
  │                  │
  │ needs_research   │ no research needed
  ▼                  │
┌──────────┐         │
│ Research │         │   Uses Tavily to search the web,
│ (Tavily) │         │   filters by recency, deduplicates
└──────────┘         │
  │                  │
  └──────────────────▼
               ┌─────────────┐
               │ Orchestrator│  ← Creates a structured blog plan (5–9 sections)
               └─────────────┘
                      │
                      │ fan-out (one Send per section)
                      ▼
          ┌──────────────────────┐
          │  Worker  │  Worker   │  ← All sections written in PARALLEL
          │  Worker  │  Worker   │
          └──────────────────────┘
                      │
                      │ (all sections merged by Annotated reducer)
                      ▼
               ┌─────────────────────────────────────────┐
               │            Reducer Subgraph             │
               │                                         │
               │  merge_content → decide_images          │
               │                       │                 │
               │                  generate_and           │
               │                  place_images           │
               └─────────────────────────────────────────┘
                      │
                     END  ← Final Markdown saved to .md file
```

### Routing Logic

The **Router** classifies every topic into one of three modes before any writing begins:

| Mode | When | Research? | Recency Window |
|---|---|---|---|
| `closed_book` | Evergreen concepts (Self-Attention, Docker, etc.) | No | — |
| `hybrid` | Mostly evergreen but needs fresh examples/tools | Yes | Last 45 days |
| `open_book` | Weekly news, "latest X", pricing, policy | Yes | Last 7 days |

---

## How the Agent Thinks

For a topic like **"State of Multimodal LLMs in 2026"**:

```
1. Router         → mode=hybrid, needs_research=True
                    queries=["multimodal LLM benchmarks 2026", "GPT-4o vision updates", ...]

2. Research       → Calls Tavily for each query
                    Deduplicates results
                    Filters to last 45 days
                    Returns: 12 EvidenceItems with title, url, snippet

3. Orchestrator   → Reads topic + evidence
                    Plans 7 sections: Intro, Key Models, Benchmarks, Use Cases,
                                      Limitations, Code Example, What's Next
                    Marks 3 sections as requires_citations=True

4. Fanout         → Spawns 7 parallel worker nodes

5. Workers (x7)   → Each writes one section in Markdown
                    Citation-required sections link to Evidence URLs

6. Reducer        → Merges 7 sections in correct order
                    Asks LLM: "should this blog have diagrams?"
                    Generates up to 3 images via Gemini
                    Replaces [[IMAGE_1]] placeholders with actual images
                    Saves final .md file

7. Output         → Complete blog post in Markdown + images/ folder
```

---

## File-by-File Walkthrough

### Learning Path (Notebooks)

The notebooks are designed as a **step-by-step progression**. Read them in order to understand how the system was built.

#### `1_bwa_basic.ipynb` — Foundation
The simplest possible blog agent.

**Graph:** `orchestrator → [workers in parallel] → reducer`

- `orchestrator`: Asks LLM for a blog plan (title + sections with brief descriptions)
- `worker`: Writes one section in Markdown (runs in parallel for all sections)
- `reducer`: Joins sections in order, writes `.md` file

**LangGraph concepts introduced:** StateGraph, nodes, fixed edges, `Send` fan-out, `Annotated` reducer

---

#### `2_bwa_improved_prompting.ipynb` — Better Prompts
Same graph structure as notebook 1, but the **prompts are much richer**.

**What changed:**
- `Task` schema now has `goal`, `bullets`, `target_words`, `section_type`
- Orchestrator prompt enforces developer-grade quality: MWEs, edge cases, trade-offs
- Worker prompt enforces: cover all bullets in order, stay within word count, code fences

**Key lesson:** The graph structure doesn't change — just better schemas and prompts produce dramatically better output.

---

#### `3_bwa_research.ipynb` — Web Research
Adds a **Router** node and a **Research** node before planning.

**Graph:** `router →(conditional)→ [research →] orchestrator → [workers] → reducer`

**What's new:**
- `RouterDecision` schema: LLM classifies the topic as `closed_book`, `hybrid`, or `open_book`
- `TavilySearchResults`: Calls the Tavily search API for each query
- Research synthesizer: Deduplicates URLs, normalizes dates, passes `EvidencePack` to orchestrator
- Workers now receive evidence and cite URLs for claims

**LangGraph concepts introduced:** Conditional edges, dynamic routing

---

#### `4_bwa_research_fine_tuned.ipynb` — Recency Control
Fine-tunes the research pipeline with **date-aware filtering**.

**What's new:**
- `as_of` date and `recency_days` added to `State`
- Router sets recency window: 7 days (open_book), 45 days (hybrid), 10 years (closed_book)
- Research node hard-filters evidence: only items newer than the cutoff survive for `open_book`
- `blog_kind` field forces `news_roundup` mode for open_book — prevents workers from drifting into tutorials
- Worker prompt has a **scope guard**: if `blog_kind == news_roundup`, do not write how-to content

---

#### `5_bwa_image.ipynb` — Image Generation
Adds a **reducer subgraph** for image planning and generation.

**Graph:** same as notebook 4, but the reducer node is itself a 3-step subgraph:

```
merge_content → decide_images → generate_and_place_images
```

**What's new:**
- `merge_content`: Joins all sections into one Markdown document
- `decide_images`: LLM reviews the full blog and decides if/where images would help. Inserts `[[IMAGE_1]]` placeholders and writes `ImageSpec` objects (prompt, filename, alt, caption)
- `generate_and_place_images`: Calls **Gemini** (`gemini-2.5-flash-image`) for each spec, saves `.png` to `images/`, replaces placeholders with `![alt](images/file.png)` in the Markdown. Gracefully falls back to a text block if image generation fails.

**LangGraph concepts introduced:** Subgraphs (a compiled graph used as a node)

---

### Production Files

#### `bwa_backend.py` — Production Backend
Combines everything from notebook 5 into a single importable module.

Key differences from the notebook:
- `load_dotenv()` loads API keys from `.env` automatically
- Tavily is optional — if `TAVILY_API_KEY` is missing, research returns empty evidence and the agent proceeds in `closed_book` mode
- Gemini image generation is optional — if `GOOGLE_API_KEY` is missing, images fall back to a descriptive text block
- Uses a `_safe_slug()` helper for clean filenames

**Exports:** `app` — the compiled LangGraph application

---

#### `bwa_frontend.py` — Streamlit UI
A web interface that wraps `bwa_backend.app`.

**Features:**
- Sidebar: enter topic, pick as-of date, click Generate
- Streams graph progress in real time (shows which node is running)
- **Plan tab**: shows blog structure as a table (sections, word targets, flags)
- **Evidence tab**: shows all web sources used
- **Markdown Preview tab**: renders the final blog with inline images
- **Images tab**: shows generated images, download as ZIP
- **Logs tab**: full event log of every graph step
- **Past blogs**: sidebar lists previously generated `.md` files — click to reload any

---

#### `tavily_test.ipynb` — API Key Smoke Test
A two-cell notebook to verify your Tavily API key works before running the full agent.

---

## Setup & Installation

### 1. Create and activate the virtual environment

```bash
# Create (already done — folder "ai blog writing agent" is in the project root)
python -m venv "ai blog writing agent"

# Activate on Windows (PowerShell)
& ".\ai blog writing agent\Scripts\Activate.ps1"

# Activate on Windows (bash/Git Bash)
source "ai blog writing agent/Scripts/activate"

# Activate on macOS/Linux
source "ai blog writing agent/bin/activate"
```

### 2. Install dependencies

```bash
pip install langgraph langchain-openai langchain-core langchain-community \
            pydantic python-dotenv streamlit pandas google-genai
```

### 3. Create your `.env` file

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
GOOGLE_API_KEY=your_google_key_here
```

`TAVILY_API_KEY` and `GOOGLE_API_KEY` are optional. Without them, the agent skips research and image generation.

---

## Running the Project

### Streamlit App (Recommended)

```bash
# Make sure the venv is activated first
streamlit run bwa_frontend.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Usage:**
1. Type a topic in the sidebar (e.g. `"How does Self-Attention work?"`)
2. Set the as-of date (used for recency filtering in research mode)
3. Click **Generate Blog**
4. Watch real-time progress across the tabs
5. Download the final Markdown or a ZIP bundle with images

### Notebooks

Open Jupyter and run any notebook from top to bottom:

```bash
jupyter notebook
```

Start with `1_bwa_basic.ipynb` if you are new to LangGraph.

---

## Environment Variables

| Variable | Required | Used By | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | All files | Powers gpt-4o-mini for planning and writing |
| `TAVILY_API_KEY` | No | Notebooks 3–5, backend | Web search for research mode |
| `GOOGLE_API_KEY` | No | Notebook 5, backend | Gemini image generation |

---

## Models Used

| Model | Used For | Cost tier |
|---|---|---|
| `gpt-4o-mini` | All LLM calls (routing, planning, writing, image decisions) | Cheapest OpenAI model with structured output |
| `gemini-2.5-flash-image` | Image generation | Google Gemini (requires `GOOGLE_API_KEY`) |

To change the model, edit this line in `bwa_backend.py`:

```python
llm = ChatOpenAI(model="gpt-4o-mini")   # change to any OpenAI model
```

---

## Output

Every successful run saves:

- `<blog_title>.md` — the full blog post in Markdown format
- `images/<filename>.png` — generated diagrams (if image generation is enabled)

The Streamlit app lets you download the Markdown file or a ZIP bundle containing both.
