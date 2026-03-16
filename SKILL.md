# RDC Supervisor - Multi-Agent Orchestrator

## Overview

A chat-based data coach built with **LangGraph supervisor pattern** and a **skills architecture**. Users ask natural language questions about financial data, and the orchestrator routes queries to specialized agents that generate and execute SQL/GraphQL queries against the appropriate data sources.

## Architecture

```
User Query
    |
    v
+-----------+     +----------------+
| FastAPI   |---->| LangGraph      |
| /chat/*   |     | Supervisor     |
+-----------+     +-------+--------+
                          |
              +-----------+-----------+
              |           |           |
        +-----+--+  +----+---+  +----+----+
        |Metadata|  | Mesh   |  |GraphQL  |
        | Agent  |  | Agent  |  | Agent   |
        +--------+  +--------+  +---------+
              |           |           |
         (schemas)    (Snowflake)  (GraphQL API)
                          |           |
                    +-----+-----------+-----+
                    |  Human Feedback Agent  |
                    | (interrupt/resume)     |
                    +-----------------------+
```

## Graph Topology

```
START --> supervisor
supervisor --[conditional]--> metadata_agent | mesh_agent | graphql_agent | human_feedback | synthesizer
metadata_agent --> supervisor
mesh_agent --[has_data?]--> synthesizer (yes) | supervisor (no, try fallback)
graphql_agent --[has_data?]--> synthesizer (yes) | supervisor (no)
human_feedback --> supervisor (after interrupt resumes)
synthesizer --> END
```

## Data Domain Routing

| Data Domain | Primary Agent | Fallback | Source |
|---|---|---|---|
| Instrument data | Mesh (Snowflake) | GraphQL | Both |
| Issuer data | Mesh (Snowflake) | None | Snowflake |
| Trade data | Mesh (Snowflake) | None | Snowflake |
| Position data | Mesh (Snowflake) | None | Snowflake |
| Morningstar data | GraphQL | None | GraphQL |
| Pricing data | GraphQL | None | GraphQL |

Routing is metadata-driven: the metadata agent identifies relevant tables and their source systems, then the supervisor routes to the correct data agent based on business rules.

## Skills Architecture

Skills are modular, prompt-driven specializations loaded on-demand by agents.

### Available Skills

| Skill | Domain | Description |
|---|---|---|
| `trade_query` | mesh | Trade execution SQL patterns |
| `position_query` | mesh | Position/portfolio SQL patterns |
| `instrument_mesh` | mesh | Instrument data from Snowflake (primary, fallback to graphql) |
| `issuer_query` | mesh | Issuer reference data SQL |
| `instrument_graphql` | graphql | Instrument data from GraphQL (fallback) |
| `morningstar_query` | graphql | Morningstar ratings/analytics |
| `pricing_query` | graphql | Pricing/market data |
| `classification` | classification | Industry/instrument classification |

### Adding a New Skill

1. Add skill config to `app/application.yml` under `skills:`:
   ```yaml
   my_new_skill:
     description: "What this skill does"
     domain: "mesh"  # or "graphql" or "classification"
     keywords: ["keyword1", "keyword2"]
     fallback_skill: null
     prompt_template: |
       Your specialized prompt with {tables} and {query} placeholders.
     examples:
       - query: "Example user query"
         output: "Expected SQL/GraphQL output"
   ```
2. No code changes needed - the skill registry auto-loads from config.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/send` | Send a user message (may return response or interrupt) |
| `POST` | `/chat/resume` | Resume after human-in-the-loop interrupt |
| `GET` | `/chat/history/{thread_id}` | Get conversation history |
| `GET` | `/health` | Health check with skill count |

### Send Message

```bash
curl -X POST http://localhost:8000/chat/send \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me all trades from last week", "thread_id": "session-1"}'
```

Response (success):
```json
{
  "thread_id": "session-1",
  "response": "Here are the trades from last week...",
  "is_complete": true,
  "agent_trace": ["mesh_agent"]
}
```

Response (interrupt):
```json
{
  "thread_id": "session-1",
  "interrupt": {
    "feedback_type": "classification",
    "question": "Which industry sector?",
    "options": ["Technology", "Healthcare", "Finance", "Energy", "Other"],
    "context": "The query requires classification details."
  },
  "is_complete": false
}
```

### Resume After Interrupt

```bash
curl -X POST http://localhost:8000/chat/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-1", "response": "Technology"}'
```

## Human-in-the-Loop

Two interrupt scenarios:
1. **Classification**: query mentions industry/sector/instrument type without specifics
2. **No data**: all applicable agents returned empty results

Uses LangGraph's `interrupt()` to pause the graph. Resume via `/chat/resume` with `Command(resume=value)`.

## Prompt Architecture

Each component has a single-responsibility prompt:

| Component | Prompt Responsibility |
|---|---|
| Supervisor | Routing rules + business domain mapping |
| Metadata Agent | Schema/field lookup and source identification |
| Mesh Agent | Snowflake SQL generation |
| GraphQL Agent | GraphQL query generation |
| Human Feedback | Interrupt question templates |
| Synthesizer | Output formatting |

All prompts are in `app/application.yml` - edit there to change behavior without code changes.

## State Schema

```python
SupervisorState:
  messages          # Conversation history (append-only)
  current_agent     # Routing target set by supervisor
  route_reasoning   # Why supervisor chose this route
  active_skill      # Skill loaded for current query
  metadata_context  # Tables, fields, source_system, data_availability
  agent_results     # Accumulated results from all agents (append-only)
  pending_feedback  # Interrupt payload for human
  human_response    # Resume value from interrupt
  iteration_count   # Loop guard (max 5)
  is_complete       # Termination signal
```

## Running

```bash
# Install
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your-key

# Run
uvicorn app.main:app --reload --port 8000

# Test
pytest tests/ -v
```

## Project Structure

```
app/
  main.py              # FastAPI app factory
  config.py            # YAML config loader
  application.yml      # All prompts, skills, model config
  api/
    router.py          # HTTP endpoints
    schemas.py         # Request/response models
  graph/
    state.py           # SupervisorState TypedDict
    supervisor.py      # Graph nodes, edges, compile
    routing.py         # Conditional edge functions
  agents/
    metadata_agent.py  # Schema/metadata lookup
    mesh_agent.py      # Snowflake SQL agent
    graphql_agent.py   # GraphQL agent
    human_feedback_agent.py  # Interrupt handler
    synthesizer.py     # Response formatter
  tools/
    metadata_tools.py  # Mock metadata tools
    mesh_tools.py      # Mock Snowflake execution
    graphql_tools.py   # Mock GraphQL execution
  skills/
    base.py            # Skill dataclass
    registry.py        # Skill discovery and loading
  prompts/
    loader.py          # Typed prompt access
tests/
  test_config.py       # Config loading tests
  test_state.py        # State schema tests
  test_routing.py      # Routing logic tests
  test_skills.py       # Skills framework tests
  test_tools.py        # Mock tool tests
```
