# rdc-supervisor
multi agent orchestrator which manages metadata-agent(fetches metadata for mesh and graphql agents for context), mesh-agent (snowflake sql agent), graphql-agent (creates and runs graphql queries)

## Reference template (LangGraph 1.2 / LangChain 1.5)

See [`reference/supervisor_template/`](reference/supervisor_template/README.md): a tested
reference implementation of this supervisor on the latest LangChain/LangGraph APIs, to copy
for similar projects. [`WHATS_NEW.md`](reference/supervisor_template/WHATS_NEW.md) maps the
new APIs onto this codebase.
