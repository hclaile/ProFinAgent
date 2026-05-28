Official implementation for "Towards Professional-Grade Financial Agents: Benchmarking, Tooling, and Structured Reasoning (ICML'2026)"

# ProFinAgent

ProFinAgent is a financial-agent benchmark and tool-augmented reasoning framework. It supports two agent backends, multiple prompt/tool testing modes, and an inline MCP-style tool server for executing financial data, analysis, training, forecasting, search, and reporting tools.

ProFinR huggingface page: https://huggingface.co/datasets/huangchenglaile/ProFinR

This README focuses on how to run the project and how to choose the correct agent and test script.

## Project Layout

```text
ProFinAgent/
├── agent/                  # Agent implementations and agent configuration
├── configs/                # Tool registry, tool categories, and tool-selection rules
├── dataset/                # Benchmark data, including ProFinR.json
├── modules/                # Tool implementations
├── prompt/                 # Prompt frameworks/templates used by agents
├── workspace/              # Local algorithm/workspace dependencies
├── mcp_server_inline.py    # Inline MCP tool registration and dispatch
├── base_agent_*.py         # Local-model benchmark scripts
├── test_agent_*.py         # API-model benchmark scripts
└── requirements.txt        # Python dependencies
```

## Agent Types

ProFinAgent provides two main agent families.

### `base_agent`: local deployed agent

`base_agent` scripts use `agent/base_agent.py`. This path is intended for a locally deployed model service, for example an SGLang/OpenAI-compatible local endpoint.

Use this when:

- you run the language model locally;
- you configure a local `BASE_URL`, `MODEL_NAME`, and `MODEL_PATH`;
- you want to benchmark a self-hosted model.

Typical scripts:

```text
base_agent_no_tool.py
base_agent_tool_straight.py
base_agent_tool.py
```

### `test_agent`: API-based agent

`test_agent` scripts use `agent/test_agent.py`. This path is intended for API-based model calls through the OpenAI-compatible client.

Use this when:

- you call a remote model API;
- you configure an API key, base URL, and model name;
- you want to benchmark hosted or third-party OpenAI-compatible models.

Typical scripts:

```text
test_agent_notool.py
test_agent_tool_straight.py
test_agent_tool.py
```

## Test Script Naming Rules

The suffix of a test file indicates the evaluation mode.

### `no_tool`: no tool usage

Scripts ending with `no_tool` or `notool` run the agent without using any tool. The model answers directly from its own reasoning.

Examples:

```text
base_agent_no_tool.py
test_agent_notool.py
```

Use this mode to measure pure LLM performance without external tools.

### `tool_straight`: generic prompt framework + tools

Scripts ending with `tool_straight` use a generic prompt framework with tool calling. Users can replace or edit the prompt framework in the `prompt/` directory.

Examples:

```text
base_agent_tool_straight.py
test_agent_tool_straight.py
```

Use this mode when you want to test a custom or generic tool-use prompt. The relevant prompt templates are under:

```text
prompt/base_agent.py
prompt/test_agent.py
```

You can modify these files to replace the prompt framework used for tool planning and answer generation.

### `tool`: ProFinAgent framework + tools

Scripts ending directly with `tool` use the ProFinAgent tool framework. These tests use the project's full tool-planning and execution flow, including the configured tool registry and ProFinAgent-specific tool orchestration.

Examples:

```text
base_agent_tool.py
test_agent_tool.py
```

Use this mode when you want to evaluate the complete ProFinAgent framework rather than only a generic prompt/tool baseline.

## Configuration

Edit the following file before running benchmarks:

```text
agent/agent_config.py
```

Important fields for local `base_agent` runs:

```python
BASE_URL = ""
MODEL_NAME = ""
MODEL_PATH = ""
API_KEY = "EMPTY"
TEMPERATURE = 0.0
MAX_TOKENS = 6144
TIMEOUT = 600
AUTO_LOAD = True
```

Important fields for API-based `test_agent` runs:

```python
test_agent_apikey = ""
test_agent_base_url = ""
test_agent_model = ""
```

Embedding/RAG-related fields:

```python
EMBEDDING_MODEL_PATH = ""
DAG_MEMORY_JSON_PATH = ""
```

Tool configuration files:

```text
configs/tools.json       # Full tool definitions
configs/tool_type.json   # Tool categories
configs/must_have_rules.json
```

The inline tool dispatcher is:

```text
mcp_server_inline.py
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
cd ./ProFinAgent
pip install -r requirements.txt
```

Some tools depend on external services, API keys, model files, local workspaces, or system libraries. Configure those dependencies before running tool-based benchmarks.

## Dataset

The included benchmark dataset is:

```text
dataset/ProFinR.json
```

Most benchmark scripts accept `--benchmark_path`. If a script defaults to `dataset/benchmark.json`, pass the ProFinR path explicitly:

```bash
--benchmark_path ./ProFinAgent/dataset/ProFinR.json
```

## Running Benchmarks

Run all commands from the project root:

```bash
cd ./ProFinAgent
```

### Local deployed agent, no tools

```bash
python base_agent_no_tool.py \
  --benchmark_path ./ProFinAgent/dataset/ProFinR.json
```

### API agent, no tools

```bash
python test_agent_notool.py \
  --benchmark_path ./ProFinAgent/dataset/ProFinR.json
```

### Local deployed agent, generic prompt + tools

```bash
python base_agent_tool_straight.py \
  --benchmark_path ./ProFinAgent/dataset/ProFinR.json
```

### API agent, generic prompt + tools

```bash
python test_agent_tool_straight.py \
  --benchmark_path ./ProFinAgent/dataset/ProFinR.json
```

### Local deployed agent, ProFinAgent framework + tools

```bash
python base_agent_tool.py \
  --benchmark_path ./ProFinAgent/dataset/ProFinR.json \
  --tools_path ./ProFinAgent/configs/tools.json
```

### API agent, ProFinAgent framework + tools

```bash
python test_agent_tool.py \
  --benchmark_path ./ProFinAgent/dataset/ProFinR.json \
  --tools_path ./ProFinAgent/configs/tools.json
```

## Prompt Customization

Prompt templates are stored in:

```text
prompt/
```

Use these files to customize how the agent plans tools, formats tool calls, and generates final answers:

```text
prompt/base_agent.py
prompt/test_agent.py
prompt/judge_agent.py
prompt/spotify_prompt.py
prompt/tmdb_prompt.py
```

For `tool_straight` tests, replace or edit the generic prompt framework in the relevant prompt file.

For direct `tool` tests, modify the ProFinAgent prompt/framework logic used by the corresponding agent implementation and prompt template.

## Tool Execution

Tool-based scripts call tools through:

```text
mcp_server_inline.py
```

Tool metadata is configured in:

```text
configs/tools.json
configs/tool_type.json
```

Tool implementations live in:

```text
modules/
```

Examples of available tool groups include:

- market data tools;
- financial statement and valuation tools;
- macroeconomic data tools;
- media/news/search tools;
- regulatory filing tools;
- technical indicator and backtesting tools;
- model training tools;
- report generation tools.

## Results and Resume

Benchmark scripts write results under `Result/` subdirectories. Most scripts support resumable execution through:

```bash
--results_path <path>
--resume_mode skip_success_and_error
--start 1
--limit 10
```

Useful examples:

```bash
python test_agent_tool.py \
  --benchmark_path dataset/ProFinR.json \
  --tools_path configs/tools.json \
  --limit 20
```

```bash
python base_agent_no_tool.py \
  --benchmark_path dataset/ProFinR.json \
  --results_path Result/local_no_tool.json \
  --resume_mode retry_errors
```

## Recommended Workflow

1. Configure `agent/agent_config.py`.
2. Choose the agent backend:
   - `base_agent` for a local deployed model;
   - `test_agent` for an API-based model.
3. Choose the evaluation mode:
   - `no_tool` for direct answering;
   - `tool_straight` for generic prompt + tool usage;
   - `tool` for the full ProFinAgent framework.
4. Pass `dataset/ProFinR.json` as `--benchmark_path`.
5. For tool runs, verify `configs/tools.json`, `configs/tool_type.json`, and `mcp_server_inline.py`.
6. Run a small subset first with `--limit`.
7. Inspect outputs under `Result/`.

## Quick Decision Table

| Goal | Script pattern |
| --- | --- |
| Local model, no tools | `base_agent_no_tool.py` |
| API model, no tools | `test_agent_notool.py` |
| Local model, generic prompt + tools | `base_agent_tool_straight.py` |
| API model, generic prompt + tools | `test_agent_tool_straight.py` |
| Local model, ProFinAgent framework + tools | `base_agent_tool.py` |
| API model, ProFinAgent framework + tools | `test_agent_tool.py` |

## Notes

- `base_agent` means the locally deployed agent path.
- `test_agent` means the API-call agent path.
- `no_tool` means no tools are used.
- `tool_straight` means generic prompt framework with tools; prompts can be replaced in `prompt/`.
- `tool` means the ProFinAgent framework with tools.
- Always run commands from `./ProFinAgent` unless you use absolute paths.
