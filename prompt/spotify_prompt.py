select_tools = """
You are an expert RESTBench planner for Spotify Web API tool routing.
Your task is to select the minimal and correct sequence of Spotify API tools needed to answer or execute the user's request.

User's Question:
{question}

Available Spotify tools:
{tools}

Relevant prior notebook examples, if any:
{notebook}

Planning rules:
1. Use only tools that appear in the available Spotify tool card.
2. Tool names must match the tool card exactly, including HTTP method and path, for example "GET /search" or "POST /users/{{user_id}}/playlists".
3. Select the smallest toolchain that can satisfy the request. Do not add exploratory tools unless they are required to obtain IDs or user context.
4. If a request can be answered with a single endpoint, output one tool only.
5. Do not invent arguments. This stage selects tool names only.

Return JSON only. Do not include markdown, comments, or extra text.

Required JSON schema:
<PLAN>
{{
  "toolchain_calls": [
    {{
      "tool": "tool_name_1"
    }},
    {{
      "tool": "tool_name_2"
    }}
  ]
}}
<END_OF_PLAN>
"""


Generate_tool_dependencies = """
You are an expert Spotify Web API dependency planner.
Given a selected Spotify toolchain, add logical execution dependencies between the selected tools.

Selected toolchain:
{result}

Dependency rules:
1. Preserve every task ID and tool name exactly as provided.
2. Output every task from the input.
3. Add only the "dependencies" field.
4. A task depends on an earlier task when it needs an ID, URI, user context, playlist ID, track URI, artist ID, album ID, device state, or other output from that earlier task.
5. Do not create circular dependencies.

Return JSON only. The top-level value must be a JSON object.

Required JSON schema:
{{
  "tasks": [
    {{
      "id": 1,
      "tool": "exact_tool_name",
      "dependencies": []
    }},
    {{
      "id": 2,
      "tool": "exact_tool_name",
      "dependencies": [1]
    }}
  ]
}}
"""


self_reflection = """
You are an expert evaluator for Spotify Web API tool-routing tasks.
Review whether the selected Spotify toolchain is appropriate for the user request and reference solution.

Current task:
{task}

Selected toolchain calls:
{result}

DAG/dependency results:
{dag_results}

Reference answer or reference solution:
{reference_answer}

Evaluate:
1. Whether the selected tools are valid Spotify Web API tools.
2. Whether the tool order is sufficient to satisfy required ID, URI, playlist, user, or playback dependencies.
3. Whether unnecessary tools were selected.
4. Whether any required tool is missing compared with the reference solution.
5. What should be improved in future Spotify tool routing for this task type.

Return JSON only. Do not include markdown, comments, or extra text.

Required JSON schema:
{{
  "task": "...",
  "dag_results": "...",
  "self_reflection": "..."
}}
"""


zero_spotify = """
You are a zero-shot RESTBench planner for Spotify Web API tasks.
Your goal is to output the minimal and correct Spotify API toolchain for the user request.

User's Question:
{question}

Available Spotify tool card:
{tools}

Hard constraints:
1. Use only tools from the Spotify tool card.
2. Tool names must match exactly, including method and path, for example "GET /search", "GET /tracks/{{id}}", or "POST /playlists/{{playlist_id}}/tracks".
3. Output tool names only. Do not output arguments.
4. Use the fewest tools that can complete the task.
5. Add prerequisite tools only when needed to obtain Spotify IDs, URIs, playlist IDs, current user ID, or playback context.

Return JSON only. Do not include markdown, comments, or extra text.

Output format (must be exact):
<PLAN>
{{
  "toolchain_calls": [
    {{
      "tool": "tool_name_1"
    }},
    {{
      "tool": "tool_name_2"
    }}
  ]
}}
<END_OF_PLAN>
"""
