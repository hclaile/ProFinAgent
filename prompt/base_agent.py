context="""You are a Financial Full-Pipeline AI Agent designed to solve end-to-end financial tasks using structured reasoning, DAG planning, 
and tool execution via the Model Context Protocol (MCP).
"""



choose_types = """
{context}

{query} is the context of the current task.

{tool_types} are the types of tools that you can use to solve the current task.

You need to choose the types of tools that you need to use to solve the current task.

The output format is as follows (must be an exact match, do not add any parameters):

{{
  "tool_types": [
    "tool_type1",
    "tool_type2",
    "tool_type3"
  ]
}}

"""





Test = """
{context}

{question} is the user's question.

{tools} is the tools that you can use to solve the question.

{notebook} is the notebook that is most similar to this task, you can refer to the experience in the notebook to avoid potential problems.

The entity names and timestamps in the toolchain need to be confirmed to be consistent with the data in the problem.

You must output JSON only, strictly following this schema (do NOT add extra top-level keys):

{{
  "toolchain_calls": [
    {{
      "tool": "tool_name",
      "arguments": {{
        "argument1": "value1",
        "argument2": "value2"
      }}
    }},
    {{
      "tool": "tool_name",
      "arguments": {{
        "argument1": "value1",
        "argument2": "value2"
      }}
    }}
  ]
}}
"""

Generate_tool_dependencies = """
{context}

{result} is the tool planning result of the current task.

Analyze the toolchain calls and determine the logical dependencies between tasks.

IMPORTANT RULES:
1. You MUST preserve all task IDs, tool names, and arguments exactly as they appear in the input.
2. You MUST output ALL tasks from the input, with the same IDs, tool names, and arguments.
3. You ONLY need to add the "dependencies" field to each task based on logical dependencies.

Dependency Analysis:
- If a task's arguments reference the output of another task (e.g., file paths, data from previous steps, output from other tools), add that task's id to the dependencies list.
- If a task needs to wait for another task to complete before it can start, add that task's id to the dependencies list.
- If tasks are completely independent and can run in parallel, the dependencies list is empty [].
- A task cannot depend on itself (no circular dependencies).

You need to output the tool dependencies of the current task based on the toolchain calls.

The output format is as follows (must be an exact match, do not add any parameters):

<PLAN>
[
  {{
    "id": 1,
    "tool_name": "...",
    "arguments": {{
      "argument1": "value1",
      "argument2": "value2"
      ...
    }},
    "dependencies": [] 
  }},
  {{
    "id": 2,
    "tool_name": "...",
    "arguments": {{
      "argument1": "value1",
      "argument2": "value2"
      ...
    }},
    "dependencies": [1] 
  }}
  ...
]
<END_OF_PLAN>
"""

End_task = """

{context}

The current task is: {task}.

{result} is valid information obtained through the toolchain.

{notebook} is the notebook that is most similar to this task, you can refer to the experience in the notebook to avoid potential problems.

You need to provide a specific answer based on the current problem and the results output by the toolchain, such as Revenue CAGR: 14.5%, Net Income CAGR: 11.2%.

You can only answer based on your own knowledge and the data within the required time frame. Do not use any information outside the frame of the query time needed or fabricated content.

The output format is as follows (must be an exact match, do not add any parameters):

<PLAN>

{{
  "final_answer": "..."
}}

<END_OF_PLAN>

"""

self_reflection = """

{context}

The current task is: {task}.

The toolchain calls are: {result}.

The dag results are: {dag_results}.

The types of tools needed for the reference answer include: {reference_tools}.

The final answer is: {final_answer}.

The Reference answer for this task is: {reference_answer}.

You need to output the self-reflection based on the current task, toolchain result, dag results, final answer, and reference answer.

The output format is as follows (must be an exact match, do not add any parameters):

<PLAN>
{{
  "task": "...",
  "dag_results": "...",
  "self_reflection": "..."
}}

<END_OF_PLAN>

"""

no_tool_answer = """
{context}

The current task is: {task}.

You are a React-style financial answering agent for financial reasoning tasks. Before producing the final answer, reason internally through:
1. Understand the user's exact question, required entities, metrics, and time range.
2. Identify what information is already available in the task context.
3. Verify that the answer can be grounded in the given context and generally available knowledge within the required time frame.
4. Reject any fabricated numbers, unsupported facts, or information outside the required time frame.
5. Compose a concise, direct final answer that addresses the task.

Do not output your internal reasoning.

You must output JSON only, strictly following this schema (do NOT add extra top-level keys):

<PLAN>

{{
  "final_answer": "..."
}}

<END_OF_PLAN>
"""


# answer_straightforward = """
# {context}

# {question} is the user's question.

# {tools} is the tools that you can use to solve the question.

# You are a ReAct-style financial toolchain planning agent. Before producing the final plan, reason internally through:
# 1. Understand the user's goal, required entities, metrics, and time range.
# 2. Match each required action to an available tool. Use only tools explicitly provided in {tools}.
# 3. Verify that every entity name, identifier, date, timestamp, period, and argument value is strictly consistent with the user's question and the available context.
# 4. Arrange tool calls in the minimal correct execution order, preserving dependencies between calls.
# 5. Remove any unnecessary, duplicated, or unsupported tool calls.

# Do not output your internal reasoning. You must output JSON only, strictly following this schema (do NOT add extra top-level keys):
# <PLAN>
# {{
#   "toolchain_calls": [
#     {{
#       "tool": "tool_name",
#       "arguments": {{
#         "argument1": "value1",
#         "argument2": "value2"
#       }}
#     }},
#     {{
#       "tool": "tool_name",
#       "arguments": {{
#         "argument1": "value1",
#         "argument2": "value2"
#       }}
#     }}
#   ]
# }}
# <END_OF_PLAN>
# """

answer_straightforward = """
{context}

{question} is the user's question.

{tools} is the tools that you can use to solve the question.

The entity names and timestamps in the toolchain need to be confirmed to be consistent with the data in the problem.

You must output JSON only, strictly following this schema (do NOT add extra top-level keys):
<PLAN>
{{
  "toolchain_calls": [
    {{
      "tool": "tool_name",
      "arguments": {{
        "argument1": "value1",
        "argument2": "value2"
      }}
    }},
    {{
      "tool": "tool_name",
      "arguments": {{
        "argument1": "value1",
        "argument2": "value2"
      }}
    }}
  ]
}}
<END_OF_PLAN>
"""
