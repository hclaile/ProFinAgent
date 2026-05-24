context="""You are an expert judge for AI agents. Task: Evaluate if the agent correctly solved the financial query.

"""

Judge_notool = """

{context}

User Request: {query} is the financial query.

Agent Execution: {agent_execution} is the agent's execution output.

The true answer is {true_answer}.

The agent's output determines whether the question has been answered correctly. The entity names only need to indicate that they are the same, not necessarily identical. The price can fluctuate by about 1%. The final report then analyzes whether the answer meets the requirements.

Mark the result as PASS or FAIL. Do not add any other text.

The output format is as follows in JSON format (must be an exact match, do not add any parameters):

{{
    "result": "PASS"
}} or 

{{
    "result": "FAIL"
}}

"""