"""Prompts

The prompts are defined here.
"""

from langchain_core.prompts import ChatPromptTemplate


def get_fact_checking_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _system_msg_1),
        ("human", _human_msg_1)
    ])


_system_msg_1 = """\
You are an helpful assistant for fact checking.
You have to check if the <fact> said by the <speaker> is true based on the provided <context>.

Your response MUST be either:
- "TRUE" if the fact is true;
- "FALSE" if the fact is false.
"""

_human_msg_1 = """\
Check the following:

<speaker>
{speaker}
</speaker>

<fact>
{fact}
</fact>

<context>
{context}
</context>
"""

retry_msg = """\
You must respond in a single word and the response MUST be either:
- "TRUE" if the fact is true;
- "FALSE" if the fact is false.
"""
