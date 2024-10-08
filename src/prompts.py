"""Prompts.

The main prompts are defined here.
"""

from langchain_core.prompts import ChatPromptTemplate
from config import config


def get_fact_checking_prompt_template() -> ChatPromptTemplate:
    match config.CLASSIFICATION_LEVELS:
        case 2:
            return _get_fact_checking_prompt_template_for_2_classification_levels()
        case 6:
            return _get_fact_checking_prompt_template_for_6_classification_levels()
        case _:
            raise ValueError()


def _get_fact_checking_prompt_template_for_2_classification_levels() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _system_msg_1),
        ("human", _human_msg_1)
    ])


def _get_fact_checking_prompt_template_for_6_classification_levels() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _system_msg_2),
        ("human", _human_msg_1)
    ])


_system_msg_1 = """\
You are a fact-checking expert trained to evaluate the truthfulness of statements based on provided evidence.
Your task is to assess whether the <fact>, stated by the <speaker>, is true based on the <context> provided.

You must respond with a single word:
- "TRUE" if the fact is true;
- "FALSE" if the fact is false.
"""


_system_msg_2 = """\
You are a fact-checking expert trained to evaluate the truthfulness of statements based on provided evidence.
Your task is to assess whether the <fact>, stated by the <speaker>, is accurate based on the <context> provided.

The fact-checking will be done using the "TRUTH-O-METER" system from "politifact.com", which rates the truthfulness of statements on a scale of 0 to 5, as follows:
- 5 = True
- 4 = Mostly True
- 3 = Half True
- 2 = Mostly False
- 1 = False
- 0 = Pants on Fire

You must respond with a single number (0, 1, 2, 3, 4, 5) that best represents the accuracy of the fact according to the TRUTH-O-METER scale.
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


def get_retry_msg() -> str:
    match config.CLASSIFICATION_LEVELS:
        case 2:
            return _retry_msg_for_2_classification_levels
        case 6:
            return _retry_msg_for_6_classification_levels
        case _:
            raise ValueError()


_retry_msg_for_2_classification_levels = """\
You must respond in a single word and the response MUST be either:
- "TRUE" if the fact is true;
- "FALSE" if the fact is false.
"""


_retry_msg_for_6_classification_levels = """\
You must respond with a single number (0, 1, 2, 3, 4, 5) that best represents the accuracy of the fact according to the TRUTH-O-METER scale.
"""
