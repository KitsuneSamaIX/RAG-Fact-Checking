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
        ("human", _human_msg_2)
    ])


_system_msg_1 = """\
You are a fact-checking expert trained to evaluate the truthfulness of statements based on provided evidence.
Your task is to assess whether a statement, stated by a speaker, is true based on the context provided.
The context provided is composed of pieces of documents that might be relevant to verify whether or not the statement is true.

You must respond with a single word:
- "TRUE" if the statement is true
- "FALSE" if the statement is false
"""


_system_msg_2 = """\
You are a fact-checking expert trained to evaluate the truthfulness of statements based on provided evidence.
Your task is to assess whether a statement, stated by a speaker, is accurate based on the context provided.
The context provided is composed of pieces of documents that might be relevant to verify whether or not the statement is true.

You have to rate the statement using the following scale which rates the truthfulness of statements with a value from 0 to 5 as follows:
- 5 = True (the statement is accurate and there’s nothing significant missing)
- 4 = Mostly True (the statement is accurate but needs clarification or additional information)
- 3 = Half True (the statement is partially accurate but leaves out important details or takes things out of context)
- 2 = Mostly False (the statement contains an element of truth but ignores critical facts that would give a different impression)
- 1 = False (the statement is not accurate)
- 0 = Pants on Fire (the statement is not accurate and makes a ridiculous claim)

You must respond with a single value (0, 1, 2, 3, 4, 5) that best represents the accuracy of the statement according to the scale.
"""


_human_msg_1 = """\
Check the truthfulness of the following statement:
{speaker} said {fact}

Context: 
{context}
"""


_human_msg_2 = """\
Rate the truthfulness of the following statement:
{speaker} said {fact}

Context: 
{context}
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
You must respond with a single word and the response must be either:
- "TRUE" if the statement is true
- "FALSE" if the statement is false
"""


_retry_msg_for_6_classification_levels = """\
You must respond with a single value (0, 1, 2, 3, 4, 5) that best represents the accuracy of the statement according to the scale.
"""
