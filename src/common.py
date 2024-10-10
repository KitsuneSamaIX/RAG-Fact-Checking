"""Common utilities.
"""

from dataclasses import dataclass


@dataclass
class Fact:
    speaker: str | None
    text: str
    date: str | None
