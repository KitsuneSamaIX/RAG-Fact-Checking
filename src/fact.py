from dataclasses import dataclass


@dataclass
class Fact:
    speaker: str
    text: str
    date: str
