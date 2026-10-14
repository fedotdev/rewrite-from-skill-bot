from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class RewriteRequest:
    source_text: str
    user_instruction: Optional[str]
    chat_id: int
    message_id: int
    message_thread_id: Optional[int]
    user_id: int
    update_id: int


@dataclass(frozen=True, slots=True)
class RewriteResult:
    text: str
    warning: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RewriteError(Exception):
    message: str
    user_facing: str

    def __str__(self) -> str:
        return self.message