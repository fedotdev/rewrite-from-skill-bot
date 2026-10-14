from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewriteRequest:
    source_text: str
    user_instruction: str | None
    chat_id: int
    message_id: int
    message_thread_id: int | None
    user_id: int


@dataclass(frozen=True, slots=True)
class RewriteResult:
    text: str
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class RewriteError(Exception):
    message: str
    user_facing: str

    def __str__(self) -> str:
        return self.message
