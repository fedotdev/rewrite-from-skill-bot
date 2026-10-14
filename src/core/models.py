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


class RewriteError(Exception):
    # plain exception on purpose: a frozen+slots dataclass base breaks when the
    # error is raised while another exception is being handled (contextlib/chaining)
    def __init__(self, message: str, user_facing: str) -> None:
        super().__init__(message)
        self.message = message
        self.user_facing = user_facing

    def __str__(self) -> str:
        return self.message
