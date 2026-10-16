"""Short-term memory: the message list the agent sends on every step."""

import base64


class Memory:
    """Conversation history in OpenAI chat format, with image support."""

    def __init__(self, max_history_turns: int = 10):
        self.messages: list[dict] = []
        self.max_history_turns = max_history_turns

    def add(
        self,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        images: list[tuple[bytes, str]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> None:
        """Append a message.

        Arguments:
            images: list of (raw_bytes, mime_type) attached to a user message.
            tool_call_id/name: set for role="tool" observations.
        """
        if images:
            parts = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"},
                }
                for raw, mime in images
            ]
            content = parts + [{"type": "text", "text": content}]

        message = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if name:
            message["name"] = name
        self.messages.append(message)

    def load_history(self, history: list[dict]) -> None:
        """Restore prior chat turns ({role: user|assistant|model, text}), trimmed to recent turns."""
        for turn in history[-self.max_history_turns * 2 :]:
            role = "assistant" if turn.get("role") in ("assistant", "model") else "user"
            text = turn.get("text") or turn.get("content") or ""
            if text:
                self.add(role, text)

    def get_messages(self) -> list[dict]:
        return self.messages
