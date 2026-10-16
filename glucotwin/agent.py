"""TinyAgent: a hand-written native ReAct loop (THOUGHT -> ACTION -> OBSERVATION -> ... -> ANSWER)."""

import json
import logging
import re
from dataclasses import asdict, dataclass

from glucotwin.llm import LLM, Response
from glucotwin.memory import Memory
from glucotwin.tools import NativeTools

log = logging.getLogger("glucotwin.agent")


@dataclass
class Step:
    """One step of the trajectory: either a tool action + observation, or the final answer."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class TinyAgent:
    def __init__(
        self,
        llm: LLM,
        tools: NativeTools,
        system_prompt: str,
        memory: Memory | None = None,
        max_steps: int = 6,
        max_observation_chars: int = 6000,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory or Memory()
        self.max_steps = max_steps
        self.max_observation_chars = max_observation_chars
        self.trajectory: list[Step] = []
        self.memory.messages.insert(0, {"role": "system", "content": system_prompt})

    def run(self, task: str, images: list[tuple[bytes, str]] | None = None) -> str:
        """Run the autonomy loop until the model answers without calling a tool."""
        self.memory.add("user", task, images=images)

        for step in range(self.max_steps):
            response = self.llm.generate(self.memory.get_messages(), tools=self.tools.schemas)
            self.memory.add("assistant", response.content, tool_calls=response.tool_calls)

            # ANSWER: no tool call means the agent is done
            if not response.tool_calls:
                self.trajectory.append(
                    Step(thought=response.reasoning or "", answer=response.content, metadata=response.metadata)
                )
                return response.content

            # ACTION + OBSERVATION: execute every tool call returned in this turn
            for tool_call in response.tool_calls:
                self._act(tool_call, response)

        # Out of steps: ask for a final answer with tools disabled
        log.warning("Max steps (%s) reached, forcing final answer", self.max_steps)
        self.memory.add("user", "已达到工具调用上限，请基于目前获得的信息直接给出最终回答。")
        response = self.llm.generate(self.memory.get_messages())
        self.memory.add("assistant", response.content)
        self.trajectory.append(Step(answer=response.content, metadata=response.metadata))
        return response.content

    def run_json(self, task: str, images: list[tuple[bytes, str]] | None = None) -> dict:
        """Run the agent and return its final answer parsed as a JSON object."""
        text = self.run(task, images)
        parsed = extract_json(text)
        if parsed is not None:
            return parsed

        # Repair pass: same context, no tools, provider-enforced JSON output
        log.warning("Final answer was not valid JSON, running repair pass")
        self.memory.add("user", "请仅输出符合要求格式的 JSON 对象，不要包含任何其他文字。")
        response = self.llm.generate(self.memory.get_messages(), json_mode=True)
        parsed = extract_json(response.content)
        if parsed is None:
            raise ValueError(f"AI返回格式错误: {response.content[:200]}")
        return parsed

    def _act(self, tool_call: dict, response: Response) -> None:
        try:
            name, kwargs = self.tools.parse(tool_call)
            observation = self.tools.execute(name, kwargs)
        except (KeyError, json.JSONDecodeError) as e:
            name, kwargs = tool_call.get("function", {}).get("name", "?"), {}
            observation = f"Error: could not parse tool call arguments: {e}"

        if len(observation) > self.max_observation_chars:
            observation = observation[: self.max_observation_chars] + "\n...(truncated)"

        log.info("ACTION %s(%s) -> %s chars", name, kwargs, len(observation))
        self.memory.add("tool", observation, tool_call_id=tool_call.get("id"), name=name)
        self.trajectory.append(
            Step(
                thought=response.reasoning or response.content or "",
                action={"tool": name, "kwargs": kwargs},
                observation=observation,
                metadata=response.metadata,
            )
        )

    def trajectory_dict(self, max_chars: int = 500) -> list[dict]:
        """Trajectory for the API response, with long observations shortened."""
        out = []
        for step in self.trajectory:
            d = asdict(step)
            if d["observation"] and len(d["observation"]) > max_chars:
                d["observation"] = d["observation"][:max_chars] + "..."
            out.append(d)
        return out


def extract_json(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating ```json fences and surrounding prose."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
