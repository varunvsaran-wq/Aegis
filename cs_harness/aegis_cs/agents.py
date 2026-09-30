"""Guarded customer-service agents for tau2-bench.

``GuardedAgent`` is tau2's own ``LLMAgent`` (same prompt, same model) plus one
thing: before any database-changing tool call leaves the agent, a checker
reviews it. If the checker finds a policy violation, the call is *not*
returned to the environment. Instead the agent sees a tool result saying
exactly what is wrong and gets to try again (up to ``max_repairs`` times).
After that it must answer the customer in words, with no tool call.

Two checkers, same loop, so they can be compared directly:

- ``checker="code"``: :mod:`aegis_cs.airline_policy`, rules written as code
  that read the live database (read-only).
- ``checker="llm"``: the same model is shown the policy, the recent dialogue,
  the proposed call and the same database records, and asked whether the call
  complies. This is the "noisy judge" design that sank the original Aegis.

Every extra model call (rejected attempts, LLM-checker calls) is added to the
cost of the message that is finally returned, so tau2's cost accounting stays
complete. Guard events are attached to that message under
``raw_data["aegis_guard"]`` for later analysis.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.utils.llm_utils import generate

from aegis_cs.airline_policy import WRITE_TOOLS, CheckContext, check_tool_calls

FEEDBACK_HEADER = "POLICY CHECK FAILED. This action was NOT executed."
FEEDBACK_FOOTER = (
    "Fix the problem and try again, or, if the customer's request is against the "
    "policy, explain that to the customer and deny it."
)

LLM_CHECK_PROMPT = """You are a strict compliance checker for an airline customer-service agent.

<policy>
{policy}
</policy>

<recent_conversation>
{dialogue}
</recent_conversation>

<database_records>
{records}
</database_records>

The agent now wants to make this tool call, which changes the database:
{call}

Does this call comply with every rule in the policy, given the conversation and the
database records? Consider eligibility rules, required confirmations, arithmetic, and
payment rules.

Reply with only a JSON object: {{"compliant": true or false, "violations": ["specific problem", ...]}}"""


def _tool_db(tools: list) -> Any:
    """The airline database the tools are bound to (read-only use)."""
    for tool in tools:
        for attr in ("_func", "func", "function"):
            fn = getattr(tool, attr, None)
            owner = getattr(fn, "__self__", None)
            if owner is not None and hasattr(owner, "db"):
                return owner.db
    raise RuntimeError("Could not find the database behind the tau2 tools.")


def _context(db, messages: list) -> CheckContext:
    ctx = CheckContext(db=db, user_texts=[])
    for m in messages:
        if isinstance(m, UserMessage) and m.content:
            ctx.user_texts.append(m.content)
        elif isinstance(m, AssistantMessage) and m.tool_calls:
            for c in m.tool_calls:
                args = c.arguments or {}
                if c.name == "get_user_details" and args.get("user_id"):
                    ctx.looked_up_users.add(args["user_id"])
                elif c.name == "get_reservation_details" and args.get("reservation_id"):
                    ctx.looked_up_reservations.add(args["reservation_id"])
    return ctx


def _feedback(violations: list[str]) -> str:
    return "\n".join([FEEDBACK_HEADER, *[f"- {v}" for v in violations], FEEDBACK_FOOTER])


class GuardedAgent(LLMAgent):
    """tau2 ``LLMAgent`` with a pre-execution policy gate and repair loop."""

    def __init__(self, tools, domain_policy, llm, llm_args=None,
                 checker: str = "code", max_repairs: int = 2):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        if checker not in ("code", "llm"):
            raise ValueError(f"checker must be 'code' or 'llm', got {checker!r}")
        self.checker = checker
        self.max_repairs = max_repairs
        self._db = _tool_db(tools)

    # -------------------------------------------------------------- checkers

    def _check(self, calls: list, messages: list) -> tuple[dict[str, list[str]], float]:
        """Return ({call_id: violations}, extra_cost)."""
        if self.checker == "code":
            return check_tool_calls(calls, _context(self._db, messages)), 0.0
        return self._check_with_llm(calls, messages)

    def _records_for(self, call) -> dict:
        args = call.arguments or {}
        out: dict[str, Any] = {}
        rid = args.get("reservation_id")
        res = self._db.reservations.get(rid) if rid else None
        if res is not None:
            out["reservation"] = res.model_dump()
            out["flight_statuses"] = {
                f"{f.flight_number} {f.date}": (
                    self._db.flights[f.flight_number].dates[f.date].status
                    if f.flight_number in self._db.flights
                    and f.date in self._db.flights[f.flight_number].dates else "unknown")
                for f in res.flights
            }
        uid = args.get("user_id") or (res.user_id if res is not None else None)
        user = self._db.users.get(uid) if uid else None
        if user is not None:
            out["user"] = user.model_dump()
        for f in args.get("flights") or []:
            fl = self._db.flights.get(f.get("flight_number"))
            if fl is not None and f.get("date") in fl.dates:
                out.setdefault("proposed_flights", {})[f"{fl.flight_number} {f['date']}"] = {
                    "origin": fl.origin, "destination": fl.destination,
                    "status": fl.dates[f["date"]].status}
        return out

    def _check_with_llm(self, calls, messages) -> tuple[dict[str, list[str]], float]:
        writes = [c for c in calls if c.name in WRITE_TOOLS]
        if not writes:
            return {}, 0.0
        dialogue = []
        for m in messages[-14:]:
            if isinstance(m, UserMessage) and m.content:
                dialogue.append(f"Customer: {m.content}")
            elif isinstance(m, AssistantMessage):
                if m.content:
                    dialogue.append(f"Agent: {m.content}")
                for c in m.tool_calls or []:
                    dialogue.append(f"Agent tool call: {c.name}({json.dumps(c.arguments)})")
            elif isinstance(m, ToolMessage) and m.content:
                dialogue.append(f"Tool result: {m.content[:600]}")
        problems: dict[str, list[str]] = {}
        cost = 0.0
        for call in writes:
            prompt = LLM_CHECK_PROMPT.format(
                policy=self.domain_policy,
                dialogue="\n".join(dialogue),
                records=json.dumps(self._records_for(call), default=str)[:6000],
                call=f"{call.name}({json.dumps(call.arguments)})",
            )
            resp = generate(model=self.llm, messages=[UserMessage(role="user", content=prompt)],
                            call_name="aegis_llm_checker", **(self.llm_args or {}))
            cost += resp.cost or 0.0
            verdict = _parse_verdict(resp.content or "")
            if not verdict["compliant"]:
                problems[call.id] = verdict["violations"] or ["The checker judged this call non-compliant."]
        return problems, cost

    # ---------------------------------------------------------------- loop

    def _generate_next_message(self, message, state) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio.")
        if hasattr(message, "tool_messages"):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        extra_cost = 0.0
        events: list[dict] = []
        for attempt in range(self.max_repairs + 1):
            msg = generate(model=self.llm, tools=self.tools,
                           messages=state.system_messages + state.messages,
                           call_name="agent_response", **(self.llm_args or {}))
            calls = msg.tool_calls or []
            write_idx = [i for i, c in enumerate(calls) if c.name in WRITE_TOOLS]
            if not write_idx:
                break
            if len(write_idx) > 1:
                # Several changes in one turn: let only the first go ahead. Later
                # ones may depend on it (upgrade, then cancel), and must be judged
                # against the database after it has run, so the agent re-issues them.
                held = [calls[i].name for i in write_idx[1:]]
                calls = [c for i, c in enumerate(calls)
                         if c.name not in WRITE_TOOLS or i == write_idx[0]]
                msg.tool_calls = calls
                events.append({"attempt": attempt, "outcome": "held_later_writes", "held": held})
            problems, check_cost = self._check(calls, state.messages)
            extra_cost += check_cost
            if not problems:
                if attempt:
                    events.append({"attempt": attempt, "outcome": "passed_after_repair",
                                   "tool": calls[0].name})
                break
            events.append({"attempt": attempt, "outcome": "blocked",
                           "calls": [{"name": c.name, "arguments": c.arguments} for c in calls],
                           "violations": problems})
            # The rejected call stays in the agent's private history with a tool
            # result explaining the violation; the environment never sees it.
            extra_cost += msg.cost or 0.0
            state.messages.append(msg)
            for c in calls:
                state.messages.append(ToolMessage(
                    id=c.id, role="tool", requestor="assistant", error=True,
                    content=_feedback(problems.get(c.id) or
                                      ["Not executed because another call in this turn was blocked."]),
                ))
            if attempt == self.max_repairs:
                # Out of repairs: answer the customer in words, no tool call.
                msg = generate(model=self.llm, tools=self.tools, tool_choice="none",
                               messages=state.system_messages + state.messages,
                               call_name="agent_response", **(self.llm_args or {}))
                events.append({"attempt": attempt, "outcome": "gave_up_to_text"})
                break

        msg.cost = (msg.cost or 0.0) + extra_cost
        if events:
            msg.raw_data = {**(msg.raw_data or {}), "aegis_guard": events}
        return msg


def _parse_verdict(text: str) -> dict:
    """Parse the LLM checker's JSON verdict; unparseable means compliant (fail open)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    compliant = data.get("compliant", True)
    if isinstance(compliant, str):
        compliant = compliant.strip().lower() != "false"
    violations = [str(v) for v in data.get("violations") or [] if str(v).strip()]
    return {"compliant": bool(compliant), "violations": violations}


def register() -> None:
    """Register the guarded agents with tau2's registry (idempotent)."""
    from tau2.registry import registry

    def factory(checker):
        def create(tools, domain_policy, **kwargs):
            return GuardedAgent(tools=tools, domain_policy=domain_policy,
                                llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"),
                                checker=checker)
        return create

    for name, checker in (("guarded_code_agent", "code"), ("guarded_llm_agent", "llm")):
        try:
            registry.register_agent_factory(factory(checker), name)
        except (ValueError, KeyError):
            pass  # already registered
