"""A more human-sounding simulated customer for tau2-bench.

tau2's default customer simulator writes in full, polite, well-organised
sentences, which is easier for an agent than real support chats. This subclass
keeps everything that defines the task (the scenario, the goal, the facts, the
stop signals, the scoring) and only changes *how* the customer writes, so the
same tasks become a harder, more realistic test.

The one hard constraint: style must never change the task. The customer may be
terse, informal or impatient, but may not change their goal, misstate or mistype
IDs, names, dates or amounts, or invent new requests. Otherwise a harder score
would mean a broken task, not a harder conversation.
"""

from __future__ import annotations

from tau2.user.user_simulator import UserSimulator

HUMANLIKE_STYLE = """

## How you write (style only)
Write the way real people type into an airline support chat, not like an assistant:
- Keep messages short. Often one line, sometimes a fragment ("ok", "the one on the 20th").
  Only write a longer message when you are explaining your problem the first time.
- Casual register: contractions, lowercase is fine, little punctuation, no bullet lists,
  no "Certainly!" or "Thank you for your assistance".
- Give information when you are asked for it, not all up front; answer only the question asked.
- Show mild emotion when it fits the situation (impatience if the agent is slow or repeats
  itself, relief when something is fixed), but stay civil.
- Occasional small typos in ordinary words are fine (e.g. "reservaton", "tmrw").
- If the agent writes a long message, reply to the part that matters to you.

## What you must NOT change (this overrides the style rules above)
- Your goal, your constraints and every fact in your instructions stay exactly the same.
- Type IDs, names, dates, flight numbers, card details and amounts exactly as given,
  with no typos or abbreviations.
- Do not invent new requests, and do not change your mind unless your instructions say to.
- Keep using the conversation-ending signals exactly as described above."""


class HumanlikeUserSimulator(UserSimulator):
    """tau2's UserSimulator with realistic, human chat style layered on top."""

    @property
    def global_simulation_guidelines(self) -> str:
        return super().global_simulation_guidelines + HUMANLIKE_STYLE


def register() -> None:
    """Register as ``humanlike_user`` with tau2's registry (idempotent)."""
    from tau2.registry import registry

    try:
        registry.register_user(HumanlikeUserSimulator, "humanlike_user")
    except (ValueError, KeyError):
        pass  # already registered
