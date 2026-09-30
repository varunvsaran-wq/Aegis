"""Airline policy as code: checks a proposed database-changing tool call.

Written only from tau2-bench's airline ``policy.md`` (the document every agent is
given), before looking at any evaluation transcript, and frozen for the test
run. Each rule returns a short, specific violation message the agent can act
on ("reservation X is basic economy; its flights cannot be changed"), never
just "not allowed".

Most rules are exact: they read the airline database (read-only) and compare
against the proposed arguments. Three depend on what the customer said, which
code can only approximate, and are marked HEURISTIC:

- explicit confirmation ("yes") before a change,
- the cancellation reason being covered by insurance (health or weather),
- the customer having asked for compensation (not checked at all).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

#: The policy's fixed "current time".
NOW = datetime(2024, 5, 15, 15, 0, 0)

WRITE_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)

#: Writes that change a booking, so need an explicit "yes" first (policy §intro).
CONFIRM_REQUIRED = WRITE_TOOLS - {"send_certificate"}

#: Free checked bags per passenger, by membership then cabin (policy §Book flight).
FREE_BAGS = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}

FLOWN = frozenset({"landed", "flying"})

_YES = re.compile(
    r"\b(yes|yeah|yep|yup|confirm(ed)?|go ahead|proceed|please do|do it|sure|"
    r"that'?s (right|correct)|correct|ok(ay)?|sounds good)\b",
    re.IGNORECASE,
)
_COMMAND = re.compile(
    r"\s*(?:(?:so|ok|okay|alright|then|now|and)\s*,?\s*)*(?:please\s+|just\s+)*"
    r"(cancel|book|change|update|upgrade|modify|add|send|proceed|go ahead)\b",
    re.IGNORECASE,
)
_LEADING_NO = re.compile(
    r"\s*[\"'(]*\s*(no\b|nope\b|not yet|wait\b|hold on|don'?t\b|do not\b|stop\b)", re.IGNORECASE
)
_COVERED_REASON = re.compile(
    r"\b(sick|ill|illness|health|medical|hospital|doctor|surgery|injur\w*|"
    r"weather|storm|hurricane|snow\w*|blizzard|flood\w*|tornado|typhoon)\b",
    re.IGNORECASE,
)


@dataclass
class CheckContext:
    """What the checker may look at: the live DB (read-only) and the dialogue."""

    db: object
    user_texts: list[str]
    """Customer messages so far, oldest first."""

    looked_up_users: set[str] = field(default_factory=set)
    """User ids the agent fetched with get_user_details (the authenticated user)."""

    looked_up_reservations: set[str] = field(default_factory=set)


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _last_user_says_yes(ctx: CheckContext) -> bool:
    """HEURISTIC: the latest customer message reads as an explicit yes.

    A message that opens with a refusal ("no", "not yet", "wait") does not count
    even if an affirmative word appears later in it.
    """
    if not ctx.user_texts:
        return False
    last = ctx.user_texts[-1]
    if _LEADING_NO.match(last):
        return False
    if _YES.search(last):
        return True
    # A direct instruction to do the action ("Just cancel both reservations.",
    # "So, please cancel all my flights") is explicit consent too.
    return any(_COMMAND.match(s) for s in re.split(r"(?<=[.!?\n])\s+", last))


def _reservation(ctx: CheckContext, rid: str, out: list[str]):
    res = ctx.db.reservations.get(rid)
    if res is None:
        out.append(f"Reservation {rid} does not exist. Look it up with get_reservation_details.")
    return res


def _owner_checks(ctx: CheckContext, user_id: str, out: list[str]) -> None:
    """The policy: "The agent must first obtain the user id from the user."

    Satisfied if the customer stated the id or the agent looked it up.
    """
    stated = any(user_id and user_id in t for t in ctx.user_texts)
    if user_id not in ctx.looked_up_users and not stated:
        out.append(
            f"User {user_id} has not been identified in this conversation. Ask the customer "
            "for their user id and look it up with get_user_details before changing anything."
        )


def _segment_status(ctx: CheckContext, flight_number: str, date: str) -> str | None:
    flight = ctx.db.flights.get(flight_number)
    if flight is None or date not in flight.dates:
        return None
    return flight.dates[date].status


def _any_flown(ctx: CheckContext, res) -> list[str]:
    return [
        f"{f.flight_number} on {f.date}"
        for f in res.flights
        if _segment_status(ctx, f.flight_number, f.date) in FLOWN
    ]


def _route_problems(ctx: CheckContext, flights: list[dict], origin: str, destination: str,
                    flight_type: str, allow_unavailable: set[tuple[str, str]]) -> list[str]:
    """Flights must exist, be bookable, and match the origin/destination/trip type."""
    out: list[str] = []
    legs = []

    def when(f):  # the policy does not fix an order, so check the itinerary chronologically
        fl = ctx.db.flights.get(f.get("flight_number"))
        return (str(f.get("date")), getattr(fl, "scheduled_departure_time_est", "") or "")

    flights = sorted(flights, key=when)
    for f in flights:
        num, date = f.get("flight_number"), f.get("date")
        flight = ctx.db.flights.get(num)
        if flight is None:
            out.append(f"Flight {num} does not exist.")
            continue
        status = _segment_status(ctx, num, date)
        if status is None:
            out.append(f"Flight {num} does not operate on {date}.")
        elif status != "available" and (num, date) not in allow_unavailable:
            out.append(f"Flight {num} on {date} is '{status}' and cannot be booked.")
        legs.append((flight.origin, flight.destination))
    if not legs or len(legs) != len(flights):
        return out
    if legs[0][0] != origin:
        out.append(f"The first flight departs {legs[0][0]}, but the trip origin is {origin}.")
    if flight_type == "one_way" and legs[-1][1] != destination:
        out.append(f"The last flight arrives at {legs[-1][1]}, but the destination is {destination}.")
    if flight_type == "round_trip":
        if legs[-1][1] != origin:
            out.append(f"A round trip must return to {origin}; the last flight arrives at {legs[-1][1]}.")
        if destination not in {b for _, b in legs}:
            out.append(f"No flight in this round trip arrives at the destination {destination}.")
    return out


def _payment(ctx: CheckContext, user, pid: str, out: list[str], allowed=None) -> None:
    pm = user.payment_methods.get(pid) if user else None
    if pm is None:
        out.append(f"Payment method {pid} is not in the user's profile; only saved methods can be used.")
    elif allowed and pm.source not in allowed:
        out.append(f"Payment method {pid} is a {pm.source}; this change needs a gift card or credit card.")


def _baggage_math(membership: str, cabin: str, n_pax: int, total: int, nonfree: int,
                  out: list[str]) -> None:
    free = FREE_BAGS.get(membership, {}).get(cabin, 0) * n_pax
    expected = max(0, total - free)
    if nonfree != expected:
        out.append(
            f"nonfree_baggages should be {expected}: {n_pax} passenger(s) in {cabin} for a "
            f"{membership} member get {free} free bag(s), and {total} bag(s) were requested "
            f"(each extra bag is $50)."
        )


# ------------------------------------------------------------------ per tool


def _check_cancel(ctx, a, out):
    res = _reservation(ctx, a.get("reservation_id", ""), out)
    if res is None:
        return
    _owner_checks(ctx, res.user_id, out)
    if getattr(res, "status", None) == "cancelled":
        out.append(f"Reservation {res.reservation_id} is already cancelled.")
        return
    flown = _any_flown(ctx, res)
    if flown:
        out.append(
            f"Part of reservation {res.reservation_id} has already been flown ({', '.join(flown)}); "
            "you cannot cancel it. Transfer the customer to a human agent instead."
        )
        return
    created = datetime.fromisoformat(res.created_at)
    within_24h = NOW - created <= timedelta(hours=24)
    airline_cancelled = any(
        _segment_status(ctx, f.flight_number, f.date) == "cancelled" for f in res.flights
    )
    business = res.cabin == "business"
    insured = res.insurance == "yes"
    covered = any(_COVERED_REASON.search(t) for t in ctx.user_texts)  # HEURISTIC
    if within_24h or airline_cancelled or business or (insured and covered):
        return
    if insured:
        out.append(
            f"Reservation {res.reservation_id} can only be cancelled under its travel insurance, "
            "which covers health or weather reasons. Confirm the customer's reason; a change of "
            "plans is not covered, so the request must be denied."
        )
    else:
        out.append(
            f"Reservation {res.reservation_id} is not eligible for cancellation: it was booked "
            f"{created:%Y-%m-%d %H:%M} (more than 24 hours ago), no flight was cancelled by the "
            f"airline, the cabin is {res.cabin}, and it has no travel insurance. Deny the request."
        )


def _check_update_flights(ctx, a, out):
    res = _reservation(ctx, a.get("reservation_id", ""), out)
    if res is None:
        return
    _owner_checks(ctx, res.user_id, out)
    new = a.get("flights") or []
    old_set = {(f.flight_number, f.date) for f in res.flights}
    new_set = {(f.get("flight_number"), f.get("date")) for f in new}
    flights_changed = new_set != old_set
    cabin = a.get("cabin", res.cabin)
    if flights_changed and res.cabin == "basic_economy":
        out.append(
            f"Reservation {res.reservation_id} is basic economy, so its flights cannot be changed "
            "(only the cabin can be upgraded, keeping the same flights). Deny the flight change."
        )
    if cabin != res.cabin:
        flown = _any_flown(ctx, res)
        if flown:
            out.append(f"The cabin cannot be changed because {', '.join(flown)} has already been flown.")
    if flights_changed:
        out += _route_problems(ctx, new, res.origin, res.destination, res.flight_type,
                               allow_unavailable=old_set)
    user = ctx.db.users.get(res.user_id)
    if a.get("payment_id"):
        _payment(ctx, user, a["payment_id"], out,
                 allowed={"credit_card", "gift_card"} if flights_changed else None)


def _check_update_bags(ctx, a, out):
    res = _reservation(ctx, a.get("reservation_id", ""), out)
    if res is None:
        return
    _owner_checks(ctx, res.user_id, out)
    total, nonfree = int(a.get("total_baggages", 0)), int(a.get("nonfree_baggages", 0))
    if total < res.total_baggages:
        out.append(
            f"Checked bags can only be added, not removed: the reservation already has "
            f"{res.total_baggages} and the request is {total}."
        )
    user = ctx.db.users.get(res.user_id)
    _baggage_math(user.membership if user else "regular", res.cabin, len(res.passengers),
                  total, nonfree, out)
    if a.get("payment_id"):
        _payment(ctx, user, a["payment_id"], out)


def _check_update_passengers(ctx, a, out):
    res = _reservation(ctx, a.get("reservation_id", ""), out)
    if res is None:
        return
    _owner_checks(ctx, res.user_id, out)
    n_new, n_old = len(a.get("passengers") or []), len(res.passengers)
    if n_new != n_old:
        out.append(
            f"The number of passengers cannot change (currently {n_old}, requested {n_new}); "
            "passenger details can be edited but not added or removed."
        )


def _check_book(ctx, a, out):
    uid = a.get("user_id", "")
    user = ctx.db.users.get(uid)
    if user is None:
        out.append(f"User {uid} does not exist.")
        return
    _owner_checks(ctx, uid, out)
    pax = a.get("passengers") or []
    if not 1 <= len(pax) <= 5:
        out.append(f"A reservation must have between 1 and 5 passengers (requested {len(pax)}).")
    out += _route_problems(ctx, a.get("flights") or [], a.get("origin"), a.get("destination"),
                           a.get("flight_type"), allow_unavailable=set())
    methods = a.get("payment_methods") or []
    counts = {"certificate": 0, "credit_card": 0, "gift_card": 0}
    for m in methods:
        pid = m.get("payment_id")
        _payment(ctx, user, pid, out)
        pm = user.payment_methods.get(pid)
        if pm is not None:
            counts[pm.source] = counts.get(pm.source, 0) + 1
    if counts["certificate"] > 1 or counts["credit_card"] > 1 or counts["gift_card"] > 3:
        out.append(
            "A reservation can use at most one travel certificate, one credit card and three gift "
            f"cards (requested {counts['certificate']}, {counts['credit_card']}, {counts['gift_card']})."
        )
    _baggage_math(user.membership, a.get("cabin", ""), len(pax), int(a.get("total_baggages", 0)),
                  int(a.get("nonfree_baggages", 0)), out)


def _check_certificate(ctx, a, out):
    uid, amount = a.get("user_id", ""), a.get("amount", 0)
    user = ctx.db.users.get(uid)
    if user is None:
        out.append(f"User {uid} does not exist.")
        return
    _owner_checks(ctx, uid, out)
    discussed = [ctx.db.reservations[r] for r in ctx.looked_up_reservations
                 if r in ctx.db.reservations and ctx.db.reservations[r].user_id == uid]
    if not discussed:
        out.append("Confirm the facts first: look up the affected reservation before offering compensation.")
        return
    eligible = user.membership in ("silver", "gold") or any(
        r.insurance == "yes" or r.cabin == "business" for r in discussed
    )
    if not eligible:
        out.append(
            "No compensation is allowed: the user is a regular member, has no travel insurance on "
            "the discussed reservation, and does not fly business."
        )
        return
    allowed = sorted({per * len(r.passengers) for r in discussed for per in (50, 100)})
    if amount not in allowed:
        out.append(
            f"Certificate amount {_fmt_money(amount)} does not match policy: $100 per passenger for "
            f"a cancelled flight or $50 per passenger for a delay (allowed here: "
            f"{', '.join(_fmt_money(x) for x in allowed)})."
        )


_CHECKS = {
    "cancel_reservation": _check_cancel,
    "update_reservation_flights": _check_update_flights,
    "update_reservation_baggages": _check_update_bags,
    "update_reservation_passengers": _check_update_passengers,
    "book_reservation": _check_book,
    "send_certificate": _check_certificate,
}


def check_tool_calls(calls: list, ctx: CheckContext) -> dict[str, list[str]]:
    """Check the tool calls in one agent turn. Returns {tool_call_id: [violations]}.

    An empty dict means every call may proceed. Read-only calls are never
    blocked on their own. Each write in the turn is checked on its own merits;
    if any write violates the policy, the whole turn is held back (tool calls in
    one turn run together) and the other calls are told they were not run.

    The policy also says "one tool call at a time", but bundling does not change
    the outcome, and blocking it caused most false blocks on the development
    tasks, so it is not enforced.
    """
    problems: dict[str, list[str]] = {}
    for call in calls:
        if call.name not in WRITE_TOOLS:
            continue
        out: list[str] = []
        if call.name in CONFIRM_REQUIRED and not _last_user_says_yes(ctx):  # HEURISTIC
            out.append(
                "Before changing the booking, list the exact action details to the customer and "
                "get an explicit 'yes'. Do that now instead of calling the tool."
            )
        _CHECKS[call.name](ctx, dict(call.arguments or {}), out)
        if out:
            problems[call.id] = out
    if problems:
        for call in calls:
            problems.setdefault(call.id, [
                "Not executed because another tool call in the same turn was blocked. "
                "Resend it if it is still needed."
            ])
    return problems
