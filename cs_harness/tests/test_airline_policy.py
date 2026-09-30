"""Tests for the airline policy checker, against tau2's real airline database.

Run from the tau2-bench environment:
    cd third_party/tau2-bench && .venv/Scripts/python.exe -m pytest ../../cs_harness/tests -q
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis_cs.agents import _tool_db  # noqa: E402
from aegis_cs.airline_policy import NOW, CheckContext, check_tool_calls  # noqa: E402


class Call:
    def __init__(self, name, arguments, id="c1"):
        self.name, self.arguments, self.id = name, arguments, id


@pytest.fixture(scope="module")
def db():
    from tau2.registry import registry

    env = registry.get_env_constructor("airline")()
    return _tool_db(env.get_tools())


def ctx_for(db, user_id, reservation_id=None, said="Yes, please go ahead."):
    c = CheckContext(db=db, user_texts=["Hi, I need help.", said])
    c.looked_up_users.add(user_id)
    if reservation_id:
        c.looked_up_reservations.add(reservation_id)
    return c


def status(db, f):
    fl = db.flights.get(f.flight_number)
    return fl.dates[f.date].status if fl and f.date in fl.dates else None


def unflown(db, r):
    return all(status(db, f) not in ("landed", "flying") for f in r.flights)


def find(db, pred):
    for r in db.reservations.values():
        if pred(r):
            return r
    pytest.skip("no matching reservation in this database")


def old(r):
    return NOW - datetime.fromisoformat(r.created_at) > timedelta(hours=24)


def no_airline_cancel(db, r):
    return all(status(db, f) != "cancelled" for f in r.flights)


# ---------------------------------------------------------------- cancel


def test_cancel_business_allowed(db):
    r = find(db, lambda r: r.cabin == "business" and unflown(db, r) and r.status != "cancelled")
    out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})],
                           ctx_for(db, r.user_id, r.reservation_id))
    assert out == {}


def test_cancel_old_uninsured_economy_blocked(db):
    r = find(db, lambda r: r.cabin != "business" and r.insurance == "no" and old(r)
             and unflown(db, r) and no_airline_cancel(db, r) and r.status != "cancelled")
    out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})],
                           ctx_for(db, r.user_id, r.reservation_id))
    assert "not eligible for cancellation" in out["c1"][0]


def test_cancel_insured_needs_covered_reason(db):
    r = find(db, lambda r: r.cabin != "business" and r.insurance == "yes" and old(r)
             and unflown(db, r) and no_airline_cancel(db, r) and r.status != "cancelled")
    c = ctx_for(db, r.user_id, r.reservation_id)
    c.user_texts = ["I want to cancel, my plans changed.", "yes"]
    out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})], c)
    assert "insurance" in out["c1"][0]
    c.user_texts = ["I have to cancel, I'm sick and in the hospital.", "yes"]
    assert check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})], c) == {}


def test_cancel_flown_blocked(db):
    r = find(db, lambda r: not unflown(db, r) and r.status != "cancelled")
    out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})],
                           ctx_for(db, r.user_id, r.reservation_id))
    assert "already been flown" in out["c1"][0]


# -------------------------------------------------------------- modify


def test_basic_economy_flight_change_blocked(db):
    r = find(db, lambda r: r.cabin == "basic_economy" and unflown(db, r))
    new = [{"flight_number": f.flight_number, "date": "2024-05-25"} for f in r.flights]
    out = check_tool_calls([Call("update_reservation_flights", {
        "reservation_id": r.reservation_id, "cabin": r.cabin, "flights": new,
        "payment_id": next(iter(db.users[r.user_id].payment_methods))})],
        ctx_for(db, r.user_id, r.reservation_id))
    assert any("basic economy" in v for v in out["c1"])


def test_cabin_upgrade_keeping_flights_allowed(db):
    r = find(db, lambda r: r.cabin == "basic_economy" and unflown(db, r)
             and any(pm.source in ("credit_card", "gift_card")
                     for pm in db.users[r.user_id].payment_methods.values()))
    pid = next(k for k, pm in db.users[r.user_id].payment_methods.items()
               if pm.source in ("credit_card", "gift_card"))
    same = [{"flight_number": f.flight_number, "date": f.date} for f in r.flights]
    out = check_tool_calls([Call("update_reservation_flights", {
        "reservation_id": r.reservation_id, "cabin": "economy", "flights": same,
        "payment_id": pid})], ctx_for(db, r.user_id, r.reservation_id))
    assert out == {}


def test_passenger_count_cannot_change(db):
    r = find(db, lambda r: len(r.passengers) >= 1)
    extra = [p.model_dump() for p in r.passengers] + [
        {"first_name": "A", "last_name": "B", "dob": "1990-01-01"}]
    out = check_tool_calls([Call("update_reservation_passengers", {
        "reservation_id": r.reservation_id, "passengers": extra})],
        ctx_for(db, r.user_id, r.reservation_id))
    assert "number of passengers cannot change" in out["c1"][0]


def test_bags_cannot_be_removed_and_math_checked(db):
    r = find(db, lambda r: r.total_baggages >= 1)
    user = db.users[r.user_id]
    out = check_tool_calls([Call("update_reservation_baggages", {
        "reservation_id": r.reservation_id, "total_baggages": r.total_baggages - 1,
        "nonfree_baggages": 99, "payment_id": next(iter(user.payment_methods))})],
        ctx_for(db, r.user_id, r.reservation_id))
    text = " ".join(out["c1"])
    assert "only be added" in text and "nonfree_baggages should be" in text


# --------------------------------------------------------------- general


def test_confirmation_required(db):
    r = find(db, lambda r: r.cabin == "business" and unflown(db, r) and r.status != "cancelled")
    for said in ("Can you check something first?", "No, wait.", "not yet"):
        out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})],
                               ctx_for(db, r.user_id, r.reservation_id, said=said))
        assert any("explicit 'yes'" in v for v in out["c1"]), said


def test_direct_command_counts_as_consent(db):
    r = find(db, lambda r: r.cabin == "business" and unflown(db, r) and r.status != "cancelled")
    for said in ("I'm sick and can't travel. Just cancel both reservations.",
                 "Thank you. So, please cancel all my flights.", "Cancel it now.",
                 "Please proceed."):
        out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})],
                               ctx_for(db, r.user_id, r.reservation_id, said=said))
        assert out == {}, said
    for said in ("Can I cancel this?", "I want to know if I could cancel.", "No, don't cancel yet."):
        out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})],
                               ctx_for(db, r.user_id, r.reservation_id, said=said))
        assert out, said


def test_route_check_ignores_flight_order(db):
    r = find(db, lambda r: r.flight_type == "round_trip" and r.cabin != "basic_economy"
             and len(r.flights) == 2 and unflown(db, r)
             and all(status(db, f) == "available" for f in r.flights))
    pid = next(iter(db.users[r.user_id].payment_methods))
    reversed_flights = [{"flight_number": f.flight_number, "date": f.date} for f in reversed(r.flights)]
    out = check_tool_calls([Call("update_reservation_flights", {
        "reservation_id": r.reservation_id, "cabin": r.cabin, "flights": reversed_flights,
        "payment_id": pid})], ctx_for(db, r.user_id, r.reservation_id))
    assert not any("round trip" in v or "origin" in v for v in out.get("c1", []))


def test_unidentified_user_blocked(db):
    r = find(db, lambda r: r.cabin == "business" and unflown(db, r) and r.status != "cancelled")
    c = ctx_for(db, "someone_else_123", r.reservation_id)
    out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})], c)
    assert any("has not been identified" in v for v in out["c1"])


def test_user_id_stated_by_customer_counts(db):
    r = find(db, lambda r: r.cabin == "business" and unflown(db, r) and r.status != "cancelled")
    c = CheckContext(db=db, user_texts=[f"My user id is {r.user_id}.", "Yes, cancel it."])
    out = check_tool_calls([Call("cancel_reservation", {"reservation_id": r.reservation_id})], c)
    assert out == {}


def test_bundled_calls_pass_when_every_write_is_valid(db):
    r = find(db, lambda r: r.cabin == "business" and unflown(db, r) and r.status != "cancelled")
    out = check_tool_calls([
        Call("get_reservation_details", {"reservation_id": r.reservation_id}, id="a"),
        Call("cancel_reservation", {"reservation_id": r.reservation_id}, id="b"),
    ], ctx_for(db, r.user_id, r.reservation_id))
    assert out == {}


def test_bundle_held_back_when_one_write_violates(db):
    good = find(db, lambda r: r.cabin == "business" and unflown(db, r) and r.status != "cancelled")
    extra = [p.model_dump() for p in good.passengers] + [
        {"first_name": "A", "last_name": "B", "dob": "1990-01-01"}]
    out = check_tool_calls([
        Call("cancel_reservation", {"reservation_id": good.reservation_id}, id="a"),
        Call("update_reservation_passengers",
             {"reservation_id": good.reservation_id, "passengers": extra}, id="b"),
    ], ctx_for(db, good.user_id, good.reservation_id))
    assert "number of passengers cannot change" in out["b"][0]
    assert "another tool call in the same turn was blocked" in out["a"][0]


def test_reads_never_blocked(db):
    out = check_tool_calls([Call("get_user_details", {"user_id": "x"})],
                           CheckContext(db=db, user_texts=[]))
    assert out == {}


def test_certificate_rules(db):
    r = find(db, lambda r: db.users[r.user_id].membership == "regular" and r.insurance == "no"
             and r.cabin != "business"
             and all(x.insurance == "no" and x.cabin != "business"
                     for x in db.reservations.values() if x.user_id == r.user_id))
    out = check_tool_calls([Call("send_certificate", {"user_id": r.user_id, "amount": 100})],
                           ctx_for(db, r.user_id, r.reservation_id))
    assert "No compensation is allowed" in out["c1"][0]
    g = find(db, lambda r: db.users[r.user_id].membership == "gold")
    n = len(g.passengers)
    ok = check_tool_calls([Call("send_certificate", {"user_id": g.user_id, "amount": 100 * n})],
                          ctx_for(db, g.user_id, g.reservation_id))
    bad = check_tool_calls([Call("send_certificate", {"user_id": g.user_id, "amount": 999})],
                           ctx_for(db, g.user_id, g.reservation_id))
    assert ok == {} and "does not match policy" in bad["c1"][0]
