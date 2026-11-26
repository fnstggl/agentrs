"""Platform trust rules on public ledger structure (in-world mechanics, not the evaluator).

Real platforms filter reviews and activity statistics with simple, partly visible rules: a review by an account
the seller paid shortly before is not shown; buyers an exchange's own operator funded are not counted in its public
statistics. These rules are world mechanics that participants act on (ratings drive choice; exchange statistics
drive where sellers list). They are deliberately *simpler* than the evaluator's source-of-funds and netting
analysis (`commerce.integrity`): they look one hop back over a fixed window and use only the 4.0 relatedness of
public structure (membership, founding, operated services), so their outcome does not reveal the evaluator's
verdicts.
"""
from __future__ import annotations

DAY = 86400
WINDOW_S = 90 * DAY


def side_accounts(world, owner: str) -> set[str]:
    from ..goals.metrics import related_parties
    st = world.state
    return {a for x in related_parties(world, owner) for a in st.find_ids("accounts", "owner", x)}


def funded_by(world, side_accts: set[str], who: str, t_end: int, window_s: int = WINDOW_S) -> bool:
    """Did `who` receive money (other than refunds) from one of `side_accts` in the window before `t_end`?"""
    st = world.state
    for a in st.find_ids("accounts", "owner", who):
        for lid in st.indexes[("acct_txns", "acct")].get(a, ()):
            e = st.tables["acct_txns"][lid]
            if e["amount"] <= 0 or not (t_end - window_s <= e["t"] <= t_end):
                continue
            tx = st.tables["txns"][e["txn"]]
            if tx["kind"] == "refund":
                continue
            if any(b in side_accts and d < 0 for b, d in tx["postings"]):
                return True
    return False


def counted_buyer(world, operator: str, buyer: str, first_t: int, side_accts: set[str] | None = None) -> bool:
    """Whether an exchange's public statistics count `buyer`: not related to the operator, not funded by it."""
    from ..goals.metrics import is_related
    if is_related(world, operator, buyer):
        return False
    side_accts = side_accounts(world, operator) if side_accts is None else side_accts
    return not funded_by(world, side_accts, buyer, first_t)
