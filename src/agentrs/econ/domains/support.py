"""Customer support & account operations (retail + airline), τ²-Bench-style CUSTOM fallback.

State lives in SQLite (customers, orders, items, catalog, refunds, exchanges, flights,
reservations). The agent only acts through API tools; a deterministic scripted customer answers
messages (reveals account details on request, confirms when explicitly asked). Hidden verifiers
check the final database state AND policy compliance (no write before explicit confirmation; no
write at all when policy requires declining). This is not τ²-Bench: it is a local environment
built because τ²'s user simulator needs an LLM API.
"""
from __future__ import annotations

import datetime as dt
import random
import string

from agentrs.econ.core import (PRODUCTS, Family, Task, ToolEnv, ToolError, person, pool, register, CITIES)

SCHEMA = """
CREATE TABLE customers(id TEXT PRIMARY KEY, name TEXT, email TEXT, tier TEXT, zip TEXT);
CREATE TABLE orders(id TEXT PRIMARY KEY, customer_id TEXT, status TEXT, order_date TEXT, delivered_date TEXT, address TEXT);
CREATE TABLE order_items(item_id TEXT PRIMARY KEY, order_id TEXT, product TEXT, variant TEXT, price REAL, status TEXT);
CREATE TABLE catalog(product TEXT, variant TEXT, price REAL, in_stock INTEGER);
CREATE TABLE refunds(id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, item_id TEXT, amount REAL);
CREATE TABLE exchanges(id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, item_id TEXT, new_variant TEXT, price_difference REAL);
CREATE TABLE flights(flight_no TEXT, date TEXT, depart_time TEXT, origin TEXT, dest TEXT, cabin TEXT, price REAL);
CREATE TABLE reservations(id TEXT PRIMARY KEY, customer_id TEXT, flight_no TEXT, date TEXT, depart_time TEXT, origin TEXT, dest TEXT, cabin TEXT, status TEXT, price REAL, refund REAL);
"""
VARIANTS = ["size: S", "size: M", "size: L", "color: black", "color: blue", "color: red", "color: green",
            "color: white", "size: XL"]
RULES = [
    "Identify the customer by the email address on the account before reading or changing their data.",
    "Before ANY write action (return, exchange, address change, flight change, cancellation) describe it to the "
    "customer and obtain explicit confirmation (e.g. 'yes') in the conversation via send_message.",
    "Returns: only items of DELIVERED orders, and only within 30 days of the delivery date. Otherwise decline.",
    "Exchanges: only items of DELIVERED orders; the new variant of the same product must be in stock. Otherwise decline.",
    "Address changes: apply to every order of the customer with status 'pending' or 'processed'; never change "
    "delivered or cancelled orders.",
    "Flight changes: basic_economy reservations cannot be changed (decline). Otherwise move the reservation to "
    "the CHEAPEST flight in the same cabin on the requested date and route.",
    "Cancellations: gold-tier customers always get a full refund. Otherwise refund 100% if cancelled more than "
    "72 hours before departure, 50% if 24-72 hours before, 0 if less than 24 hours before.",
    "Finish with resolve(outcome='completed') after making the requested change, or resolve(outcome='declined') "
    "if the policy does not allow it.",
]
CONFIRM_WORDS = ("confirm", "proceed", "go ahead", "shall i", "should i", "would you like me", "do you want me",
                 "is that ok", "ok to", "okay to", "can i go ahead")


def _id(rng, prefix, n, alphabet=string.digits):
    return prefix + "".join(rng.choice(alphabet) for _ in range(n))


class SupportEnv(ToolEnv):
    domain = "support"
    role = "a customer-support agent for an online store and an airline"
    rules = RULES

    def setup(self):
        p = self.task.private
        db = self.new_db(SCHEMA)
        for table, rows in p["tables"].items():
            for r in rows:
                cols = ",".join(r)
                db.execute(f"INSERT INTO {table}({cols}) VALUES ({','.join('?' * len(r))})", list(r.values()))
        self.confirmed = False
        self.write_before_confirm = False
        self.messages: list[tuple[str, str]] = []
        self.outcome = None

        @self.tool("find_customer", "find_customer(email: str) — look up a customer by email")
        def find_customer(email):
            r = db.execute("SELECT id,name,tier,zip FROM customers WHERE lower(email)=lower(?)", (str(email).strip(),)).fetchone()
            if not r:
                raise ToolError("no customer with that email")
            return f"customer_id={r[0]} name={r[1]} tier={r[2]} zip={r[3]}"

        @self.tool("get_customer_orders", "get_customer_orders(customer_id: str) — list a customer's orders")
        def get_customer_orders(customer_id):
            rows = db.execute("SELECT id,status,order_date FROM orders WHERE customer_id=? ORDER BY order_date",
                              (customer_id,)).fetchall()
            if not rows:
                raise ToolError("no orders for that customer_id")
            return "\n".join(f"order {r[0]} status={r[1]} ordered={r[2]}" for r in rows)

        @self.tool("get_order", "get_order(order_id: str) — order details and items")
        def get_order(order_id):
            o = db.execute("SELECT id,status,order_date,delivered_date,address FROM orders WHERE id=?", (order_id,)).fetchone()
            if not o:
                raise ToolError("order not found")
            items = db.execute("SELECT item_id,product,variant,price,status FROM order_items WHERE order_id=?", (order_id,)).fetchall()
            lines = [f"order {o[0]} status={o[1]} ordered={o[2]} delivered={o[3] or '-'} address={o[4]}"]
            lines += [f"  item {i[0]}: {i[1]} ({i[2]}) ${i[3]:.2f} status={i[4]}" for i in items]
            return "\n".join(lines)

        @self.tool("get_catalog", "get_catalog(product: str) — variants, prices and stock of a product")
        def get_catalog(product):
            rows = db.execute("SELECT variant,price,in_stock FROM catalog WHERE lower(product)=lower(?)", (product,)).fetchall()
            if not rows:
                raise ToolError("unknown product")
            return "\n".join(f"{product} ({v}) ${pr:.2f} {'in stock' if s else 'OUT OF STOCK'}" for v, pr, s in rows)

        def _write():
            if not self.confirmed:
                self.write_before_confirm = True

        @self.tool("return_items", "return_items(order_id: str, item_ids: list[str]) — return items and refund them",
                   writes=True)
        def return_items(order_id, item_ids):
            o = db.execute("SELECT status FROM orders WHERE id=?", (order_id,)).fetchone()
            if not o:
                raise ToolError("order not found")
            if o[0] != "delivered":
                raise ToolError("only delivered orders can be returned")
            if isinstance(item_ids, str):
                item_ids = [item_ids]
            for iid in item_ids:
                r = db.execute("SELECT price,status FROM order_items WHERE item_id=? AND order_id=?", (iid, order_id)).fetchone()
                if not r or r[1] != "ok":
                    raise ToolError(f"item {iid} is not returnable")
            _write()
            for iid in item_ids:
                price = db.execute("SELECT price FROM order_items WHERE item_id=?", (iid,)).fetchone()[0]
                db.execute("UPDATE order_items SET status='returned' WHERE item_id=?", (iid,))
                db.execute("INSERT INTO refunds(order_id,item_id,amount) VALUES (?,?,?)", (order_id, iid, price))
            return f"returned {len(item_ids)} item(s); refund issued"

        @self.tool("exchange_item", "exchange_item(order_id: str, item_id: str, new_variant: str) — exchange an item "
                   "for another variant of the same product", writes=True)
        def exchange_item(order_id, item_id, new_variant):
            o = db.execute("SELECT status FROM orders WHERE id=?", (order_id,)).fetchone()
            if not o or o[0] != "delivered":
                raise ToolError("only items of delivered orders can be exchanged")
            it = db.execute("SELECT product,price,status FROM order_items WHERE item_id=? AND order_id=?", (item_id, order_id)).fetchone()
            if not it or it[2] != "ok":
                raise ToolError("item not exchangeable")
            c = db.execute("SELECT price,in_stock FROM catalog WHERE product=? AND variant=?", (it[0], new_variant)).fetchone()
            if not c:
                raise ToolError("unknown variant")
            if not c[1]:
                raise ToolError("variant out of stock")
            _write()
            db.execute("UPDATE order_items SET status='exchanged' WHERE item_id=?", (item_id,))
            db.execute("INSERT INTO exchanges(order_id,item_id,new_variant,price_difference) VALUES (?,?,?,?)",
                       (order_id, item_id, new_variant, round(c[0] - it[1], 2)))
            return f"exchange created; price difference ${c[0] - it[1]:.2f}"

        @self.tool("update_order_address", "update_order_address(order_id: str, address: str) — change shipping address",
                   writes=True)
        def update_order_address(order_id, address):
            o = db.execute("SELECT status FROM orders WHERE id=?", (order_id,)).fetchone()
            if not o:
                raise ToolError("order not found")
            if o[0] not in ("pending", "processed"):
                raise ToolError(f"cannot change address of a {o[0]} order")
            _write()
            db.execute("UPDATE orders SET address=? WHERE id=?", (str(address), order_id))
            return "address updated"

        @self.tool("get_reservations", "get_reservations(customer_id: str) — list flight reservations")
        def get_reservations(customer_id):
            rows = db.execute("SELECT id,flight_no,date,depart_time,origin,dest,cabin,status,price FROM reservations "
                              "WHERE customer_id=?", (customer_id,)).fetchall()
            if not rows:
                raise ToolError("no reservations")
            return "\n".join(f"reservation {r[0]}: flight {r[1]} {r[4]}->{r[5]} on {r[2]} {r[3]} cabin={r[6]} "
                             f"status={r[7]} paid=${r[8]:.2f}" for r in rows)

        @self.tool("search_flights", "search_flights(origin: str, dest: str, date: str) — flights on a date (YYYY-MM-DD)")
        def search_flights(origin, dest, date):
            rows = db.execute("SELECT flight_no,depart_time,cabin,price FROM flights WHERE lower(origin)=lower(?) AND "
                              "lower(dest)=lower(?) AND date=? ORDER BY flight_no, cabin", (origin, dest, date)).fetchall()
            if not rows:
                return "no flights"
            return "\n".join(f"flight {r[0]} departs {r[1]} cabin={r[2]} price=${r[3]:.2f}" for r in rows)

        @self.tool("change_reservation", "change_reservation(reservation_id: str, flight_no: str, date: str) — move a "
                   "reservation to another flight", writes=True)
        def change_reservation(reservation_id, flight_no, date):
            r = db.execute("SELECT cabin,origin,dest,status FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if not r or r[3] != "active":
                raise ToolError("reservation not found or not active")
            f = db.execute("SELECT price,depart_time FROM flights WHERE flight_no=? AND date=? AND cabin=? AND origin=? AND dest=?",
                           (flight_no, date, r[0], r[1], r[2])).fetchone()
            if not f:
                raise ToolError("no such flight in the reservation's cabin and route")
            _write()
            db.execute("UPDATE reservations SET flight_no=?, date=?, depart_time=?, price=? WHERE id=?",
                       (flight_no, date, f[1], f[0], reservation_id))
            return f"reservation {reservation_id} moved to {flight_no} on {date}"

        @self.tool("cancel_reservation", "cancel_reservation(reservation_id: str, refund_amount: float) — cancel and refund",
                   writes=True)
        def cancel_reservation(reservation_id, refund_amount):
            r = db.execute("SELECT status FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if not r or r[0] != "active":
                raise ToolError("reservation not found or not active")
            _write()
            db.execute("UPDATE reservations SET status='cancelled', refund=? WHERE id=?", (float(refund_amount), reservation_id))
            return f"reservation {reservation_id} cancelled; refund ${float(refund_amount):.2f}"

        @self.tool("send_message", "send_message(text: str) — message the customer and read their reply")
        def send_message(text):
            reply = self._customer_reply(str(text))
            self.messages.append((str(text), reply))
            return f"Customer: {reply}"

        @self.tool("resolve", "resolve(outcome: 'completed'|'declined') — close the ticket (terminal)", terminal=True)
        def resolve(outcome):
            if outcome not in ("completed", "declined"):
                raise ToolError("outcome must be 'completed' or 'declined'")
            self.outcome = outcome
            return f"ticket closed: {outcome}"

    def _customer_reply(self, text: str) -> str:
        c = self.task.private["customer"]
        t = text.lower()
        out = []
        if "email" in t or "account" in t:
            out.append(f"My email is {c['email']}.")
        if any(w in t for w in CONFIRM_WORDS) or t.rstrip().endswith("?") and ("yes" in t or "correct" in t):
            out.append("Yes, please go ahead.")
            self.confirmed = True
        if not out:
            out.append(f"{c['request']} My email is {c['email']}.")
        return " ".join(out)

    def initial_observation(self) -> str:
        p = self.task.private
        return f"Current time: {p['now']}. New ticket. Customer message: \"{p['customer']['request']}\""

    def verify(self):
        p = self.task.private
        exp = p["expect"]
        db = self.db
        writes = [e for e in self.log if e["ok"] and self.tools[e["tool"]].writes]
        details = {"outcome": self.outcome, "expected_outcome": exp["outcome"], "writes": len(writes),
                   "write_before_confirm": self.write_before_confirm}
        if self.outcome != exp["outcome"]:
            return False, details | {"reason": "wrong outcome"}
        if exp["outcome"] == "declined":
            ok = not writes
            return ok, details | ({} if ok else {"reason": "wrote although the request had to be declined"})
        if self.write_before_confirm:
            return False, details | {"reason": "write before explicit customer confirmation"}
        for q, want in exp["queries"]:
            got = [tuple(round(x, 2) if isinstance(x, float) else x for x in r) for r in db.execute(q).fetchall()]
            want = [tuple(round(x, 2) if isinstance(x, float) else x for x in r) for r in want]
            if sorted(got, key=str) != sorted(want, key=str):
                return False, details | {"reason": "state mismatch", "query": q, "got": got[:5], "want": want[:5]}
        return True, details


# ---------------------------------------------------------------------------- task generation
def _base(split, seed):
    rng = random.Random(f"support:{split}:{seed}")
    now = dt.datetime(2026, 5, 1, 9) + dt.timedelta(days=rng.randint(0, 120), hours=rng.randint(0, 9))
    tables = {k: [] for k in ("customers", "orders", "order_items", "catalog", "flights", "reservations")}
    custs = []
    used_names = set()
    for k in range(3):
        while True:
            name = person(rng, split)
            if name not in used_names:
                used_names.add(name)
                break
        first, last = name.lower().split()
        cid = _id(rng, "C", 5)
        email = f"{first}.{last}{rng.randint(1, 99)}@example.com"
        cust = {"id": cid, "name": name, "email": email, "tier": rng.choice(["standard", "standard", "gold"]),
                "zip": f"{rng.randint(10000, 99999)}"}
        tables["customers"].append(cust)
        custs.append(cust)
    return rng, now, tables, custs


def _add_order(rng, tables, cust, now, split, status, delivered_days_ago=None, n_items=None, products=None):
    oid = _id(rng, "#W", 7)
    od = now - dt.timedelta(days=(delivered_days_ago or 0) + rng.randint(2, 6))
    delivered = (now - dt.timedelta(days=delivered_days_ago)).date().isoformat() if status == "delivered" else None
    tables["orders"].append({"id": oid, "customer_id": cust["id"], "status": status, "order_date": od.date().isoformat(),
                             "delivered_date": delivered, "address": f"{rng.randint(10, 999)} Main St, {rng.choice(CITIES)}"})
    items = []
    prods = products or rng.sample(PRODUCTS, n_items or rng.randint(1, 3))
    for prod in prods:
        var = rng.choice(VARIANTS)
        price = round(rng.uniform(12, 260), 2)
        it = {"item_id": _id(rng, "I", 6), "order_id": oid, "product": prod, "variant": var, "price": price, "status": "ok"}
        tables["order_items"].append(it)
        items.append(it)
    return oid, items


def _confirm_msg(desc):
    return f"I can {desc}. Do you confirm that you want me to proceed?"


def make_return(split, seed):
    rng, now, tables, custs = _base(split, seed)
    cust = custs[0]
    case = rng.choices(["ok", "late", "undelivered"], [0.6, 0.2, 0.2])[0]
    prods = rng.sample(PRODUCTS, 6)
    days = rng.randint(1, 29) if case == "ok" else rng.randint(35, 90)
    status = "processed" if case == "undelivered" else "delivered"
    oid, items = _add_order(rng, tables, cust, now, split, status, None if status != "delivered" else days,
                            products=prods[:rng.randint(2, 3)])
    _add_order(rng, tables, cust, now, split, "delivered", rng.randint(3, 60), products=prods[3:5])
    for c in custs[1:]:
        _add_order(rng, tables, c, now, split, "delivered", rng.randint(1, 40))
    ret = rng.sample(items, rng.randint(1, min(2, len(items))))
    names = " and the ".join(f"{i['product']} ({i['variant']})" for i in ret)
    request = f"Hi, I'd like to return the {names} from my recent order."
    ids = [i["item_id"] for i in ret]
    if case == "ok":
        expect = {"outcome": "completed", "queries": [
            ("SELECT item_id, status FROM order_items WHERE status!='ok'", [(i, "returned") for i in ids]),
            ("SELECT order_id, item_id, amount FROM refunds", [(oid, i["item_id"], i["price"]) for i in ret])]}
    else:
        expect = {"outcome": "declined", "queries": []}
    plan = ["send_message|Could you please share the email address on your account?",
            f"find_customer|{cust['email']}", f"get_customer_orders|{cust['id']}", f"get_order|{oid}"]
    if case == "ok":
        plan += [f"msg_confirm|return {names} from order {oid} for a refund", f"return_items|{oid}|{','.join(ids)}",
                 "resolve|completed"]
    else:
        plan += ["send_message|I'm sorry, but under our policy these items are not eligible for a return.",
                 "resolve|declined"]
    return _task("return_items", split, seed, request, now, tables, cust, expect, plan)


def make_exchange(split, seed):
    rng, now, tables, custs = _base(split, seed)
    cust = custs[0]
    case = rng.choices(["ok", "oos", "undelivered"], [0.6, 0.2, 0.2])[0]
    status = "processed" if case == "undelivered" else "delivered"
    oid, items = _add_order(rng, tables, cust, now, split, status, rng.randint(2, 50) if status == "delivered" else None,
                            n_items=rng.randint(1, 3))
    for c in custs[1:]:
        _add_order(rng, tables, c, now, split, "delivered", rng.randint(1, 40))
    item = rng.choice(items)
    others = [v for v in VARIANTS if v != item["variant"]]
    new_var = rng.choice(others)
    for v in rng.sample(others, 3) + [new_var, item["variant"]]:
        if any(r["product"] == item["product"] and r["variant"] == v for r in tables["catalog"]):
            continue
        stock = 0 if (v == new_var and case == "oos") else (1 if v == new_var else rng.randint(0, 1))
        tables["catalog"].append({"product": item["product"], "variant": v,
                                  "price": round(item["price"] + rng.choice([-10, 0, 0, 5, 15]), 2) if v != item["variant"] else item["price"],
                                  "in_stock": stock})
    new_price = next(r["price"] for r in tables["catalog"] if r["product"] == item["product"] and r["variant"] == new_var)
    request = f"Hello, I need to exchange my {item['product']} ({item['variant']}) for the {new_var} version."
    if case == "ok":
        expect = {"outcome": "completed", "queries": [
            ("SELECT order_id, item_id, new_variant, price_difference FROM exchanges",
             [(oid, item["item_id"], new_var, round(new_price - item["price"], 2))]),
            ("SELECT item_id, status FROM order_items WHERE status!='ok'", [(item["item_id"], "exchanged")])]}
    else:
        expect = {"outcome": "declined", "queries": []}
    plan = ["send_message|Could you please share the email address on your account?",
            f"find_customer|{cust['email']}", f"get_customer_orders|{cust['id']}", f"get_order|{oid}",
            f"get_catalog|{item['product']}"]
    if case == "ok":
        plan += [f"msg_confirm|exchange your {item['product']} for the {new_var} variant",
                 f"exchange_item|{oid}|{item['item_id']}|{new_var}", "resolve|completed"]
    else:
        plan += ["send_message|I'm sorry, this exchange is not possible under our policy.", "resolve|declined"]
    return _task("exchange_item", split, seed, request, now, tables, cust, expect, plan)


def make_address(split, seed):
    rng, now, tables, custs = _base(split, seed)
    cust = custs[0]
    statuses = ["pending", rng.choice(["processed", "pending"]), "delivered", rng.choice(["delivered", "cancelled", "processed"])]
    rng.shuffle(statuses)
    oids = []
    for s in statuses:
        oid, _ = _add_order(rng, tables, cust, now, split, s, rng.randint(2, 30) if s == "delivered" else None)
        oids.append((oid, s))
    for c in custs[1:]:
        _add_order(rng, tables, c, now, split, "pending")
    addr = f"{rng.randint(10, 999)} {rng.choice(['Oak', 'Elm', 'Cedar', 'Lake', 'Hill'])} Ave, {rng.choice(pool(CITIES, split))}"
    request = f"Hi, I moved. Please ship all my open orders to {addr} instead."
    targets = [o for o, s in oids if s in ("pending", "processed")]
    expect = {"outcome": "completed", "queries": [
        (f"SELECT id FROM orders WHERE customer_id='{cust['id']}' AND address='{addr}'", [(o,) for o in targets])]}
    plan = ["send_message|Could you please share the email address on your account?",
            f"find_customer|{cust['email']}", f"get_customer_orders|{cust['id']}",
            f"msg_confirm|update the shipping address of orders {', '.join(targets)} to {addr}"]
    plan += [f"update_order_address|{o}|{addr}" for o in targets] + ["resolve|completed"]
    return _task("address_update", split, seed, request, now, tables, cust, expect, plan)


ROUTES = [("Austin", "Denver"), ("Boston", "Chicago"), ("Seattle", "Phoenix"), ("Miami", "Atlanta"),
          ("Dallas", "Portland"), ("Detroit", "Tampa"), ("Omaha", "Reno"), ("Boise", "Madison")]


def _flights(rng, tables, origin, dest, date, n=4):
    out = []
    used = {f["flight_no"] for f in tables["flights"]}
    for _ in range(n):
        fno = f"{rng.choice(['AX', 'BL', 'CR'])}{rng.randint(100, 999)}"
        while fno in used:
            fno = f"{rng.choice(['AX', 'BL', 'CR'])}{rng.randint(100, 999)}"
        used.add(fno)
        t = f"{rng.randint(6, 21):02d}:{rng.choice(['00', '15', '30', '45'])}"
        for cabin in rng.sample(["basic_economy", "economy", "business"], rng.randint(2, 3)):
            base = {"basic_economy": 90, "economy": 160, "business": 520}[cabin]
            f = {"flight_no": fno, "date": date, "depart_time": t, "origin": origin, "dest": dest, "cabin": cabin,
                 "price": float(base + rng.randint(0, 140))}
            tables["flights"].append(f)
            out.append(f)
    return out


def _reservation(rng, tables, cust, now, split, cabin, days_ahead_h):
    origin, dest = rng.choice(ROUTES if split != "test" else [(b, a) for a, b in ROUTES])
    dep = now + dt.timedelta(hours=days_ahead_h)
    date = dep.date().isoformat()
    t = f"{dep.hour:02d}:{dep.minute:02d}"
    rid = _id(rng, "", 6, string.ascii_uppercase)
    price = float({"basic_economy": 99, "economy": 189, "business": 610}[cabin] + rng.randint(0, 90))
    r = {"id": rid, "customer_id": cust["id"], "flight_no": f"{rng.choice(['AX', 'BL'])}{rng.randint(100, 999)}",
         "date": date, "depart_time": t, "origin": origin, "dest": dest, "cabin": cabin, "status": "active",
         "price": price, "refund": None}
    tables["reservations"].append(r)
    return r


def make_modify_res(split, seed):
    rng, now, tables, custs = _base(split, seed)
    cust = custs[0]
    case = rng.choices(["ok", "basic"], [0.75, 0.25])[0]
    cabin = "basic_economy" if case == "basic" else rng.choice(["economy", "business"])
    r = _reservation(rng, tables, cust, now, split, cabin, rng.randint(48, 400))
    new_date = (dt.date.fromisoformat(r["date"]) + dt.timedelta(days=rng.choice([-2, -1, 1, 2, 3]))).isoformat()
    fl = _flights(rng, tables, r["origin"], r["dest"], new_date, n=rng.randint(3, 5))
    if not any(f["cabin"] == cabin for f in fl):
        f = dict(fl[0], cabin=cabin, price=float(rng.randint(100, 600)))
        tables["flights"].append(f)
        fl.append(f)
    _reservation(rng, tables, custs[1], now, split, "economy", rng.randint(30, 300))
    best = min((f for f in fl if f["cabin"] == cabin), key=lambda f: (f["price"], f["flight_no"]))
    request = f"Hi, can you move my flight from {r['origin']} to {r['dest']} on {r['date']} to {new_date}? Keep the same cabin."
    if case == "ok":
        expect = {"outcome": "completed", "queries": [
            (f"SELECT id, flight_no, date, price FROM reservations WHERE id='{r['id']}'",
             [(r["id"], best["flight_no"], new_date, best["price"])])]}
    else:
        expect = {"outcome": "declined", "queries": []}
    plan = ["send_message|Could you please share the email address on your account?",
            f"find_customer|{cust['email']}", f"get_reservations|{cust['id']}"]
    if case == "ok":
        plan += [f"search_flights|{r['origin']}|{r['dest']}|{new_date}",
                 f"msg_confirm|move reservation {r['id']} to flight {best['flight_no']} on {new_date} (${best['price']:.2f})",
                 f"change_reservation|{r['id']}|{best['flight_no']}|{new_date}", "resolve|completed"]
    else:
        plan += ["send_message|I'm sorry, basic economy reservations cannot be changed.", "resolve|declined"]
    return _task("modify_reservation", split, seed, request, now, tables, cust, expect, plan)


def make_cancel_res(split, seed):
    rng, now, tables, custs = _base(split, seed)
    cust = custs[0]
    hours = rng.choice([rng.randint(5, 23), rng.randint(25, 71), rng.randint(73, 400)])
    r = _reservation(rng, tables, cust, now, split, rng.choice(["basic_economy", "economy", "business"]), hours)
    _reservation(rng, tables, custs[2], now, split, "business", rng.randint(30, 300))
    dep = dt.datetime.fromisoformat(f"{r['date']}T{r['depart_time']}")
    h = (dep - now).total_seconds() / 3600
    frac = 1.0 if cust["tier"] == "gold" or h > 72 else (0.5 if h >= 24 else 0.0)
    refund = round(r["price"] * frac, 2)
    request = f"Hello, please cancel my flight from {r['origin']} to {r['dest']} on {r['date']}."
    expect = {"outcome": "completed", "queries": [
        (f"SELECT id, status, refund FROM reservations WHERE status='cancelled'", [(r["id"], "cancelled", refund)])]}
    plan = ["send_message|Could you please share the email address on your account?",
            f"find_customer|{cust['email']}", f"get_reservations|{cust['id']}",
            f"msg_confirm|cancel reservation {r['id']} with a refund of ${refund:.2f}",
            f"cancel_reservation|{r['id']}|{refund}", "resolve|completed"]
    return _task("cancel_reservation", split, seed, request, now, tables, cust, expect, plan)


def _task(family, split, seed, request, now, tables, cust, expect, plan):
    goal = (f"Handle this customer-support ticket according to the policy. Customer message: \"{request}\" "
            f"(current time {now.strftime('%Y-%m-%d %H:%M')})")
    return Task("support", family, split, seed, goal,
                private={"tables": tables, "now": now.strftime("%Y-%m-%d %H:%M"), "expect": expect, "plan": plan,
                         "customer": {"email": cust["email"], "request": request}},
                max_steps=12)


def oracle(task: Task) -> list[str]:
    import json

    acts = []
    for step in task.private["plan"]:
        name, *a = step.split("|")
        if name == "send_message":
            acts.append({"tool": "send_message", "args": {"text": a[0]}})
        elif name == "msg_confirm":
            acts.append({"tool": "send_message", "args": {"text": _confirm_msg(a[0])}})
        elif name == "find_customer":
            acts.append({"tool": name, "args": {"email": a[0]}})
        elif name in ("get_customer_orders", "get_reservations"):
            acts.append({"tool": name, "args": {"customer_id": a[0]}})
        elif name == "get_order":
            acts.append({"tool": name, "args": {"order_id": a[0]}})
        elif name == "get_catalog":
            acts.append({"tool": name, "args": {"product": a[0]}})
        elif name == "return_items":
            acts.append({"tool": name, "args": {"order_id": a[0], "item_ids": a[1].split(",")}})
        elif name == "exchange_item":
            acts.append({"tool": name, "args": {"order_id": a[0], "item_id": a[1], "new_variant": a[2]}})
        elif name == "update_order_address":
            acts.append({"tool": name, "args": {"order_id": a[0], "address": a[1]}})
        elif name == "search_flights":
            acts.append({"tool": name, "args": {"origin": a[0], "dest": a[1], "date": a[2]}})
        elif name == "change_reservation":
            acts.append({"tool": name, "args": {"reservation_id": a[0], "flight_no": a[1], "date": a[2]}})
        elif name == "cancel_reservation":
            acts.append({"tool": name, "args": {"reservation_id": a[0], "refund_amount": float(a[1])}})
        elif name == "resolve":
            acts.append({"tool": "resolve", "args": {"outcome": a[0]}})
    return [json.dumps(x) for x in acts]


for _name, _fn in [("return_items", make_return), ("exchange_item", make_exchange), ("address_update", make_address),
                   ("modify_reservation", make_modify_res), ("cancel_reservation", make_cancel_res)]:
    register(Family("support", _name, _fn, SupportEnv, oracle))
