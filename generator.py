"""Synthetic enterprise knowledge graph where organizational boundaries (bounded contexts)
and topological communities disagree.

Five bounded contexts (CUSTOMER, PAYMENTS, LENDING, RISK, SUPPORT) plus a shared identity
kernel of Party nodes. Parties are grouped into households, so the densest regions of the
graph are household-centric and span every context, while ownership follows contexts.
Node texts deliberately share vocabulary across contexts (account, payment, limit, ...).
"""
import random
from dataclasses import dataclass, field

CONTEXTS = ["CUSTOMER", "PAYMENTS", "LENDING", "RISK", "SUPPORT"]
IDENTITY = "IDENTITY"

FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
         "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Priya", "Wei",
         "Carlos", "Ana", "Ahmed", "Fatima", "Kenji", "Yuki", "Olga", "Ivan", "Grace", "Daniel"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Patel", "Nguyen", "Kim", "Chen", "Singh", "Lopez", "Wilson", "Anderson", "Taylor", "Moore"]
CITIES = ["Chicago", "Austin", "Denver", "Seattle", "Boston", "Phoenix", "Atlanta", "Miami", "Portland", "Dallas",
          "Detroit", "Tampa", "Omaha", "Raleigh", "Madison", "Tucson", "Fresno", "Boise", "Reno", "Albany"]
MERCHANTS = ["Walmart", "Target", "Costco", "Amazon", "Shell", "Starbucks", "Uber", "Netflix", "Apple", "Delta",
             "Marriott", "Home Depot", "Kroger", "Walgreens", "Chipotle", "Spotify", "Lyft", "Best Buy", "IKEA", "Exxon"]
ACCT = ["checking", "savings", "credit card", "money market"]
LOAN = ["auto", "mortgage", "personal", "student", "home equity"]
LOAN_STATUS = ["current", "delinquent", "paid off", "in forbearance"]
SEV = ["low", "medium", "high", "critical"]
ALERT_KIND = ["unusual location", "velocity spike", "card not present", "large transfer"]
TICKET_TOPIC = ["disputed charge", "late fee", "payment not posted", "limit increase", "autopay failure",
                "address change", "card replacement", "statement error"]
SEGMENTS = ["mass market", "emerging affluent", "small business", "private client"]


@dataclass
class Graph:
    nodes: list = field(default_factory=list)   # dict(id, ctx, type, text, party)
    edges: list = field(default_factory=list)   # (u, v, kind) kind in {intra, identity, reference, household, merchant}

    def add(self, ctx, typ, text, party):
        i = len(self.nodes)
        self.nodes.append({"id": i, "ctx": ctx, "type": typ, "text": text, "party": party})
        return i


def generate(n_parties, seed):
    rng = random.Random(seed)
    g = Graph()
    merchant_ids = {m: g.add("PAYMENTS", "Merchant", f"merchant {m} payment acceptor", None) for m in MERCHANTS}
    parties, facts = [], []
    hh_size = 0
    hh_members = []
    for p in range(n_parties):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        city = rng.choice(CITIES) if not hh_members or hh_size == 0 else parties[hh_members[0]]["city"]
        pid = g.add(IDENTITY, "Party", f"party {name} {city} customer identity", p)
        party = {"pid": pid, "name": name, "city": city, "accounts": [], "txns": [], "loans": [], "alerts": [], "tickets": []}
        parties.append(party)
        # households: consecutive parties, size 2-4, share city and link to each other
        if hh_size == 0:
            hh_size = rng.randint(2, 4)
            hh_members = []
        for q in hh_members:
            g.edges.append((pid, parties[q]["pid"], "household"))
        hh_members.append(p)
        hh_size -= 1

        prof = g.add("CUSTOMER", "Profile", f"customer profile {name} {city} segment {rng.choice(SEGMENTS)} "
                     f"contact preferences account holder since {rng.randint(2001, 2025)}", p)
        g.edges.append((prof, pid, "identity"))
        for _ in range(rng.randint(1, 3)):
            kind = rng.choice(ACCT)
            last4 = rng.randint(1000, 9999)
            a = g.add("PAYMENTS", "Account", f"{kind} account ending {last4} held by {name} credit limit payment due", p)
            g.edges += [(a, pid, "identity")]
            party["accounts"].append((a, kind, last4))
            for _ in range(rng.randint(2, 5)):
                m = rng.choice(MERCHANTS)
                amt = round(rng.uniform(5, 900), 2)
                day = f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"
                t = g.add("PAYMENTS", "Transaction", f"transaction {m} {amt} on {day} account ending {last4} payment", p)
                g.edges += [(t, a, "intra"), (t, merchant_ids[m], "merchant")]
                party["txns"].append((t, m, amt, day, last4))
        for _ in range(rng.choice([0, 1, 1, 2])):
            kind, status = rng.choice(LOAN), rng.choice(LOAN_STATUS)
            ln = g.add("LENDING", "Loan", f"{kind} loan borrower {name} status {status} monthly payment account autopay", p)
            g.edges.append((ln, pid, "identity"))
            if party["accounts"]:
                g.edges.append((ln, rng.choice(party["accounts"])[0], "reference"))  # autopay from account
            for _ in range(rng.randint(1, 2)):
                ev = g.add("LENDING", "LoanEvent", f"{kind} loan payment event {rng.choice(['on time', 'late', 'missed'])} "
                           f"limit balance", p)
                g.edges.append((ev, ln, "intra"))
            party["loans"].append((ln, kind, status))
        for _ in range(rng.choice([0, 0, 1, 2])):
            if not party["txns"]:
                break
            t = rng.choice(party["txns"])
            sev, ak = rng.choice(SEV), rng.choice(ALERT_KIND)
            al = g.add("RISK", "Alert", f"{sev} risk alert {ak} customer {name.split()[1]} payment account review", p)
            g.edges += [(al, pid, "identity"), (al, t[0], "reference")]
            party["alerts"].append((al, sev, ak, t))
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            topic = rng.choice(TICKET_TOPIC)
            tk = g.add("SUPPORT", "Ticket", f"support ticket {name} called about {topic} account payment limit", p)
            g.edges.append((tk, pid, "identity"))
            ref = None
            if party["loans"] and rng.random() < 0.4:
                ref = ("loan", rng.choice(party["loans"]))
            elif party["txns"]:
                ref = ("txn", rng.choice(party["txns"]))
            if ref:
                g.edges.append((tk, ref[1][0], "reference"))
            party["tickets"].append((tk, topic, ref))
    return g, parties


TEMPLATES = {
    # name: (gold contexts, target type)
    "S_txn": ({"PAYMENTS"}, "Transaction"),
    "S_loan": ({"LENDING"}, "Loan"),
    "S_ticket": ({"SUPPORT"}, "Ticket"),
    "S_alert": ({"RISK"}, "Alert"),
    "C_alert_txn": ({"RISK", "PAYMENTS"}, "Transaction"),
    "C_ticket_loan": ({"SUPPORT", "LENDING"}, "Loan"),
    "C_txn_ticket": ({"PAYMENTS", "SUPPORT"}, "Ticket"),
}
ROLES = {
    "fraud_analyst": {"CUSTOMER", "PAYMENTS", "RISK"},
    "servicing_agent": {"CUSTOMER", "PAYMENTS", "LENDING", "SUPPORT"},
    "collections": {"CUSTOMER", "LENDING", "PAYMENTS"},
    "dispute_specialist": {"CUSTOMER", "PAYMENTS", "SUPPORT", "RISK"},
    "risk_officer": {"CUSTOMER", "RISK", "LENDING", "PAYMENTS"},
}


def make_queries(parties, n, seed):
    rng = random.Random(seed + 1000)
    out = []
    names = list(TEMPLATES)
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        tpl = names[len(out) % len(names)]
        p = rng.choice(parties)
        nm, city = p["name"], p["city"]
        q = None
        form = rng.randrange(3)
        P = lambda opts: opts[form]
        if tpl == "S_txn" and p["txns"]:
            t, m, amt, day, l4 = rng.choice(p["txns"])
            q, tgt = P([f"Which transaction at {m} did {nm} from {city} make on {day}?",
                        f"What did {nm} in {city} spend at {m} on {day}?",
                        f"Show the {m} purchase by {nm} ({city}) dated {day}."]), t
        elif tpl == "S_loan" and p["loans"]:
            ln, kind, status = rng.choice(p["loans"])
            q, tgt = P([f"What is the {kind} loan held by {nm} in {city} that is {status}?",
                        f"Which {kind} financing does {nm} from {city} have, currently {status}?",
                        f"Pull up the {status} {kind} borrowing for {nm} in {city}."]), ln
        elif tpl == "S_ticket" and p["tickets"]:
            tk, topic, _ = rng.choice(p["tickets"])
            q, tgt = P([f"Find the support ticket {nm} from {city} opened about a {topic}.",
                        f"When {nm} in {city} called about a {topic}, what case was logged?",
                        f"Which complaint did {nm} ({city}) raise regarding a {topic}?"]), tk
        elif tpl == "S_alert" and p["alerts"]:
            al, sev, ak, _ = rng.choice(p["alerts"])
            q, tgt = P([f"Which {sev} alert for {ak} was raised on {nm} in {city}?",
                        f"Why was {nm} from {city} flagged for {ak} at {sev} severity?",
                        f"Show the {sev} {ak} flag on {nm} in {city}."]), al
        elif tpl == "C_alert_txn" and p["alerts"]:
            al, sev, ak, t = rng.choice(p["alerts"])
            q, tgt = P([f"Which payment transaction triggered the {sev} {ak} alert for {nm} in {city}?",
                        f"What purchase caused {nm} from {city} to be flagged for {ak} ({sev})?",
                        f"Find the charge behind the {sev} {ak} flag on {nm} in {city}."]), t[0]
        elif tpl == "C_ticket_loan":
            c = [x for x in p["tickets"] if x[2] and x[2][0] == "loan"]
            if c:
                tk, topic, ref = rng.choice(c)
                q, tgt = P([f"Which loan is the {topic} support ticket from {nm} in {city} about?",
                            f"{nm} from {city} complained about a {topic}; which borrowing was it about?",
                            f"What financing does the {topic} case for {nm} in {city} refer to?"]), ref[1][0]
        elif tpl == "C_txn_ticket":
            c = [x for x in p["tickets"] if x[2] and x[2][0] == "txn"]
            if c:
                tk, topic, ref = rng.choice(c)
                q, tgt = P([f"Which support ticket mentions the {ref[1][1]} transaction of {nm} from {city}?",
                            f"Did {nm} in {city} complain about the {ref[1][1]} purchase, and in which case?",
                            f"Find the case {nm} ({city}) opened about a {ref[1][1]} charge."]), tk
        if q is None:
            continue
        gold, ttype = TEMPLATES[tpl]
        roles = [r for r, ok in ROLES.items() if gold <= ok]
        role = rng.choice(roles)
        out.append({"qid": len(out), "template": tpl, "text": q, "target": tgt, "type": ttype,
                    "gold": sorted(gold), "role": role, "authorized": sorted(ROLES[role]),
                    "cross": tpl.startswith("C_"), "form": form})
    return out


# Requests the role is NOT allowed to answer. "target": the answer itself lives in a context the role
# cannot see. "evidence": the answer lives in an authorized context, but identifying it requires a record
# from an unauthorized context (e.g. which purchase was flagged, for a role without access to Risk).
RESTRICTED = {
    "S_loan": "target", "S_ticket": "target", "S_alert": "target",
    "C_alert_txn": "evidence", "C_ticket_loan": "evidence",
}


def make_restricted_queries(parties, n, seed, start_qid):
    base = make_queries(parties, n * 3, seed + 7)  # fresh draws, independent of the main query set
    out = []
    rng = random.Random(seed + 2000)
    for q in base:
        if q["template"] not in RESTRICTED or len(out) >= n:
            continue
        gold = set(q["gold"])
        tgt_ctx = {"Transaction": "PAYMENTS", "Loan": "LENDING", "Ticket": "SUPPORT", "Alert": "RISK"}[q["type"]]
        kind = RESTRICTED[q["template"]]
        if kind == "target":
            roles = [r for r, ok in ROLES.items() if tgt_ctx not in ok]
        else:
            roles = [r for r, ok in ROLES.items() if tgt_ctx in ok and not gold <= ok]
        if not roles:
            continue
        role = rng.choice(roles)
        q = dict(q, qid=start_qid + len(out), role=role, authorized=sorted(ROLES[role]), restricted=kind)
        out.append(q)
    return out
