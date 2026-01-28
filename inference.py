from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# =========================
# Core literal and clause structures
# =========================
@dataclass(frozen=True)
class Literal:
    symbol: str
    positive: bool = True

    def negate(self) -> "Literal":
        return Literal(self.symbol, not self.positive)

    def __str__(self) -> str:
        return self.symbol if self.positive else f"!{self.symbol}"


@dataclass
class Clause:
    id: int
    literals: frozenset[Literal]
    source: str
    label: str
    description: str
    parents: Optional[Tuple[int, int]] = None

    def __str__(self) -> str:
        pretty = " OR ".join(str(lit) for lit in sorted(self.literals, key=lambda l: l.symbol))
        return pretty or "EMPTY"


def make_literal(token: str) -> Literal:
    token = token.strip()
    if not token:
        raise ValueError("Empty literal token")
    positive = True
    if token.startswith("!"):
        positive = False
        token = token[1:]
    return Literal(token, positive)


# =========================
# Knowledge base + rule definitions
# =========================
@dataclass
class RuleDefinition:
    rule_id: str
    description: str
    clauses: List[List[str]]


class KnowledgeBase:
    def __init__(self) -> None:
        self._clauses: Dict[frozenset[Literal], Clause] = {}
        self._id_lookup: Dict[int, Clause] = {}
        self._next_id: int = 1

    def add_rule(self, rule: RuleDefinition) -> None:
        for clause_tokens in rule.clauses:
            literals = frozenset(make_literal(tok) for tok in clause_tokens)
            self._register_clause(literals, source="rule", label=rule.rule_id, description=rule.description)

    def add_fact(
        self,
        symbol: str,
        value: bool = True,
        *,
        source: str,
        label: str,
        description: str,
    ) -> Clause:
        literal = Literal(symbol, value)
        return self._register_clause(frozenset({literal}), source=source, label=label, description=description)

    def _register_clause(
        self,
        literals: frozenset[Literal],
        *,
        source: str,
        label: str,
        description: str,
        parents: Optional[Tuple[int, int]] = None,
    ) -> Clause:
        if literals in self._clauses:
            return self._clauses[literals]
        clause = Clause(self._next_id, literals, source, label, description, parents)
        self._clauses[literals] = clause
        self._id_lookup[clause.id] = clause
        self._next_id += 1
        return clause

    @property
    def clauses(self) -> List[Clause]:
        return list(self._clauses.values())

    @property
    def id_lookup(self) -> Dict[int, Clause]:
        return self._id_lookup


def build_rule_definitions() -> List[RuleDefinition]:
    return [
        RuleDefinition(
            "R1",
            "Today mode keeps the EWL branch active.",
            [["!TODAY_MODE", "EWL_BRANCH_ACTIVE"]],
        ),
        RuleDefinition(
            "R2",
            "Today mode implies TEL conversion is not underway.",
            [["!TODAY_MODE", "!TEL_CONVERSION"]],
        ),
        RuleDefinition(
            "R3",
            "Future mode triggers TEL conversion works.",
            [["!FUTURE_MODE", "TEL_CONVERSION"]],
        ),
        RuleDefinition(
            "R4",
            "Future mode activates the TEL extension towards T5.",
            [["!FUTURE_MODE", "TEL_EXTENSION"]],
        ),
        RuleDefinition(
            "R5",
            "Future mode activates the CRL extension to T5.",
            [["!FUTURE_MODE", "CRL_EXTENSION"]],
        ),
        RuleDefinition(
            "R6",
            "TEL extension keeps Changi T5 open.",
            [["!TEL_EXTENSION", "OPEN_T5"]],
        ),
        RuleDefinition(
            "R7",
            "CRL extension enables direct CRL service into T5.",
            [["!CRL_EXTENSION", "DIRECT_T5_SERVICE"]],
        ),
        RuleDefinition(
            "R8",
            "TEL conversion leads to systems-integration works.",
            [["!TEL_CONVERSION", "CONVERSION_WORKS"]],
        ),
        RuleDefinition(
            "R9",
            "Integration works trigger service adjustments between Tanah Merah and Expo.",
            [["!CONVERSION_WORKS", "SERVICE_TM_EXPO"]],
        ),
        RuleDefinition(
            "R10",
            "TEL conversion suspends the legacy EWL airport branch.",
            [["!TEL_CONVERSION", "!EWL_BRANCH_ACTIVE"]],
        ),
        RuleDefinition(
            "R11",
            "Tanah Merah-Expo adjustments make that segment unavailable.",
            [["!SERVICE_TM_EXPO", "!TM_EXPO_AVAILABLE"]],
        ),
        RuleDefinition(
            "R12",
            "Using Tanah Merah-Expo demands that the segment stays available.",
            [["!ROUTE_TM_EXPO", "TM_EXPO_AVAILABLE"]],
        ),
        RuleDefinition(
            "R13",
            "Using Tanah Merah-Expo also requires the EWL branch to be active.",
            [["!ROUTE_TM_EXPO", "EWL_BRANCH_ACTIVE"]],
        ),
        RuleDefinition(
            "R14",
            "Any route that visits T5 requires the station to be open.",
            [["!ROUTE_VISIT_T5", "OPEN_T5"]],
        ),
        RuleDefinition(
            "R15",
            "Routes that rely on the CRL spur to T5 require CRL service to be running.",
            [["!ROUTE_NEEDS_CRL_T5", "DIRECT_T5_SERVICE"]],
        ),
        RuleDefinition(
            "R16",
            "Crowding advisories at Paya Lebar make it unavailable for routing.",
            [["!ADVISORY_PAYA_CROWD", "!PAYA_AVAILABLE"]],
        ),
        RuleDefinition(
            "R17",
            "Passing through Paya Lebar requires that it remains available.",
            [["!ROUTE_PAYA", "PAYA_AVAILABLE"]],
        ),
        RuleDefinition(
            "R18",
            "When planners enforce a strict transfer cap, there is no spare transfer buffer.",
            [["!ADVISORY_TRANSFER_CAP", "!TRANSFER_BUFFER"]],
        ),
        RuleDefinition(
            "R19",
            "Routes with more than two transfers consume the transfer buffer.",
            [["!ROUTE_TRANSFER_HIGH", "TRANSFER_BUFFER"]],
        ),
        RuleDefinition(
            "R20",
            "HarbourFront closure advisories make the station unavailable.",
            [["!ADVISORY_HARBOUR_DOWN", "!HARBOUR_AVAILABLE"]],
        ),
        RuleDefinition(
            "R21",
            "Routing through HarbourFront requires it to be available.",
            [["!ROUTE_HARBOUR", "HARBOUR_AVAILABLE"]],
        ),
        RuleDefinition(
            "R22",
            "Operations insisting HarbourFront stays open explicitly mark it available.",
            [["!ADVISORY_KEEP_HARBOUR", "HARBOUR_AVAILABLE"]],
        ),
        RuleDefinition(
            "R23",
            "Today mode keeps T5 closed to passengers.",
            [["!TODAY_MODE", "!OPEN_T5"]],
        ),
        RuleDefinition(
            "R24",
            "Today mode means the TEL extension is not yet active.",
            [["!TODAY_MODE", "!TEL_EXTENSION"]],
        ),
    ]


# =========================
# Resolution prover
# =========================
@dataclass
class ResolutionResult:
    is_consistent: bool
    empty_clause: Optional[Clause]
    clause_lookup: Dict[int, Clause]
    hit_limit: bool = False

    def violated_rules(self) -> Set[str]:
        if self.is_consistent or not self.empty_clause:
            return set()
        return trace_rule_ids(self.empty_clause, self.clause_lookup)


def trace_rule_ids(clause: Clause, lookup: Dict[int, Clause]) -> Set[str]:
    rules: Set[str] = set()
    stack = [clause]
    visited: Set[int] = set()
    while stack:
        current = stack.pop()
        if current.id in visited:
            continue
        visited.add(current.id)
        if current.source == "rule":
            rules.add(current.label)
        if current.parents:
            for parent_id in current.parents:
                parent = lookup.get(parent_id)
                if parent:
                    stack.append(parent)
    return rules


class ResolutionProver:
    def __init__(self, clauses: Sequence[Clause]) -> None:
        self._clauses: List[Clause] = list(clauses)
        self._clause_map: Dict[frozenset[Literal], Clause] = {cl.literals: cl for cl in clauses}
        self._id_lookup: Dict[int, Clause] = {cl.id: cl for cl in clauses}
        self._next_id: int = (max(self._id_lookup.keys()) + 1) if self._id_lookup else 1

    def check(self, max_iterations: int = 2000) -> ResolutionResult:
        processed_pairs: Set[Tuple[int, int]] = set()
        steps = 0
        while steps <= max_iterations:
            new_clause_added = False
            current_size = len(self._clauses)
            for i in range(current_size):
                for j in range(i + 1, current_size):
                    ci = self._clauses[i]
                    cj = self._clauses[j]
                    pair_key = (min(ci.id, cj.id), max(ci.id, cj.id))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)
                    resolvents = resolve(ci, cj)
                    for res_literals in resolvents:
                        steps += 1
                        if steps > max_iterations:
                            return ResolutionResult(True, None, self._id_lookup, hit_limit=True)
                        if not res_literals:
                            empty_clause = self._register_derivation(frozenset(), ci, cj)
                            return ResolutionResult(False, empty_clause, self._id_lookup)
                        if res_literals not in self._clause_map:
                            new_clause = self._register_derivation(res_literals, ci, cj)
                            self._clauses.append(new_clause)
                            new_clause_added = True
            if not new_clause_added:
                return ResolutionResult(True, None, self._id_lookup)
        return ResolutionResult(True, None, self._id_lookup, hit_limit=True)

    def _register_derivation(self, literals: frozenset[Literal], left: Clause, right: Clause) -> Clause:
        clause = Clause(
            self._next_id,
            literals,
            source="derived",
            label="RES",
            description="Derived via resolution",
            parents=(left.id, right.id),
        )
        self._next_id += 1
        self._clause_map[literals] = clause
        self._id_lookup[clause.id] = clause
        return clause


def resolve(left: Clause, right: Clause) -> List[frozenset[Literal]]:
    resolvents: List[frozenset[Literal]] = []
    for literal in left.literals:
        complement = literal.negate()
        if complement not in right.literals:
            continue
        new_literals = set(left.literals).union(right.literals)
        new_literals.discard(literal)
        new_literals.discard(complement)
        if is_tautology(new_literals):
            continue
        resolvents.append(frozenset(new_literals))
    return resolvents


def is_tautology(literals: Iterable[Literal]) -> bool:
    literal_set = set(literals)
    for literal in literal_set:
        if literal.negate() in literal_set:
            return True
    return False


# =========================
# Logic engine utilities
# =========================
MODE_FACTS: Dict[str, List[Tuple[str, bool, str]]] = {
    "today": [
        ("TODAY_MODE", True, "Current operating assumptions"),
        ("FUTURE_MODE", False, "Future-specific modules inactive"),
    ],
    "future": [
        ("FUTURE_MODE", True, "Future TELe/CRL configuration"),
        ("TODAY_MODE", False, "Not the current network"),
    ],
}

ADVISORY_DESCRIPTIONS: Dict[str, str] = {
    "SERVICE_TM_EXPO": "Manual notice that Tanah Merah-Expo services are suspended",
    "ADVISORY_PAYA_CROWD": "Crowding risk at Paya Lebar",
    "ADVISORY_TRANSFER_CAP": "Enforce max two transfers",
    "ADVISORY_HARBOUR_DOWN": "HarbourFront closed for works",
    "ADVISORY_KEEP_HARBOUR": "Operations require HarbourFront to stay available",
}


def route_feature_facts(route: Optional[Dict[str, object]]) -> List[Tuple[str, bool, str]]:
    if not route:
        return []
    facts: List[Tuple[str, bool, str]] = []
    if route.get("uses_tm_expo"):
        facts.append(("ROUTE_TM_EXPO", True, "Route uses Tanah Merah <-> Expo"))
    if route.get("visits_t5"):
        facts.append(("ROUTE_VISIT_T5", True, "Route stops at Changi T5"))
    if route.get("needs_crl"):
        facts.append(("ROUTE_NEEDS_CRL_T5", True, "Route depends on CRL spur to T5"))
    if route.get("through_paya"):
        facts.append(("ROUTE_PAYA", True, "Route passes through Paya Lebar"))
    if route.get("through_harbour"):
        facts.append(("ROUTE_HARBOUR", True, "Route passes through HarbourFront"))
    transfers = route.get("transfers")
    if isinstance(transfers, int) and transfers >= 3:
        facts.append(("ROUTE_TRANSFER_HIGH", True, f"Route needs {transfers} transfers"))
    return facts


class LogicEngine:
    def __init__(self) -> None:
        self._rules = build_rule_definitions()

    def build_kb(self, mode: str, advisories: Optional[Sequence[str]], route: Optional[Dict[str, object]]) -> KnowledgeBase:
        kb = KnowledgeBase()
        for rule in self._rules:
            kb.add_rule(rule)
        normalized_mode = mode.lower()
        if normalized_mode not in MODE_FACTS:
            raise ValueError(f"Unsupported mode: {mode}")
        for symbol, value, desc in MODE_FACTS[normalized_mode]:
            kb.add_fact(symbol, value, source="mode", label=f"MODE_{symbol}", description=desc)
        if advisories:
            for adv in advisories:
                desc = ADVISORY_DESCRIPTIONS.get(adv, "Operational advisory")
                kb.add_fact(adv, True, source="advisory", label=adv, description=desc)
        for symbol, value, desc in route_feature_facts(route):
            kb.add_fact(symbol, value, source="route", label=symbol, description=desc)
        return kb

    def check_consistency(
        self,
        mode: str,
        *,
        advisories: Optional[Sequence[str]] = None,
        route: Optional[Dict[str, object]] = None,
        max_iterations: int = 2000,
    ) -> ResolutionResult:
        kb = self.build_kb(mode, advisories, route)
        prover = ResolutionProver(kb.clauses)
        return prover.check(max_iterations=max_iterations)

    def check_route_validity(
        self,
        mode: str,
        route: Dict[str, object],
        advisories: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        result = self.check_consistency(mode, advisories=advisories, route=route)
        return {
            "is_valid": result.is_consistent,
            "violated_rules": sorted(result.violated_rules()),
            "hit_limit": result.hit_limit,
        }

    def check_advisory_consistency(
        self,
        mode: str,
        advisories: Sequence[str],
    ) -> Dict[str, object]:
        result = self.check_consistency(mode, advisories=advisories, route=None)
        return {
            "is_consistent": result.is_consistent,
            "violated_rules": sorted(result.violated_rules()),
            "hit_limit": result.hit_limit,
        }


# =========================
# Scenario harness
# =========================
@dataclass
class Scenario:
    name: str
    mode: str
    description: str
    route: Optional[Dict[str, object]]
    advisories: List[str]


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(
            "S1: Today baseline",
            "today",
            "Changi Airport -> City Hall via Tanah Merah / Expo / Paya Lebar with no advisories.",
            {"uses_tm_expo": True, "through_paya": True, "transfers": 2},
            [],
        ),
        Scenario(
            "S2: Today with TM-Expo shutdown",
            "today",
            "Planned works advisory closes Tanah Merah <-> Expo while route still uses it.",
            {"uses_tm_expo": True, "through_paya": False, "transfers": 1},
            ["SERVICE_TM_EXPO"],
        ),
        Scenario(
            "S3: Paya Lebar crowding",
            "today",
            "Crowding advisory asks riders to avoid Paya Lebar, but route insists on passing through.",
            {"uses_tm_expo": True, "through_paya": True, "transfers": 2},
            ["ADVISORY_PAYA_CROWD"],
        ),
        Scenario(
            "S4: Future T5 direct access",
            "future",
            "Future-mode route rides TEL -> CRL to reach T5 without using Tanah Merah <-> Expo segment.",
            {"visits_t5": True, "needs_crl": True, "transfers": 1},
            [],
        ),
        Scenario(
            "S5: Future route still using Tanah Merah <-> Expo",
            "future",
            "Route ignores TEL conversion and tries to run through Tanah Merah <-> Expo during works.",
            {"uses_tm_expo": True, "visits_t5": False, "transfers": 2},
            [],
        ),
        Scenario(
            "S6: Conflicting HarbourFront advisories",
            "future",
            "Operations simultaneously close HarbourFront and demand it stay open, creating contradiction.",
            None,
            ["ADVISORY_HARBOUR_DOWN", "ADVISORY_KEEP_HARBOUR"],
        ),
    ]


def run_scenarios() -> None:
    engine = LogicEngine()
    scenarios = default_scenarios()
    for scenario in scenarios:
        print("\n==============================")
        print(scenario.name)
        print("==============================")
        print(f"Mode: {scenario.mode.title()}")
        print(f"Description: {scenario.description}")
        print(f"Advisories: {scenario.advisories or ['(none)']}")
        if scenario.route:
            print(f"Route features: {scenario.route}")
            route_outcome = engine.check_route_validity(scenario.mode, scenario.route, scenario.advisories)
            if route_outcome["hit_limit"]:
                print(" Route check: inconclusive (iteration limit reached)")
            else:
                status = "VALID" if route_outcome["is_valid"] else "INVALID"
                print(f" Route check: {status}")
                if route_outcome["violated_rules"]:
                    print(f"  Violated rules: {route_outcome['violated_rules']}")
        else:
            print("Route features: (none)")
        if scenario.advisories:
            advisory_outcome = engine.check_advisory_consistency(scenario.mode, scenario.advisories)
            if advisory_outcome["hit_limit"]:
                print(" Advisory set: inconclusive (iteration limit reached)")
            else:
                status = "CONSISTENT" if advisory_outcome["is_consistent"] else "INCONSISTENT"
                print(f" Advisory set: {status}")
                if advisory_outcome["violated_rules"]:
                    print(f"  Violated rules: {advisory_outcome['violated_rules']}")
        else:
            print("Advisory set: CONSISTENT (no advisories)")


if __name__ == "__main__":
    run_scenarios()
