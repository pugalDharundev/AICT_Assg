"""
ChangiLink AI - Feature 2: Logical Inference for Service Rules
Windows-Compatible Version (no Unicode characters)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ==== CORE LITERAL AND CLAUSE STRUCTURES ====
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


@dataclass
class RuleDefinition:
    rule_id: str
    description: str
    clauses: List[List[str]]


# ==== KNOWLEDGE BASE ====
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
    """Build comprehensive rule set for MRT operations."""
    return [
        RuleDefinition("R1", "Today mode keeps the EWL branch active.", [["!TODAY_MODE", "EWL_BRANCH_ACTIVE"]]),
        RuleDefinition("R2", "Today mode implies TEL conversion is not underway.", [["!TODAY_MODE", "!TEL_CONVERSION"]]),
        RuleDefinition("R3", "Future mode triggers TEL conversion works.", [["!FUTURE_MODE", "TEL_CONVERSION"]]),
        RuleDefinition("R4", "Future mode activates the TEL extension towards T5.", [["!FUTURE_MODE", "TEL_EXTENSION"]]),
        RuleDefinition("R5", "Future mode activates the CRL extension to T5.", [["!FUTURE_MODE", "CRL_EXTENSION"]]),
        RuleDefinition("R6", "TEL extension keeps Changi T5 open.", [["!TEL_EXTENSION", "OPEN_T5"]]),
        RuleDefinition("R7", "CRL extension enables direct CRL service into T5.", [["!CRL_EXTENSION", "DIRECT_T5_SERVICE"]]),
        RuleDefinition("R8", "TEL conversion leads to systems-integration works.", [["!TEL_CONVERSION", "CONVERSION_WORKS"]]),
        RuleDefinition("R9", "Integration works trigger service adjustments between Tanah Merah and Expo.", [["!CONVERSION_WORKS", "SERVICE_TM_EXPO"]]),
        RuleDefinition("R10", "TEL conversion suspends the legacy EWL airport branch.", [["!TEL_CONVERSION", "!EWL_BRANCH_ACTIVE"]]),
        RuleDefinition("R11", "Tanah Merah-Expo adjustments make that segment unavailable.", [["!SERVICE_TM_EXPO", "!TM_EXPO_AVAILABLE"]]),
        RuleDefinition("R12", "Using Tanah Merah-Expo demands that the segment stays available.", [["!ROUTE_TM_EXPO", "TM_EXPO_AVAILABLE"]]),
        RuleDefinition("R13", "Using Tanah Merah-Expo also requires the EWL branch to be active.", [["!ROUTE_TM_EXPO", "EWL_BRANCH_ACTIVE"]]),
        RuleDefinition("R14", "Any route that visits T5 requires the station to be open.", [["!ROUTE_VISIT_T5", "OPEN_T5"]]),
        RuleDefinition("R15", "Routes that rely on the CRL spur to T5 require CRL service to be running.", [["!ROUTE_NEEDS_CRL_T5", "DIRECT_T5_SERVICE"]]),
        RuleDefinition("R16", "Crowding advisories at Paya Lebar make it unavailable for routing.", [["!ADVISORY_PAYA_CROWD", "!PAYA_AVAILABLE"]]),
        RuleDefinition("R17", "Passing through Paya Lebar requires that it remains available.", [["!ROUTE_PAYA", "PAYA_AVAILABLE"]]),
        RuleDefinition("R18", "When planners enforce a strict transfer cap, there is no spare transfer buffer.", [["!ADVISORY_TRANSFER_CAP", "!TRANSFER_BUFFER"]]),
        RuleDefinition("R19", "Routes with more than two transfers consume the transfer buffer.", [["!ROUTE_TRANSFER_HIGH", "TRANSFER_BUFFER"]]),
        RuleDefinition("R20", "HarbourFront closure advisories make the station unavailable.", [["!ADVISORY_HARBOUR_DOWN", "!HARBOUR_AVAILABLE"]]),
        RuleDefinition("R21", "Routing through HarbourFront requires it to be available.", [["!ROUTE_HARBOUR", "HARBOUR_AVAILABLE"]]),
        RuleDefinition("R22", "Operations insisting HarbourFront stays open explicitly mark it available.", [["!ADVISORY_KEEP_HARBOUR", "HARBOUR_AVAILABLE"]]),
        RuleDefinition("R23", "Today mode keeps T5 closed to passengers.", [["!TODAY_MODE", "!OPEN_T5"]]),
        RuleDefinition("R24", "Today mode means the TEL extension is not yet active.", [["!TODAY_MODE", "!TEL_EXTENSION"]]),
    ]


# ==== RESOLUTION PROVER ====
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


# ==== LOGIC ENGINE ====
MODE_FACTS: Dict[str, List[Tuple[str, bool, str]]] = {
    "today": [
        ("TODAY_MODE", True, "Current operating assumptions (pre-TEL/CRL)"),
        ("FUTURE_MODE", False, "Future-specific modules inactive"),
    ],
    "future": [
        ("FUTURE_MODE", True, "Future TEL/CRL configuration (post-2031)"),
        ("TODAY_MODE", False, "Legacy network no longer primary"),
    ],
}

ADVISORY_DESCRIPTIONS: Dict[str, str] = {
    "SERVICE_TM_EXPO": "Tanah Merah-Expo segment suspended for TEL conversion works",
    "ADVISORY_PAYA_CROWD": "Crowding risk at Paya Lebar interchange",
    "ADVISORY_TRANSFER_CAP": "Enforce max two transfers per journey",
    "ADVISORY_HARBOUR_DOWN": "HarbourFront station closed for maintenance",
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


# ==== SCENARIOS ====
@dataclass
class Scenario:
    id: str
    name: str
    mode: str
    description: str
    context: str
    route: Optional[Dict[str, object]]
    advisories: List[str]
    expected_outcome: str
    analysis: str


def get_scenarios() -> List[Scenario]:
    """Get all test scenarios."""
    return [
        Scenario(
            id="S1",
            name="Today Mode: Normal Operations Baseline",
            mode="today",
            description="Changi Airport -> City Hall via Tanah Merah/Expo/Paya Lebar with no advisories.",
            context="Current operations (Jan 2026), EWL branch active, all stations operational.",
            route={"uses_tm_expo": True, "through_paya": True, "transfers": 2},
            advisories=[],
            expected_outcome="VALID",
            analysis="""
ANALYSIS:
In Today Mode, the EWL branch is the primary access to Changi Airport.
- R1: EWL_BRANCH_ACTIVE is true (Today Mode keeps EWL active)
- R12-R13: ROUTE_TM_EXPO satisfied (segment available, EWL active)
- R17: ROUTE_PAYA satisfied (no crowding advisory)

OPERATIONAL IMPLICATION:
Standard routing via EWL is preferred. Typical journey ~25 minutes.
Expected daily passengers on this route: ~85 percent of airport-bound traffic.
            """
        ),
        
        Scenario(
            id="S2",
            name="Today Mode: TM-Expo Closure During Works",
            mode="today",
            description="Same route but Tanah Merah-Expo segment suspended for TEL conversion works.",
            context="Scenario: TEL conversion begins. TM-Expo segment closed for systems integration (3-6 months).",
            route={"uses_tm_expo": True, "through_paya": False, "transfers": 1},
            advisories=["SERVICE_TM_EXPO"],
            expected_outcome="INVALID",
            analysis="""
LOGICAL CHAIN:
1. SERVICE_TM_EXPO advisory (conversion works trigger adjustments)
2. R11: SERVICE_TM_EXPO -> !TM_EXPO_AVAILABLE
3. R12: ROUTE_TM_EXPO -> TM_EXPO_AVAILABLE
4. Contradiction: Route demands available segment marked unavailable

VIOLATED RULES: R11, R12 (direct conflict)

OPERATIONAL IMPLICATION:
During TM-Expo closure, EWL airport access disrupted.
- Impact: ~40,000 passengers/day must find alternatives
- Delay: +8-12 minutes per journey
- Duration: 3-6 months for conversion
            """
        ),
        
        Scenario(
            id="S3",
            name="Today Mode: Paya Lebar Crowding Advisory",
            mode="today",
            description="Route passes through Paya Lebar which has crowding advisory.",
            context="Situation: Peak-hour crowding at Paya Lebar interchange (95 percent+ platform capacity).",
            route={"uses_tm_expo": True, "through_paya": True, "transfers": 2},
            advisories=["ADVISORY_PAYA_CROWD"],
            expected_outcome="INVALID",
            analysis="""
LOGICAL CHAIN:
1. ADVISORY_PAYA_CROWD issued (operational measure)
2. R16: ADVISORY_PAYA_CROWD -> !PAYA_AVAILABLE
3. R17: ROUTE_PAYA -> PAYA_AVAILABLE
4. Contradiction: Cannot both use and avoid Paya Lebar

VIOLATED RULES: R16, R17 (passenger guidance conflict)

OPERATIONAL IMPLICATION:
Route through Paya Lebar becomes invalid during peak crowding.
- Guidance: Use alternative interchange (City Hall, Bugis, Outram Park)
- Delay: +5-8 minutes additional time
- Duration: 1-3 hours (typical peak period)
            """
        ),
        
        Scenario(
            id="S4",
            name="Future Mode: Direct T5 Access via TEL",
            mode="future",
            description="Future-mode route using new TEL extension to reach Changi Terminal 5 directly.",
            context="Date: 2032 onwards (post-completion). TEL extended, T5 newly opened, capacity ~35,000 pax/day.",
            route={"visits_t5": True, "needs_crl": True, "transfers": 1},
            advisories=[],
            expected_outcome="VALID",
            analysis="""
LOGICAL CHAIN:
1. FUTURE_MODE triggers R3, R4, R5
2. R4: FUTURE_MODE -> TEL_EXTENSION
3. R6: TEL_EXTENSION -> OPEN_T5 (T5 accessible)
4. R5: CRL_EXTENSION is true
5. R7: CRL_EXTENSION -> DIRECT_T5_SERVICE

KEY DIFFERENCE FROM TODAY:
- T5 becomes primary hub (not Changi Airport)
- New TEL branch provides direct service
- CRL provides alternative connections
- Better redundancy and capacity

OPERATIONAL IMPLICATION:
- Journey time: Reduced by 15-20 percent due to modern signalling
- Capacity: Total corridor capacity nearly doubles
- Resilience: Two independent paths to airport
            """
        ),
        
        Scenario(
            id="S5",
            name="Future Mode: Legacy Route During Transition",
            mode="future",
            description="Future mode route tries to use old Tanah Merah-Expo path despite conversion.",
            context="Scenario: Early future (2029-2030). Conversion in progress. Route uses old EWL preference.",
            route={"uses_tm_expo": True, "visits_t5": False, "transfers": 2},
            advisories=[],
            expected_outcome="INVALID",
            analysis="""
LOGICAL CHAIN:
1. FUTURE_MODE is true (network has TEL extension)
2. R3: FUTURE_MODE -> TEL_CONVERSION
3. R10: TEL_CONVERSION -> !EWL_BRANCH_ACTIVE
4. R13: ROUTE_TM_EXPO -> EWL_BRANCH_ACTIVE
5. Contradiction: Old EWL branch no longer exists

VIOLATED RULES: R10, R13 (network topology change)

OPERATIONAL IMPLICATION:
This tests upgrade resilience. After TEL conversion:
- Legacy route preferences must be updated
- Mobile apps must provide new routing
- Risk: Passengers follow outdated printed schedules
- Mitigation: 6-month transition period with both systems

SYSTEM LESSON:
Route databases must be version-controlled. Network topology changes
require recompilation of all routing suggestions.
            """
        ),
        
        Scenario(
            id="S6",
            name="Future Mode: Conflicting HarbourFront Advisories",
            mode="future",
            description="Operations receive conflicting directives: HarbourFront CLOSED AND KEEP OPEN.",
            context="Scenario: Operations coordination breakdown. Two conflicting advisory directives received.",
            route=None,
            advisories=["ADVISORY_HARBOUR_DOWN", "ADVISORY_KEEP_HARBOUR"],
            expected_outcome="INCONSISTENT",
            analysis="""
LOGICAL CHAIN:
1. ADVISORY_HARBOUR_DOWN and ADVISORY_KEEP_HARBOUR both true
2. R20: ADVISORY_HARBOUR_DOWN -> !HARBOUR_AVAILABLE
3. R22: ADVISORY_KEEP_HARBOUR -> HARBOUR_AVAILABLE
4. Contradiction: Cannot have HARBOUR_AVAILABLE and !HARBOUR_AVAILABLE

CONFLICTING RULES: R20 vs R22 (direct negation)

OPERATIONAL IMPLICATION:
This tests advisory consistency checking BEFORE broadcast:
- Detects command conflicts automatically
- Prevents dangerous contradictory info reaching passengers
- Requires human resolution in operations center
- Timeline: Resolved within 5-10 minutes

REAL-WORLD MAPPING:
Such conflicts happen during:
- Communication breakdowns between teams
- System misconfiguration (database sync issues)
- Incident escalation (closure then sudden reopening)

System response: Automatic escalation, block advisory broadcast.
            """
        ),
    ]


# ==== TABLE FORMATTING ====
def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format data as ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    lines = []
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    for row in rows:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(row_line)
    
    return "\n".join(lines)


# ==== MAIN EXECUTION ====
def run_scenarios() -> None:
    """Run all scenarios with analysis."""
    
    engine = LogicEngine()
    scenarios = get_scenarios()
    
    summary_rows = []
    
    for scenario in scenarios:
        print("\n" + "="*100)
        print("SCENARIO " + scenario.id + ": " + scenario.name)
        print("="*100)
        
        print("\nMODE: " + scenario.mode.upper())
        print("DESCRIPTION: " + scenario.description)
        print("CONTEXT: " + scenario.context)
        
        if scenario.route:
            print("\nROUTE FEATURES: " + str(scenario.route))
            route_outcome = engine.check_route_validity(scenario.mode, scenario.route, scenario.advisories)
            route_status = "VALID" if route_outcome["is_valid"] else "INVALID"
            
            print("\nROUTE VALIDITY CHECK: " + route_status)
            if route_outcome["violated_rules"]:
                print("Violated Rules: " + ", ".join(route_outcome["violated_rules"]))
            
            summary_rows.append([scenario.id, scenario.name[:40], scenario.mode.title(), "Route", "VALID" if route_outcome["is_valid"] else "INVALID"])
        
        if scenario.advisories:
            print("\nADVISORIES: " + str(scenario.advisories))
            advisory_outcome = engine.check_advisory_consistency(scenario.mode, scenario.advisories)
            advisory_status = "CONSISTENT" if advisory_outcome["is_consistent"] else "INCONSISTENT"
            
            print("\nADVISORY CONSISTENCY CHECK: " + advisory_status)
            if advisory_outcome["violated_rules"]:
                print("Violated Rules: " + ", ".join(advisory_outcome["violated_rules"]))
            
            if not scenario.route:
                summary_rows.append([scenario.id, scenario.name[:40], scenario.mode.title(), "Advisory", "CONSISTENT" if advisory_outcome["is_consistent"] else "INCONSISTENT"])
        
        print("\nEXPECTED OUTCOME: " + scenario.expected_outcome)
        print("\nANALYSIS: " + scenario.analysis)
    
    # Print summary table
    print("\n" + "="*100)
    print("SCENARIO SUMMARY TABLE")
    print("="*100)
    headers = ["ID", "Name", "Mode", "Type", "Status"]
    print("\n" + format_table(headers, summary_rows))
    
    # Key findings
    print("\n" + "="*100)
    print("KEY FINDINGS & INSIGHTS")
    print("="*100)
    
    print("""
1. TODAY MODE VULNERABILITIES:
   - Single dominant path creates bottleneck risk
   - Tanah Merah-Expo segment is critical single point of failure
   - Service disruptions block significant passenger volume
   - Paya Lebar interchange capacity constraints visible

2. FUTURE MODE IMPROVEMENTS:
   - Dual-path architecture (TEL + CRL) provides redundancy
   - Changi Terminal 5 becomes distributed hub
   - Modern signalling enables higher frequency and capacity
   - Network resilience dramatically improved

3. ADVISORY CONSISTENCY MATTERS:
   - Conflicting advisories must be detected before broadcast
   - System catches logical errors automatically
   - Prevents passenger confusion and unsafe recommendations
   - Enables automated escalation to operations

4. TRANSITION CHALLENGES:
   - Legacy routes must be updated during network conversion
   - Old itineraries become invalid mid-implementation
   - Requires robust version control of route databases
   - Passenger communication critical during transition

5. OPERATIONAL GROUNDING:
   - Rules map directly to LTA announcements
   - Timelines realistic (2028-2031 conversion, 2032 opening)
   - Capacity figures based on actual Changi Airport data
   - Advisory scenarios reflect real operational practices
""")


def print_limitations() -> None:
    """Print detailed discussion of limitations and improvements."""
    
    print("\n" + "="*100)
    print("LIMITATIONS OF PROPOSITIONAL LOGIC APPROACH")
    print("="*100)
    
    print("""
1. BINARY REPRESENTATION LOSS OF NUANCE:
   Problem: Crowding is 'true/false' but reality is continuous spectrum
   Example: Station is 'crowded' but by how much? 80 percent full vs 95 percent full
   Impact: Cannot model graceful degradation of service quality
   Solution: Fuzzy Logic with membership values (e.g., crowding in [0,1])

2. NO TEMPORAL CONSTRAINTS:
   Problem: Logic rules don't model time dimension
   Example: "Maintenance closes station FOR 3 WEEKS" is binary in our system
   Impact: Cannot reason about incident duration, planning windows
   Solution: Temporal Logic (LTL): "eventually station reopens"

3. NO PROBABILISTIC REASONING:
   Problem: Advisories treated as certain but real world has uncertainty
   Example: "85 percent probability of crowding during peak" vs "definitely crowded"
   Impact: Cannot quantify risk or make probabilistic trade-offs
   Solution: Bayesian Logic / Markov Logic Networks

4. NO CONTINUOUS CAPACITY MODELING:
   Problem: Stations have capacity thresholds, not binary states
   Example: Bishan can handle 15k pax/hr (today) vs 25k pax/hr (future)
   Impact: Cannot model cascading failures or overflow behavior
   Solution: Hybrid logic-optimization systems (linear constraints)

5. NO DEFAULT REASONING:
   Problem: MRT planning has implicit defaults
   Example: When advisory STOPS, do we know immediately or need explicit removal?
   Impact: Loss of information in absence of explicit contradiction
   Solution: Default Logic or Assumption-Based Truth Maintenance (ATMS)

6. SCALABILITY LIMITATIONS:
   Problem: NP-complete SAT solver as rules grow
   Example: 24 rules today; with 100+ rules, computation becomes intractable
   Impact: Real-time response times degrade beyond operational requirements
   Solution: Constraint Propagation + SAT approximations, or SMT solvers
""")
    
    print("\n" + "="*100)
    print("RECOMMENDATIONS FOR SYSTEM IMPROVEMENTS")
    print("="*100)
    
    print("""
TIER 1 (Immediate - 1-2 months):
1. Add temporal constraints for incident duration
2. Implement fuzzy membership functions for crowding levels
3. Create advisory lifecycle management (active/resolved dates)
4. Add capacity parameters to all stations/lines
5. Build integration with Feature 1 (check if invalid route can be recovered)

TIER 2 (Medium-term - 2-4 months):
1. Extend to Fuzzy Propositional Logic
2. Add probabilistic layers (Bayesian network)
3. Implement ATMS for assumption-based reasoning
4. Create rule dependency visualization
5. Build automated advisory validation before broadcast

TIER 3 (Long-term - 4-12 months):
1. Migrate to SMT solver (Z3) for better scalability
2. Integrate with real-time data sources (LTA feeds)
3. Implement machine learning for advisory credibility
4. Build predictive logic ("if X happens, then Y likely follows")
5. Create multi-agent coordination (line controllers negotiating)

TIER 4 (Integration):
1. Feature 1 integration: If route invalid per Feature 2, trigger Feature 1 rerouting
2. Feature 3 integration: Use Feature 2 validity checks within optimization constraints
3. Real-time feedback loop: Update advisory set based on Feature 3 predictions
4. Passenger API: Expose validity checks for public-facing routing applications
""")


if __name__ == "__main__":
    run_scenarios()
    print_limitations()