# AICT_Assg

## Logical Inference Overview
- Resolution-based prover (see `inference.py`) checks MRT routing plans against propositional rules drawn from LTA's 25 Jul 2025 TELe/CRL change notice.
- Modes inject facts: Today Mode keeps the legacy East-West Line (EWL) airport spur active; Future Mode asserts TEL/CRL takeover of the Changi corridor.
- Advisory facts describe short-term operational notices (crowding, closures, mandatory availability, transfer limits).
- Route feature facts describe what the proposed journey needs (Tanah Merah ↔ Expo usage, T5 visit, CRL spur, etc.).

## Rule Catalog (excerpt)
Each clause is listed in propositional form (using $\lor$ for OR and $\neg$ for NOT) plus a plain-English meaning. Full set is defined in `build_rule_definitions()`.

| Rule | Clause | Interpretation |
| --- | --- | --- |
| R1 | $\neg TODAY\_MODE \lor EWL\_BRANCH\_ACTIVE$ | Today Mode keeps the EWL Tanah Merah–Expo–Changi Airport spur live. |
| R2 | $\neg TODAY\_MODE \lor \neg TEL\_CONVERSION$ | Today Mode implies TEL conversion work is not underway. |
| R3 | $\neg FUTURE\_MODE \lor TEL\_CONVERSION$ | Future Mode asserts that the Tanah Merah–Expo–Changi Airport segment is being converted to TEL systems. |
| R4 | $\neg FUTURE\_MODE \lor TEL\_EXTENSION$ | Future Mode activates the TEL extension towards Changi T5. |
| R5 | $\neg FUTURE\_MODE \lor CRL\_EXTENSION$ | Future Mode also activates the CRL spur reaching T5. |
| R8 | $\neg TEL\_CONVERSION \lor CONVERSION\_WORKS$ | TEL conversion implies systems-integration works are ongoing. |
| R9 | $\neg CONVERSION\_WORKS \lor SERVICE\_TM\_EXPO$ | Integration works trigger service adjustments between Tanah Merah and Expo. |
| R10 | $\neg TEL\_CONVERSION \lor \neg EWL\_BRANCH\_ACTIVE$ | TEL conversion suspends the legacy EWL airport branch during the switchover. |
| R14 | $\neg ROUTE\_VISIT\_T5 \lor OPEN\_T5$ | Any route that visits T5 must have the station open. |
| R15 | $\neg ROUTE\_NEEDS\_CRL\_T5 \lor DIRECT\_T5\_SERVICE$ | Routes relying on the CRL spur demand that service to T5 is running. |
| R16 | $\neg ADVISORY\_PAYA\_CROWD \lor \neg PAYA\_AVAILABLE$ | Crowding advisory at Paya Lebar removes it from routing consideration. |
| R19 | $\neg ROUTE\_TRANSFER\_HIGH \lor TRANSFER\_BUFFER$ | High-transfer routes consume any spare transfer buffer. |
| R20 | $\neg ADVISORY\_HARBOUR\_DOWN \lor \neg HARBOUR\_AVAILABLE$ | HarbourFront-closure advisories make that station unavailable. |
| R22 | $\neg ADVISORY\_KEEP\_HARBOUR \lor HARBOUR\_AVAILABLE$ | Operations insisting HarbourFront stays open explicitly keep it available. |

These rules explicitly encode the TELe/CRL takeover of the Changi corridor, its Tanah Merah–Expo–Changi Airport conversion impacts, and the mandated service adjustments during integration works.

## Scenario Evidence
`python inference.py` runs six mixed scenarios (valid, invalid, contradictory) across Today and Future modes. Results:

| Scenario | Mode | Route status | Advisory status | Violated rules |
| --- | --- | --- | --- | --- |
| S1 Today baseline | Today | Valid | Consistent | — |
| S2 TM-Expo shutdown | Today | Invalid | Consistent | R11, R12 |
| S3 Paya crowding | Today | Invalid | Consistent | R16, R17 |
| S4 Future T5 CRL | Future | Valid | Consistent | — |
| S5 Future TM-Expo attempt | Future | Invalid | Consistent | R3, R10, R13 |
| S6 HarbourFront conflict | Future | — | Inconsistent | R20, R22 |

- Invalid routes list the specific violated rules surfaced by the resolution trace, highlighting the operational constraint they broke.
- Scenario S6 has no route, demonstrating that contradicting advisories alone (close vs keep-open HarbourFront) make the knowledge base unsatisfiable.

## How It Works
- `build_kb()` injects the rule clauses plus mode, advisory, and route facts into the knowledge base.
- `ResolutionProver` performs pairwise clause resolution until either the empty clause appears (inconsistency) or no new clauses can be derived.
- `trace_rule_ids()` backtracks from the empty clause to enumerate which root rules participated in the proof of contradiction, satisfying the “identify violated rules” requirement.

## Challenges & Improvements
- **Changi corridor fidelity**: Balancing the level of detail for TEL vs CRL facts without modelling the entire network required careful scoping; future work could add separate symbols for Expo–Airport staging and for Sungei Bedok tie-ins.
- **Advisory granularity**: Current advisories are binary. Introducing severities (e.g., soft crowding vs absolute closure) would allow partial penalties instead of outright contradictions.
- **Inference performance**: Resolution is sufficient for this KB size, but a forward-chaining SAT solver with clause learning could scale better if more stations/advisories are added later.

## Running It Yourself
```
cd AICT_Assg
python inference.py
```
The script prints each scenario, route/advisory verdicts, and the rule IDs responsible for any inconsistencies.

## Passenger Re-Routing Optimization (Bonus Feature)
- `disruption_optimization.py` models peak-hour disruptions along the Changi Airport ↔ T5 corridor and applies local-search planners to minimize $\text{avg_delay} + \text{penalty}$.
- **Disruption settings (≥2):** (1) Tanah Merah–Expo track suspension, (2) Expo–Changi Airport reduced frequency, (3) peak-time transfer surge at Paya Lebar (node penalty + extra transfer cost).  These manifest as edge removals, slowed links, and time-window penalties.
- **Objective:** Minimize average delay vs the no-disruption baseline while respecting hard penalties for infeasible assignments.
- **Constraints (≥2):** (a) Max three transfers per journey ($>3$ triggers `TRANSFER_PENALTY`), (b) capacity caps on critical edges such as Sungei Bedok–Tanah Merah and Changi Airport–Expo (excess flow incurs `CAPACITY_PENALTY`).
- **AI techniques (≥2):** Deterministic hill climbing refines a naive plan; simulated annealing with geometric cooling explores higher-cost neighborhoods to escape local minima. Both operate over discrete state vectors.

### State, Neighborhood, Stopping Criteria
- **State representation:** Index vector where entry $i$ stores the selected route candidate for OD demand $i$ (generated via bounded-depth DFS on the disrupted graph).
- **Neighborhood move:** Swap a single OD demand to an alternative candidate route (reroute via TEL express link, etc.); hill climbing greedily accepts strictly better moves, while simulated annealing probabilistically accepts uphill moves.
- **Stopping:** Hill climbing halts once a full sweep finds no improvements; simulated annealing runs 30 inner moves per temperature step until $T<0.1$ with cooling factor 0.85.

### Baseline vs Optimized Plans (Future-mode disruptions)
| Plan | Objective | Avg delay | Penalty | Key effect |
| --- | --- | --- | --- | --- |
| Greedy reroute | 20.80 | 8.00 | 12.80 | All flows cling to Sungei Bedok → Tanah Merah, breaching the edge’s capacity cap and racking up penalties despite shorter runtimes. |
| Hill climbing | 10.39 | 10.39 | 0.00 | Reassigns both Changi Airport OD pairs onto TEL express links via Gardens by the Bay/HarbourFront, eliminating congestion penalties even though individual trips lengthen. |
| Simulated annealing | 10.39 | 10.39 | 0.00 | Confirms no better mix exists under current candidate set; matches hill climbing after stochastic exploration. |

Average delay is measured against the no-disruption baseline (`Passenger Re-Routing Baseline` block in the script output). The optimized plan doubles the greedy plan’s objective improvement by trading modest extra travel time for removing 12.8 penalty minutes tied to the Sungei Bedok bottleneck.

### Limitations & Future Work
- Edge capacities are stylized; integrating live crowd density data would let penalties scale dynamically instead of using fixed thresholds.
- Candidate enumeration currently uses depth-bounded DFS; incorporating k-shortest paths or CSP-based pruning would cover longer itineraries without combinatorial blow-up.
- Multi-objective handling (e.g., minimizing maximum delay while capping average delay) could be explored via Pareto-front search instead of scalarized objectives.

### Running the Optimization Study
```
cd AICT_Assg
python disruption_optimization.py
```
The script prints the baseline OD set, applied disruptions, and the objective breakdown for the greedy, hill-climbing, and simulated-annealing plans, including the concrete reroutes per OD demand.
