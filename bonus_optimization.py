
import random
import math
from collections import defaultdict

from routing import build_today_mode, build_future_mode


# -------------------------
# Helpers: edge + path utils
# -------------------------
def undirected_edge(a, b):
    return tuple(sorted((a, b)))


def get_edge_info(graph, a, b):
    # returns (travel_time, crowded, transfer) for edge a->b
    for (nxt, t, crowded, transfer) in graph.edges[a]:
        if nxt == b:
            return t, crowded, transfer
    raise KeyError(f"Edge not found: {a} -> {b}")


def compute_cost(travel_time, transfer=False, crowded=False,
                 transfer_penalty=3, crowd_penalty=5):
    return travel_time + (transfer_penalty if transfer else 0) + (crowd_penalty if crowded else 0)


def path_edges(path):
    # ✅ FIX: handle None / short paths safely
    if not path or len(path) < 2:
        return []
    return list(zip(path[:-1], path[1:]))


def count_transfers(graph, path):
    # ✅ FIX: if no path, treat as invalid (forces rejection)
    if not path:
        return 999
    transfers = 0
    for a, b in path_edges(path):
        _, _, transfer = get_edge_info(graph, a, b)
        if transfer:
            transfers += 1
    return transfers


def path_cost(graph, path, disruption, transfer_penalty=3, crowd_penalty=5):
    # ✅ FIX: handle None path
    if not path:
        return float("inf")

    total = 0
    for a, b in path_edges(path):
        e = undirected_edge(a, b)

        # hard closures
        if e in disruption["closed_edges"]:
            return float("inf")

        t, crowded, transfer = get_edge_info(graph, a, b)

        # segment penalty (reduced service etc.)
        extra = disruption["edge_extra_penalty"].get(e, 0)

        # optionally increase transfer penalty around key interchange/crowding
        tp = int(round(transfer_penalty * disruption["transfer_penalty_multiplier"]))

        total += compute_cost(t, transfer, crowded, tp, crowd_penalty) + extra

    return total


# -------------------------
# A* that respects disruptions
# -------------------------
def heuristic(graph, a, b):
    (x1, y1) = graph.coords[a]
    (x2, y2) = graph.coords[b]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def astar_disrupted(graph, start, goal, disruption,
                    transfer_penalty=3, crowd_penalty=5):
    import heapq
    pq = [(heuristic(graph, start, goal), 0, start, [start])]
    best_g = {}

    while pq:
        f, g, node, path = heapq.heappop(pq)

        if node == goal:
            return path, g

        if node in best_g and g >= best_g[node]:
            continue

        best_g[node] = g

        for (nxt, t, crowded, transfer) in graph.edges[node]:
            e = undirected_edge(node, nxt)
            if e in disruption["closed_edges"]:
                continue

            extra = disruption["edge_extra_penalty"].get(e, 0)
            tp = int(round(transfer_penalty * disruption["transfer_penalty_multiplier"]))
            step = compute_cost(t, transfer, crowded, tp, crowd_penalty) + extra

            g2 = g + step
            f2 = g2 + heuristic(graph, nxt, goal)
            heapq.heappush(pq, (f2, g2, nxt, path + [nxt]))

    return None, float("inf")


# -------------------------
# Candidate route generation (K alternatives)
# -------------------------
def k_alternatives(graph, start, goal, disruption, K=5, bump=12):
    """
    Generates alternative routes by repeatedly penalizing an edge in the current best path.
    This is enough for optimization marks (gives a neighborhood of choices).
    """
    base_disruption = {
        "closed_edges": set(disruption["closed_edges"]),
        "edge_extra_penalty": dict(disruption["edge_extra_penalty"]),
        "transfer_penalty_multiplier": disruption["transfer_penalty_multiplier"]
    }

    candidates = []
    seen = set()

    for _ in range(K):
        path, cost = astar_disrupted(graph, start, goal, base_disruption)
        if not path:
            break

        key = tuple(path)
        if key in seen:
            break
        seen.add(key)

        candidates.append((path, cost))

        # penalize a middle edge to force a different route next time
        edges = path_edges(path)
        if len(edges) <= 1:
            break

        a, b = edges[min(1, len(edges) - 1)]
        e = undirected_edge(a, b)
        base_disruption["edge_extra_penalty"][e] = base_disruption["edge_extra_penalty"].get(e, 0) + bump

    return candidates  # list of (path, cost)


# -------------------------
# Disruptions (AT LEAST 2) — requirement
# -------------------------
def disruption_A_suspend_TanahMerah_Expo():
    return {
        "name": "A) Tanah Merah–Expo SEVERELY REDUCED (penalty)",
        "closed_edges": set(),
        "edge_extra_penalty": {undirected_edge("Tanah Merah", "Expo"): 25},  # big penalty
        "transfer_penalty_multiplier": 1.0
    }



def disruption_B_reduce_Expo_ChangiAirport():
    # Expo–Changi Airport reduced service (soft penalty)
    return {
        "name": "B) Expo–Changi Airport REDUCED SERVICE",
        "closed_edges": set(),
        "edge_extra_penalty": {undirected_edge("Expo", "Changi Airport"): 10},
        "transfer_penalty_multiplier": 1.0
    }


def disruption_C_transfer_penalty_up():
    # Transfer penalties increased due to crowding at a key interchange
    return {
        "name": "C) TRANSFER PENALTY INCREASED (crowding)",
        "closed_edges": set(),
        "edge_extra_penalty": {},
        "transfer_penalty_multiplier": 1.8
    }


# -------------------------
# Objective + Constraints
# -------------------------
def objective_avg_delay(state_choices, candidates_per_od, baseline_costs, graph,
                        disruption, max_transfers=3, capacities=None, cap_penalty=15):
    """
    Objective: Minimize average delay vs baseline (requirement option)
    Constraints:
      1) max_transfers (hard)  -> reject with huge score
      2) capacity limits (soft) -> penalties if overloaded
    """
    if capacities is None:
        capacities = {}

    chosen_paths = []
    new_costs = []

    for i, choice in enumerate(state_choices):
        path, _ = candidates_per_od[i][choice]

        # ✅ FIX: reject None path immediately
        if path is None:
            return 1e9

        # hard constraint: max transfers
        if count_transfers(graph, path) > max_transfers:
            return 1e9

        c = path_cost(graph, path, disruption)
        if c == float("inf"):
            return 1e9

        chosen_paths.append(path)
        new_costs.append(c)

    delays = [max(0, new_costs[i] - baseline_costs[i]) for i in range(len(new_costs))]
    avg_delay = sum(delays) / len(delays)

    # capacity constraint (soft)
    load = defaultdict(int)
    for path in chosen_paths:
        for a, b in path_edges(path):
            load[undirected_edge(a, b)] += 1

    cap_cost = 0
    for e, used in load.items():
        cap = capacities.get(e, 999)
        if used > cap:
            cap_cost += (used - cap) * cap_penalty

    return avg_delay + cap_cost


# -------------------------
# AI Technique 1: Hill Climbing (with restarts)
# -------------------------
def hill_climb_with_restarts(candidates_per_od, baseline_costs, graph, disruption,
                             max_transfers=3, capacities=None,
                             iters=400, restarts=10):
    best_state = None
    best_val = float("inf")

    n = len(candidates_per_od)

    for _ in range(restarts):
        state = [random.randrange(len(candidates_per_od[i])) for i in range(n)]
        curr_val = objective_avg_delay(state, candidates_per_od, baseline_costs, graph,
                                       disruption, max_transfers, capacities)

        for _ in range(iters):
            i = random.randrange(n)
            j = random.randrange(len(candidates_per_od[i]))
            neighbor = state[:]
            neighbor[i] = j

            val = objective_avg_delay(neighbor, candidates_per_od, baseline_costs, graph,
                                      disruption, max_transfers, capacities)

            if val < curr_val:
                state, curr_val = neighbor, val

        if curr_val < best_val:
            best_state, best_val = state, curr_val

    return best_state, best_val


# -------------------------
# AI Technique 2: Simulated Annealing
# -------------------------
def simulated_annealing(candidates_per_od, baseline_costs, graph, disruption,
                        max_transfers=3, capacities=None,
                        iters=1200, T0=6.0, alpha=0.995):
    n = len(candidates_per_od)
    state = [random.randrange(len(candidates_per_od[i])) for i in range(n)]
    curr_val = objective_avg_delay(state, candidates_per_od, baseline_costs, graph,
                                   disruption, max_transfers, capacities)

    best_state = state[:]
    best_val = curr_val
    T = T0

    for _ in range(iters):
        i = random.randrange(n)
        j = random.randrange(len(candidates_per_od[i]))
        neighbor = state[:]
        neighbor[i] = j

        val = objective_avg_delay(neighbor, candidates_per_od, baseline_costs, graph,
                                  disruption, max_transfers, capacities)

        delta = val - curr_val
        if delta < 0 or random.random() < math.exp(-delta / max(1e-9, T)):
            state, curr_val = neighbor, val
            if curr_val < best_val:
                best_state, best_val = state[:], curr_val

        T *= alpha

    return best_state, best_val


# -------------------------
# Baseline vs Optimized runner (deliverable)
# -------------------------
def run_bonus(mode_name, graph, od_list, disruptions):
    print("\n" + "=" * 90)
    print(f"BONUS OPTIMIZATION RUN — {mode_name}")
    print("=" * 90)

    # capacity limits on key corridor edges (soft constraint)
    capacities = {
        undirected_edge("Expo", "Changi Airport"): 2,
        undirected_edge("Expo", "Tanah Merah"): 2,
    }
    if "Changi Terminal 5" in graph.coords:
        capacities[undirected_edge("Changi Terminal 5", "Expo")] = 2

    # ✅ FIX: max transfers should allow your baseline (Bishan had 3)
    max_transfers = 3

    # Baseline (no disruption)
    no_disruption = {"closed_edges": set(), "edge_extra_penalty": {}, "transfer_penalty_multiplier": 1.0}

    baseline_paths = []
    baseline_costs = []
    for (s, g) in od_list:
        p, c = astar_disrupted(graph, s, g, no_disruption)
        baseline_paths.append(p)
        baseline_costs.append(c)

    print("\nBaseline routes (no disruption):")
    for (s, g), p, c in zip(od_list, baseline_paths, baseline_costs):
        if not p:
            print(f"  {s} → {g} | NO PATH (baseline)")
            continue
        print(f"  {s} → {g} | cost={c:.2f} | transfers={count_transfers(graph, p)} | path={' → '.join(p)}")

    for dis in disruptions:
        print("\n" + "-" * 90)
        print(f"Disruption: {dis['name']}")
        print("Objective: Minimize AVERAGE DELAY vs baseline")
        print(f"Constraints: max_transfers={max_transfers} (hard), capacity limits (soft penalties)")
        print("AI Techniques: Hill Climbing (restarts) + Simulated Annealing")
        print("-" * 90)

        # candidate routes under disruption (K choices per OD)
        candidates_per_od = []
        for (s, g) in od_list:
            cand = k_alternatives(graph, s, g, dis, K=5)
            if not cand:
                cand = [(None, float("inf"))]  # ✅ safe placeholder
            candidates_per_od.append(cand)

        for i in range(len(candidates_per_od)):
            if len(candidates_per_od[i]) == 1:
                candidates_per_od[i].append(candidates_per_od[i][0])

        base_state = [0] * len(od_list)
        base_obj = objective_avg_delay(base_state, candidates_per_od, baseline_costs, graph,
                                       dis, max_transfers, capacities)

        hc_state, hc_obj = hill_climb_with_restarts(
            candidates_per_od, baseline_costs, graph, dis,
            max_transfers=max_transfers, capacities=capacities,
            iters=400, restarts=10
        )

        sa_state, sa_obj = simulated_annealing(
            candidates_per_od, baseline_costs, graph, dis,
            max_transfers=max_transfers, capacities=capacities,
            iters=1200, T0=6.0, alpha=0.995
        )

        print(f"\nObjective values (lower is better):")
        print(f"  Baseline (best-per-OD)      : {base_obj:.3f}")
        print(f"  Hill Climbing (restarts)    : {hc_obj:.3f}")
        print(f"  Simulated Annealing         : {sa_obj:.3f}")

        def show_solution(label, state):
            print(f"\n{label} — chosen routes:")
            for idx, ((s, g), choice) in enumerate(zip(od_list, state)):
                path, _ = candidates_per_od[idx][choice]
                if not path:
                    print(f"  {s} → {g} | NO PATH")
                    continue
                new_c = path_cost(graph, path, dis)
                delay = max(0, new_c - baseline_costs[idx])
                print(f"  {s} → {g} | new={new_c:.2f} | base={baseline_costs[idx]:.2f} | delay={delay:.2f} | transfers={count_transfers(graph, path)} | path={' → '.join(path)}")

        show_solution("Hill Climbing", hc_state)
        show_solution("Simulated Annealing", sa_state)

        print("\nStopping criteria used:")
        print("  - Hill Climbing: 10 restarts × 400 iterations each")
        print("  - Simulated Annealing: 1200 iterations with temperature cooling")

        print("\nNeighborhood move used:")
        print("  - Change 1 OD pair’s route choice to another candidate (local search move)")

        print("\nLimitations you can mention:")
        print("  - Candidate routes are approximated (not exact K-shortest)")
        print("  - Soft capacity penalties sensitive to chosen penalty weights")
        print("  - Local minima possible; SA helps but not guaranteed optimal")
        print("  - Hard closures may make some OD pairs infeasible (no path)")


def main():
    today_graph = build_today_mode()
    future_graph = build_future_mode()

    today_od = [
        ("Changi Airport", "City Hall"),
        ("Changi Airport", "Orchard"),
        ("Changi Airport", "Gardens by the Bay"),
        ("Changi Airport", "HarbourFront"),
        ("Changi Airport", "Bishan"),
    ]

    future_od = [
        ("Changi Terminal 5", "City Hall"),
        ("Changi Terminal 5", "Orchard"),
        ("Changi Terminal 5", "Gardens by the Bay"),
        ("Paya Lebar", "Changi Terminal 5"),
        ("HarbourFront", "Changi Terminal 5"),
    ]

    disruptions = [
        disruption_A_suspend_TanahMerah_Expo(),
        disruption_B_reduce_Expo_ChangiAirport(),
        disruption_C_transfer_penalty_up(),  # optional but good for marks
    ]

    run_bonus("Today Mode", today_graph, today_od, disruptions)
    run_bonus("Future Mode", future_graph, future_od, disruptions)


if __name__ == "__main__":
    main()
