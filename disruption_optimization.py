from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math

from routing import MRTGraph, build_future_mode, compute_cost


@dataclass
class RouteCandidate:
    path: List[str]
    cost: float
    transfers: int
    edges: List[Tuple[str, str]]


@dataclass
class ODDemand:
    origin: str
    destination: str
    volume: float
    label: str


@dataclass
class DisruptionSetting:
    name: str
    description: str
    apply_graph: Callable[[MRTGraph], None]
    transfer_penalty_boost: float = 0.0
    node_penalties: Dict[str, float] = field(default_factory=dict)


MAX_TRANSFERS = 3
TRANSFER_PENALTY = 6.0
CAPACITY_PENALTY = 8.0
MISSING_ROUTE_PENALTY = 20.0
CAPACITY_LIMITS: Dict[Tuple[str, str], float] = {
    ("Changi Airport", "Expo"): 3.0,
    ("Expo", "Tanah Merah"): 2.0,
    ("Sungei Bedok", "Tanah Merah"): 2.0,
}


def edge_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def clone_graph(graph: MRTGraph) -> MRTGraph:
    cloned = MRTGraph()
    cloned.coords = dict(graph.coords)
    cloned.edges = defaultdict(list)
    for node, neighbors in graph.edges.items():
        cloned.edges[node] = list(neighbors)
    return cloned


def remove_connection(graph: MRTGraph, a: str, b: str) -> None:
    graph.edges[a] = [(dst, t, c, tr) for dst, t, c, tr in graph.edges[a] if dst != b]
    graph.edges[b] = [(dst, t, c, tr) for dst, t, c, tr in graph.edges[b] if dst != a]


def update_connection(
    graph: MRTGraph,
    a: str,
    b: str,
    *,
    travel_time: Optional[float] = None,
    crowded: Optional[bool] = None,
    transfer: Optional[bool] = None,
) -> None:
    graph.edges[a] = [
        (
            dst,
            travel_time if (dst == b and travel_time is not None) else t,
            crowded if (dst == b and crowded is not None) else c,
            transfer if (dst == b and transfer is not None) else tr,
        )
        for dst, t, c, tr in graph.edges[a]
    ]
    graph.edges[b] = [
        (
            dst,
            travel_time if (dst == a and travel_time is not None) else t,
            crowded if (dst == a and crowded is not None) else c,
            transfer if (dst == a and transfer is not None) else tr,
        )
        for dst, t, c, tr in graph.edges[b]
    ]


def enumerate_route_candidates(
    graph: MRTGraph,
    start: str,
    goal: str,
    *,
    max_depth: int = 7,
    top_k: int = 5,
    transfer_penalty_extra: float = 0.0,
    node_penalties: Optional[Dict[str, float]] = None,
) -> List[RouteCandidate]:
    node_penalties = node_penalties or {}
    routes: List[RouteCandidate] = []
    seen: set[Tuple[str, ...]] = set()

    def dfs(node: str, path: List[str], cost: float, transfers: int, edges_used: List[Tuple[str, str]]):
        if len(path) - 1 > max_depth:
            return
        if node == goal and len(path) > 1:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                routes.append(RouteCandidate(list(path), cost, transfers, list(edges_used)))
            return
        for nxt, travel_time, crowded, transfer in graph.edges[node]:
            if nxt in path:
                continue
            step_cost = compute_cost(travel_time, transfer, crowded)
            if transfer and transfer_penalty_extra:
                step_cost += transfer_penalty_extra
            step_cost += node_penalties.get(nxt, 0.0)
            path.append(nxt)
            edges_used.append(edge_key(node, nxt))
            dfs(nxt, path, cost + step_cost, transfers + (1 if transfer else 0), edges_used)
            path.pop()
            edges_used.pop()

    dfs(start, [start], 0.0, 0, [])
    routes.sort(key=lambda r: r.cost)
    return routes[:top_k]


def dijkstra_shortest_path(graph: MRTGraph, start: str, goal: str) -> Optional[RouteCandidate]:
    import heapq

    pq: List[Tuple[float, str, List[str], int, List[Tuple[str, str]]]] = [(0.0, start, [start], 0, [])]
    best_cost: Dict[str, float] = {}
    while pq:
        cost, node, path, transfers, edges_used = heapq.heappop(pq)
        if node == goal:
            return RouteCandidate(path, cost, transfers, edges_used)
        if cost > best_cost.get(node, float("inf")):
            continue
        best_cost[node] = cost
        for nxt, travel_time, crowded, transfer in graph.edges[node]:
            edge_cost = compute_cost(travel_time, transfer, crowded)
            new_path = path + [nxt]
            new_edges = edges_used + [edge_key(node, nxt)]
            heapq.heappush(
                pq,
                (
                    cost + edge_cost,
                    nxt,
                    new_path,
                    transfers + (1 if transfer else 0),
                    new_edges,
                ),
            )
    return None


def generate_candidate_sets(
    graph: MRTGraph,
    demands: Sequence[ODDemand],
    *,
    transfer_penalty_extra: float = 0.0,
    node_penalties: Optional[Dict[str, float]] = None,
) -> List[List[RouteCandidate]]:
    return [
        enumerate_route_candidates(
            graph,
            demand.origin,
            demand.destination,
            transfer_penalty_extra=transfer_penalty_extra,
            node_penalties=node_penalties,
        )
        for demand in demands
    ]


def make_plan_evaluator(candidate_routes: List[List[RouteCandidate]], demands: Sequence[ODDemand], baseline_costs: Sequence[float]):
    total_volume = sum(d.volume for d in demands)

    def evaluate(state: List[int], detail: bool = False):
        total_delay = 0.0
        penalty = 0.0
        edge_usage: Dict[Tuple[str, str], float] = defaultdict(float)
        for idx, demand in enumerate(demands):
            routes = candidate_routes[idx]
            volume = demand.volume
            choice_idx = state[idx] if idx < len(state) else -1
            if not routes or choice_idx < 0 or choice_idx >= len(routes):
                penalty += MISSING_ROUTE_PENALTY * volume
                continue
            route = routes[choice_idx]
            if route.transfers > MAX_TRANSFERS:
                penalty += TRANSFER_PENALTY * (route.transfers - MAX_TRANSFERS) * volume
            delay = max(0.0, route.cost - baseline_costs[idx])
            total_delay += delay * volume
            for edge in route.edges:
                edge_usage[edge] += volume
        for edge, usage in edge_usage.items():
            limit = CAPACITY_LIMITS.get(edge)
            if limit and usage > limit:
                penalty += CAPACITY_PENALTY * (usage - limit)
        avg_delay = total_delay / total_volume if total_volume else 0.0
        objective = avg_delay + penalty
        if detail:
            return {
                "objective": objective,
                "avg_delay": avg_delay,
                "penalty": penalty,
                "edge_usage": dict(edge_usage),
            }
        return objective

    return evaluate


def hill_climb_optimize(initial_state: List[int], evaluator, candidate_routes: List[List[RouteCandidate]]):
    current_state = initial_state[:]
    current_score = evaluator(current_state)
    improved = True
    while improved:
        improved = False
        for idx, routes in enumerate(candidate_routes):
            if not routes:
                continue
            for option_idx in range(len(routes)):
                if option_idx == current_state[idx]:
                    continue
                new_state = current_state[:]
                new_state[idx] = option_idx
                new_score = evaluator(new_state)
                if new_score < current_score:
                    current_state = new_state
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    return current_state, current_score


def simulated_annealing_optimize(start_state: List[int], evaluator, candidate_routes: List[List[RouteCandidate]]):
    current_state = start_state[:]
    current_score = evaluator(current_state)
    best_state = current_state[:]
    best_score = current_score
    temperature = 5.0
    cooling = 0.85
    iterations = 30
    valid_indices = [idx for idx, routes in enumerate(candidate_routes) if routes]
    if not valid_indices:
        return current_state, current_score
    while temperature > 0.1:
        for _ in range(iterations):
            idx = random.choice(valid_indices)
            routes = candidate_routes[idx]
            option_idx = random.randrange(len(routes))
            new_state = current_state[:]
            new_state[idx] = option_idx
            new_score = evaluator(new_state)
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_state = new_state
                current_score = new_score
                if new_score < best_score:
                    best_state = new_state[:]
                    best_score = new_score
        temperature *= cooling
    return best_state, best_score


def describe_plan(state: List[int], demands: Sequence[ODDemand], candidate_routes: List[List[RouteCandidate]], baseline_costs: Sequence[float]):
    details = []
    for idx, demand in enumerate(demands):
        routes = candidate_routes[idx]
        if not routes:
            details.append((demand.label, "(no feasible route)", None, None))
            continue
        choice_idx = state[idx]
        if choice_idx < 0 or choice_idx >= len(routes):
            details.append((demand.label, "(unassigned)", None, None))
            continue
        route = routes[choice_idx]
        delay = max(0.0, route.cost - baseline_costs[idx])
        details.append((demand.label, " -> ".join(route.path), route.cost, delay))
    return details


def print_plan_summary(title: str, state: List[int], evaluator, demands: Sequence[ODDemand], candidate_routes: List[List[RouteCandidate]], baseline_costs: Sequence[float]):
    detail = evaluator(state, detail=True)
    print(f"\n{title}")
    print(f" Objective: {detail['objective']:.2f} | Avg delay: {detail['avg_delay']:.2f} | Penalty: {detail['penalty']:.2f}")
    for label, path_desc, cost, delay in describe_plan(state, demands, candidate_routes, baseline_costs):
        if cost is None:
            print(f"  {label}: {path_desc}")
        else:
            print(f"  {label}: {path_desc} (cost {cost:.1f}, delay +{delay:.1f})")


def apply_disruptions(base_graph: MRTGraph, disruptions: Sequence[DisruptionSetting]):
    disrupted_graph = clone_graph(base_graph)
    transfer_boost = 0.0
    node_penalty_map: Dict[str, float] = defaultdict(float)
    for disruption in disruptions:
        disruption.apply_graph(disrupted_graph)
        transfer_boost += disruption.transfer_penalty_boost
        for node, penalty in disruption.node_penalties.items():
            node_penalty_map[node] += penalty
    return disrupted_graph, transfer_boost, dict(node_penalty_map)


def build_disruption_scenarios() -> List[DisruptionSetting]:
    return [
        DisruptionSetting(
            "Tanah Merah–Expo suspension",
            "Segment closed for conversion works",
            lambda g: remove_connection(g, "Tanah Merah", "Expo"),
        ),
        DisruptionSetting(
            "Expo–Changi Airport reduced frequency",
            "Travel time doubled to reflect half-speed shuttle",
            lambda g: update_connection(g, "Expo", "Changi Airport", travel_time=10, crowded=True),
        ),
        DisruptionSetting(
            "Paya Lebar transfer crowding",
            "Extra penalty applied when passing through Paya Lebar",
            lambda g: None,
            transfer_penalty_boost=2.0,
            node_penalties={"Paya Lebar": 2.0},
        ),
    ]


def run_disruption_planner():
    random.seed(42)
    base_graph = build_future_mode()
    base_graph.add_connection("Changi Terminal 5", "Gardens by the Bay", 30, crowded=True, transfer=True)
    base_graph.add_connection("Changi Terminal 5", "HarbourFront", 34, crowded=True, transfer=True)
    demands = [
        ODDemand("Changi Airport", "City Hall", 1.0, "CA → City Hall"),
        ODDemand("Changi Airport", "HarbourFront", 0.8, "CA → HarbourFront"),
        ODDemand("Changi Terminal 5", "Orchard", 1.2, "T5 → Orchard"),
        ODDemand("Changi Terminal 5", "Bishan", 0.6, "T5 → Bishan"),
    ]

    baseline_routes = [dijkstra_shortest_path(base_graph, d.origin, d.destination) for d in demands]
    baseline_costs = [route.cost if route else float("inf") for route in baseline_routes]

    print("\n------------------------------")
    print("Passenger Re-Routing Baseline (no disruption)")
    for demand, route in zip(demands, baseline_routes):
        if route:
            print(f" {demand.label}: {' -> '.join(route.path)} (cost {route.cost:.1f})")
        else:
            print(f" {demand.label}: no feasible path on baseline graph")

    disruptions = build_disruption_scenarios()
    disrupted_graph, transfer_boost, node_penalties = apply_disruptions(base_graph, disruptions)

    print("\nApplied disruptions:")
    for disruption in disruptions:
        print(f" - {disruption.name}: {disruption.description}")

    candidate_routes = generate_candidate_sets(
        disrupted_graph,
        demands,
        transfer_penalty_extra=transfer_boost,
        node_penalties=node_penalties,
    )
    initial_state = [0 if routes else -1 for routes in candidate_routes]
    evaluator = make_plan_evaluator(candidate_routes, demands, baseline_costs)

    print_plan_summary(
        "Greedy per-OD reroute (baseline under disruption)",
        initial_state,
        evaluator,
        demands,
        candidate_routes,
        baseline_costs,
    )

    hill_state, _ = hill_climb_optimize(initial_state, evaluator, candidate_routes)
    print_plan_summary(
        "Hill climbing refinement",
        hill_state,
        evaluator,
        demands,
        candidate_routes,
        baseline_costs,
    )

    anneal_state, _ = simulated_annealing_optimize(hill_state, evaluator, candidate_routes)
    print_plan_summary(
        "Simulated annealing (final plan)",
        anneal_state,
        evaluator,
        demands,
        candidate_routes,
        baseline_costs,
    )


if __name__ == "__main__":
    run_disruption_planner()
