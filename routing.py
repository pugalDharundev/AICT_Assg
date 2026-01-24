from collections import defaultdict, deque
import heapq
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# MRT GRAPH DEFINITION
# =========================
class MRTGraph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.coords = {}

    def add_station(self, name, x, y):
        self.coords[name] = (x, y)

    def add_connection(self, a, b, travel_time, crowded=False, transfer=False):
        self.edges[a].append((b, travel_time, crowded, transfer))
        self.edges[b].append((a, travel_time, crowded, transfer))


# =========================
# COST & HEURISTIC
# =========================
def compute_cost(travel_time, transfer=False, crowded=False):
    transfer_penalty = 3 if transfer else 0
    crowd_penalty = 5 if crowded else 0
    return travel_time + transfer_penalty + crowd_penalty


def heuristic(graph, a, b):
    x1, y1 = graph.coords[a]
    x2, y2 = graph.coords[b]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# =========================
# SEARCH ALGORITHMS
# =========================

def bfs(graph, start, goal):
    queue = deque([(start, [start], 0)])
    visited = set()
    expanded = 0

    while queue:
        node, path, cost = queue.popleft()
        expanded += 1

        if node == goal:
            return path, cost, expanded

        if node not in visited:
            visited.add(node)
            for nxt, t, c, tr in graph.edges[node]:
                new_cost = cost + compute_cost(t, tr, c)
                queue.append((nxt, path + [nxt], new_cost))

    return None, float("inf"), expanded


def dfs(graph, start, goal):
    stack = [(start, [start], 0)]
    visited = set()
    expanded = 0

    while stack:
        node, path, cost = stack.pop()
        expanded += 1

        if node == goal:
            return path, cost, expanded

        if node not in visited:
            visited.add(node)
            for nxt, t, c, tr in graph.edges[node]:
                new_cost = cost + compute_cost(t, tr, c)
                stack.append((nxt, path + [nxt], new_cost))

    return None, float("inf"), expanded


def gbfs(graph, start, goal):
    pq = [(heuristic(graph, start, goal), start, [start], 0)]
    visited = set()
    expanded = 0

    while pq:
        _, node, path, cost = heapq.heappop(pq)
        expanded += 1

        if node == goal:
            return path, cost, expanded

        if node not in visited:
            visited.add(node)
            for nxt, t, c, tr in graph.edges[node]:
                h = heuristic(graph, nxt, goal)
                new_cost = cost + compute_cost(t, tr, c)
                heapq.heappush(pq, (h, nxt, path + [nxt], new_cost))

    return None, float("inf"), expanded


def astar(graph, start, goal):
    pq = [(heuristic(graph, start, goal), 0, start, [start])]
    visited = {}
    expanded = 0

    while pq:
        f, g, node, path = heapq.heappop(pq)
        expanded += 1

        if node == goal:
            return path, g, expanded

        if node not in visited or g < visited[node]:
            visited[node] = g
            for nxt, t, c, tr in graph.edges[node]:
                g_new = g + compute_cost(t, tr, c)
                f_new = g_new + heuristic(graph, nxt, goal)
                heapq.heappush(pq, (f_new, g_new, nxt, path + [nxt]))

    return None, float("inf"), expanded


# =========================
# NETWORK MODES
# =========================
def build_today_mode():
    """
    Build Today Mode network (current EWL airport branch operations).
    Includes 25 stations across Singapore MRT lines.
    """
    g = MRTGraph()

    # Station coordinates (normalized for meaningful heuristic distance)
    stations = {
        # East-West Line (Airport Branch)
        "Changi Airport": (10, 2),
        "Expo": (9, 3),
        "Tanah Merah": (8, 4),
        
        # East-West Line (Main)
        "Paya Lebar": (7, 5),
        "Aljunied": (6.5, 5.5),
        "Bugis": (6, 6),
        "Dhoby Ghaut": (5.5, 6.5),
        "City Hall": (5, 6),
        "Raffles Place": (4.5, 5.5),
        "Tanjong Pagar": (4, 5),
        "Outram Park": (3.5, 5),
        "Tiong Bahru": (3, 5),
        "Clementi": (2, 6),
        "Jurong East": (1, 7),
        
        # North-South Line
        "Yio Chu Kang": (7, 9),
        "Bishan": (6, 9),
        "Novena": (5.5, 8),
        "Newton": (5, 7.5),
        "Orchard": (4, 7),
        "Somerset": (3.5, 6.5),
        "Dhoby Ghaut": (5.5, 6.5),
        
        # Circle Line & Others
        "HarbourFront": (2, 5),
        "Tampines": (8, 3),
        "Potong Pasir": (7, 6),
        "Gardens by the Bay": (5, 4)
    }

    for s, (x, y) in stations.items():
        g.add_station(s, x, y)

    # EWL Airport Branch
    g.add_connection("Changi Airport", "Expo", 5, crowded=True)
    g.add_connection("Expo", "Tanah Merah", 4)
    
    # EWL Main Line
    g.add_connection("Tanah Merah", "Paya Lebar", 6, transfer=True)
    g.add_connection("Paya Lebar", "Aljunied", 3)
    g.add_connection("Aljunied", "Bugis", 3)
    g.add_connection("Bugis", "Dhoby Ghaut", 2)
    g.add_connection("Dhoby Ghaut", "City Hall", 2)
    g.add_connection("City Hall", "Raffles Place", 2)
    g.add_connection("Raffles Place", "Tanjong Pagar", 3)
    g.add_connection("Tanjong Pagar", "Outram Park", 2)
    g.add_connection("Outram Park", "Tiong Bahru", 2)
    g.add_connection("Tiong Bahru", "Clementi", 5)
    g.add_connection("Clementi", "Jurong East", 3)
    
    # NSL (Bishan area)
    g.add_connection("Yio Chu Kang", "Bishan", 4)
    g.add_connection("Bishan", "Novena", 3)
    g.add_connection("Novena", "Newton", 2)
    g.add_connection("Newton", "Orchard", 3, transfer=True)
    g.add_connection("Orchard", "Somerset", 2)
    
    # Interchange & Connectors
    g.add_connection("Dhoby Ghaut", "Orchard", 3, transfer=True)
    g.add_connection("HarbourFront", "Outram Park", 4, transfer=True)
    g.add_connection("City Hall", "Gardens by the Bay", 3)
    g.add_connection("Tampines", "Tanah Merah", 5)
    g.add_connection("Paya Lebar", "Potong Pasir", 3)

    return g


def build_future_mode():
    """
    Build Future Mode network with TELe and CRL extensions.
    Incorporates:
    - TELe: Sungei Bedok → T5 → Tanah Merah (converted from EWL)
    - CRL extension to T5
    - Conversion of Tanah Merah–Expo–Changi Airport to TEL systems
    """
    g = build_today_mode()

    # New stations for TEL/CRL
    g.add_station("Changi Terminal 5", 11, 2)
    g.add_station("Sungei Bedok", 9, 1)
    g.add_station("Bedok", 9.5, 2)
    g.add_station("Punggol Digital District", 11, 7)  # CRL extension

    # TEL Extension (Sungei Bedok → T5 → Tanah Merah conversion)
    g.add_connection("Sungei Bedok", "Bedok", 3)
    g.add_connection("Bedok", "Changi Terminal 5", 4)
    g.add_connection("Changi Terminal 5", "Expo", 5, transfer=True)  # TEL connection
    g.add_connection("Expo", "Tanah Merah", 4)  # Now part of TEL
    
    # Changi Airport to T5 (converted to TEL)
    g.add_connection("Changi Airport", "Changi Terminal 5", 4)
    
    # CRL Extension to T5
    g.add_connection("Punggol Digital District", "Changi Terminal 5", 8, transfer=True)

    return g





# =========================
# PERFORMANCE TRACKING & ANALYSIS
# =========================
def run_experiments(graphs, od_pairs, algorithms):
    """Run all experiments and collect results in a structured format."""
    results = []
    
    for mode, graph in graphs.items():
        print(f"\n{'='*80}")
        print(f"TESTING: {mode}")
        print(f"{'='*80}")
        
        for start, goal in od_pairs[mode]:
            print(f"\n[Route: {start} → {goal}]")
            
            for algo_name, algo_func in algorithms.items():
                try:
                    start_time = time.time()
                    path, cost, expanded = algo_func(graph, start, goal)
                    runtime = (time.time() - start_time) * 1000  # milliseconds
                    
                    result = {
                        "Mode": mode,
                        "Start": start,
                        "Goal": goal,
                        "Algorithm": algo_name,
                        "Path Length": len(path) if path else 0,
                        "Path Cost": round(cost, 2) if cost != float("inf") else "N/A",
                        "Nodes Expanded": expanded,
                        "Runtime (ms)": round(runtime, 3),
                        "Path": " → ".join(path) if path else "No path found"
                    }
                    results.append(result)
                    
                    print(f"  {algo_name:6} | Cost: {result['Path Cost']:6} | Nodes: {expanded:4} | Time: {runtime:7.3f}ms")
                except Exception as e:
                    print(f"  {algo_name:6} | ERROR: {str(e)}")
    
    return pd.DataFrame(results)


def analyze_results(df):
    """Generate performance analysis tables."""
    print(f"\n\n{'='*80}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Group by mode and algorithm
    for mode in df["Mode"].unique():
        mode_df = df[df["Mode"] == mode]
        
        print(f"\n{mode.upper()}")
        print("-" * 80)
        
        # Summary statistics
        algo_summary = mode_df.groupby("Algorithm").agg({
            "Nodes Expanded": ["mean", "min", "max"],
            "Runtime (ms)": ["mean", "min", "max"],
            "Path Cost": ["mean"]
        }).round(2)
        
        print("\nAlgorithm Performance Statistics:")
        print(algo_summary)
        
        # Best algorithm for each metric
        print("\nBest Performers:")
        print(f"  Fastest:        {mode_df.loc[mode_df['Runtime (ms)'].idxmin(), 'Algorithm']}")
        print(f"  Fewest Nodes:   {mode_df.loc[mode_df['Nodes Expanded'].idxmin(), 'Algorithm']}")
        
        # Show sample paths
        print("\nSample Paths (First Route):")
        first_route = mode_df.groupby("Algorithm").first()
        for algo in ["BFS", "DFS", "GBFS", "A*"]:
            if algo in first_route.index:
                path_info = first_route.loc[algo]
                print(f"  {algo:6} → {path_info['Path']}")


def create_visualizations(df):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MRT Routing Algorithm Performance Comparison", fontsize=16, fontweight='bold')
    
    # 1. Runtime Comparison
    ax1 = axes[0, 0]
    for mode in df["Mode"].unique():
        mode_data = df[df["Mode"] == mode].groupby("Algorithm")["Runtime (ms)"].mean()
        ax1.bar(mode_data.index, mode_data.values, label=mode, alpha=0.8)
    ax1.set_title("Average Runtime by Algorithm")
    ax1.set_ylabel("Runtime (ms)")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Nodes Expanded Comparison
    ax2 = axes[0, 1]
    for mode in df["Mode"].unique():
        mode_data = df[df["Mode"] == mode].groupby("Algorithm")["Nodes Expanded"].mean()
        ax2.bar(mode_data.index, mode_data.values, label=mode, alpha=0.8)
    ax2.set_title("Average Nodes Expanded by Algorithm")
    ax2.set_ylabel("Number of Nodes")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Path Cost Comparison
    ax3 = axes[1, 0]
    mode_df_today = df[df["Mode"] == "Today Mode"]
    if not mode_df_today.empty:
        cost_data = pd.to_numeric(mode_df_today["Path Cost"], errors='coerce').dropna()
        algo_costs = mode_df_today.groupby("Algorithm")["Path Cost"].apply(
            lambda x: pd.to_numeric(x, errors='coerce').mean()
        )
        ax3.bar(algo_costs.index, algo_costs.values, alpha=0.8, color='steelblue')
        ax3.set_title("Average Path Cost (Today Mode)")
        ax3.set_ylabel("Total Cost")
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Efficiency Score (low runtime + low nodes = good)
    ax4 = axes[1, 1]
    efficiency = {}
    for mode in df["Mode"].unique():
        mode_data = df[df["Mode"] == mode]
        for algo in mode_data["Algorithm"].unique():
            algo_data = mode_data[mode_data["Algorithm"] == algo]
            avg_time = algo_data["Runtime (ms)"].mean()
            avg_nodes = algo_data["Nodes Expanded"].mean()
            # Normalize and combine (lower is better)
            score = (avg_time / max(mode_data["Runtime (ms)"]) + 
                    avg_nodes / max(mode_data["Nodes Expanded"])) / 2
            label = f"{algo} ({mode[:6]})"
            efficiency[label] = score
    
    sorted_eff = sorted(efficiency.items(), key=lambda x: x[1])
    labels, scores = zip(*sorted_eff)
    colors = ['green' if s < 0.4 else 'orange' if s < 0.7 else 'red' for s in scores]
    ax4.barh(labels, scores, color=colors, alpha=0.8)
    ax4.set_title("Efficiency Score (Lower is Better)")
    ax4.set_xlabel("Score")
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('routing_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'routing_analysis.png'")
    plt.show()



if __name__ == "__main__":
    print("\n" + "="*80)
    print("CHANGILINK AI: MRT ROUTING & DISRUPTION SUPPORT SYSTEM")
    print("="*80)
    
    # Build graphs
    graphs = {
        "Today Mode": build_today_mode(),
        "Future Mode": build_future_mode()
    }
    
    # Recommended OD pairs from assignment
    od_pairs = {
        "Today Mode": [
            ("Changi Airport", "City Hall"),
            ("Changi Airport", "Orchard"),
            ("Changi Airport", "Gardens by the Bay"),
            ("Changi Airport", "HarbourFront"),
            ("Changi Airport", "Bishan"),
        ],
        "Future Mode": [
            ("Changi Terminal 5", "City Hall"),
            ("Changi Terminal 5", "Orchard"),
            ("Changi Terminal 5", "Gardens by the Bay"),
            ("Paya Lebar", "Changi Terminal 5"),
            ("HarbourFront", "Changi Terminal 5"),
            ("Bishan", "Changi Terminal 5"),
            ("Tampines", "Changi Terminal 5"),
        ]
    }
    
    # Algorithms to test
    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "GBFS": gbfs,
        "A*": astar
    }
    
    # Run experiments
    print("\nRunning experiments...")
    results_df = run_experiments(graphs, od_pairs, algorithms)
    
    # Analyze results
    analyze_results(results_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results_df)
    
    # Save results to CSV
    results_df.to_csv('routing_results.csv', index=False)
    print("\n✓ Detailed results saved to 'routing_results.csv'")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
