from collections import defaultdict, deque
import heapq
import math
import time

# Define the MRT graph structure
class MRTGraph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.coords = {}

    def add_station(self, name, x, y):
        self.coords[name] = (x, y)

    def add_connection(self, a, b, time):
        self.edges[a].append((b, time))
        self.edges[b].append((a, time))


def heuristic(graph, a, b):
    x1, y1 = graph.coords[a]
    x2, y2 = graph.coords[b]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def compute_cost(travel_time, transfers=0, crowded=False):
    """
    Cost function:
    - travel_time: minutes between stations
    - transfers: number of line transfers
    - crowded: whether the segment is crowded
    """
    transfer_penalty = 3 * transfers
    crowd_penalty = 5 if crowded else 0
    return travel_time + transfer_penalty + crowd_penalty



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
            for nxt, w in graph.edges[node]:
                queue.append((nxt, path + [nxt], cost + w))

    return None, float('inf'), expanded


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
            for nxt, w in graph.edges[node]:
                stack.append((nxt, path + [nxt], cost + w))

    return None, float('inf'), expanded


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
            for nxt, w in graph.edges[node]:
                h = heuristic(graph, nxt, goal)
                heapq.heappush(pq, (h, nxt, path + [nxt], cost + w))

    return None, float('inf'), expanded


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
            for nxt, w in graph.edges[node]:
                g_new = g + w
                f_new = g_new + heuristic(graph, nxt, goal)
                heapq.heappush(pq, (f_new, g_new, nxt, path + [nxt]))

    return None, float('inf'), expanded


def build_today_mode():
    g = MRTGraph()

    stations = {
        "Changi Airport": (10, 2),
        "Expo": (9, 3),
        "Tanah Merah": (8, 4),
        "Paya Lebar": (7, 5),
        "City Hall": (5, 6),
        "Orchard": (4, 7)
    }

    for s, (x, y) in stations.items():
        g.add_station(s, x, y)

    g.add_connection("Changi Airport", "Expo", 5)
    g.add_connection("Expo", "Tanah Merah", 4)
    g.add_connection("Tanah Merah", "Paya Lebar", 6)
    g.add_connection("Paya Lebar", "City Hall", 8)
    g.add_connection("City Hall", "Orchard", 4)

    return g


def build_future_mode():
    g = build_today_mode()

    g.add_station("Changi Terminal 5", 11, 2)
    g.add_station("Sungei Bedok", 9, 1)

    g.add_connection("Changi Terminal 5", "Sungei Bedok", 5)
    g.add_connection("Changi Terminal 5", "Tanah Merah", 6)

    return g


if __name__ == "__main__":

    graphs = {
        "Today Mode": build_today_mode(),
        "Future Mode": build_future_mode()
    }

    od_pairs_today = [
        ("Changi Airport", "City Hall"),
        ("Changi Airport", "Orchard"),
        ("Paya Lebar", "Tanah Merah"),
        ("Changi Airport", "Expo"),
    ]

    od_pairs_future = [
        ("Changi Airport", "City Hall"),
        ("Changi Airport", "Orchard"),
        ("Paya Lebar", "Tanah Merah"),
        ("Changi Airport", "Expo"),
        ("Tanah Merah", "Changi Terminal 5")
    ]

    od_pairs_by_mode = {
        "Today Mode": od_pairs_today,
        "Future Mode": od_pairs_future
    }

    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "GBFS": gbfs,
        "A*": astar
    }

    for mode, graph in graphs.items():
        print("\n==============================")
        print(mode)
        print("==============================")

        for start, goal in od_pairs_by_mode[mode]:
            print(f"\nRoute: {start} → {goal}")

            for name, algo in algorithms.items():
                start_time = time.time()
                path, cost, expanded = algo(graph, start, goal)
                runtime = time.time() - start_time

                print(f"\n{name}")
                print(" Path:", path)
                print(" Travel Cost:", cost)
                print(" Expanded Nodes:", expanded)
                print(" Runtime:", round(runtime, 6), "seconds")
