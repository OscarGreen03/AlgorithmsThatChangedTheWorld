from collections import deque # faster v. of list for pop and append
import random
import time
import matplotlib.pyplot as plt
import unittest

from networkx.algorithms.flow import maximum_flow
import networkx as nx


def bfs(residual_graph, source, sink, parent, precision_threshold=0):
    visited = [False] * len(residual_graph) # track visited
    queue = deque([source])
    visited[source] = True

    while queue:
        u = queue.popleft()

        for v, capacity in enumerate(residual_graph[u]):
            if not visited[v] and capacity > precision_threshold: # precision threshold for float handling
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False # no augmenting path found (Fini)


def edmonds_karp(graph, source, sink, precision_threshold=0):
    ## graph is represented by an adjacency matrix

    residual_graph = [row[:] for row in graph] #first iteration, residual graph is identical to original capacities
    parent = [-1] * len(graph) # stores augmenting path
    max_flow = 0

    while bfs(residual_graph, source, sink, parent, precision_threshold=precision_threshold):
        path_flow = float('inf')
        s = sink
        while s != source: # backtrack from sink to source finding the augmenting path capacity
            path_flow = min(path_flow, residual_graph[parent[s]][s]) # minimum capacity across augmenting path
            s = parent[s]

        # update residual capacities
        current_sink = sink
        while current_sink != source:
            prev_node = parent[current_sink]
            residual_graph[prev_node][current_sink] -= path_flow # each edge has addition path_flow units flowing thru
            residual_graph[current_sink][prev_node] += path_flow # reverse edge residual capacity updated (for backflow)
            current_sink = prev_node

        max_flow += path_flow


    return max_flow

def generate_graph(nodes, edge_probability=0.5, max_capacity=20, precision=0):
    """Generates a random graph"""
    graph = [[0 for _ in range(nodes)] for _ in range(nodes)]
    for i in range(nodes):
        for j in range(nodes):
            if i != j and random.random() < edge_probability:
                if precision == 0:
                    graph[i][j] = random.randint(1, max_capacity)
                else:
                    graph[i][j] = round(random.uniform(1, max_capacity), precision)
    return graph


def parity_check_with_networkx(graph, source, sink):

    G = nx.DiGraph()
    for u in range(len(graph)):
        for v, capacity in enumerate(graph[u]):
            if capacity > 0:
                G.add_edge(u, v, capacity=capacity)
    flow_value, _ = maximum_flow(G, source, sink)
    return flow_value





def test_edmonds_karp(repetitions=5, sizes=[5, 10, 50, 100]):
    """Test Edmonds-Karp on graphs of varying sizes and plot timings."""

    results = {}

    for size in sizes:
        print(f"Graph Size: {size}")
        times = []
        for _ in range(repetitions):
            graph = generate_graph(size, edge_probability=0.3, max_capacity=50, precision=0)
            source = 0
            sink = size - 1

            start_time = time.time()
            edmonds_karp(graph, source, sink, precision_threshold=0)
            end_time = time.time()

            times.append(end_time - start_time)
        results[size] = sum(times) / len(times)  # Average time for this size

    # Plot the results
    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title("Edmonds-Karp Execution Time vs Graph Size")
    plt.xlabel("Graph Size (Number of Nodes)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.grid(True)
    plt.show()

def test_edmonds_karp_precision(repetitions=5, precision_range=range(0, 5), graph_size=50):
    """Test how precision affects runtime"""
    results = {}

    for precision in precision_range:
        print(f"Precision: {precision}")
        times = []
        for _ in range(repetitions):
            graph = generate_graph(graph_size, edge_probability=0.3, max_capacity=50, precision=precision)
            source = 0
            sink = graph_size - 1

            start_time = time.time()
            edmonds_karp(graph, source, sink, precision_threshold=10**-precision)
            end_time = time.time()

            times.append(end_time - start_time)
        results[precision] = sum(times) / len(times)  # Average time for this precision

    # Plot the results
    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title(f"Edmonds-Karp Execution Time vs Precision (Graph Size: {graph_size})")
    plt.xlabel("Precision (Decimal Places)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.grid(True)
    plt.show()



class TestEdmondsKarp(unittest.TestCase):
    def test_single_edge(self):
        """Simple single edge graph"""
        graph = [
            [0, 5],
            [0, 0]
        ]
        source = 0
        sink = 1
        self.assertEqual(edmonds_karp(graph, source, sink), 5)

    def test_no_path(self):
        """
        Graph with no augmenting paths
        """
        graph = [
            [0, 0],  # from node 0
            [0, 0]   # from node 1
        ]
        source = 0
        sink = 1
        self.assertEqual(edmonds_karp(graph, source, sink), 0)

    def test_small_network(self):
        """
        Simple network of 4 nodes:
            0 -> 1 capacity 10
            0 -> 2 capacity 5
            1 -> 2 capacity 15
            1 -> 3 capacity 10
            2 -> 3 capacity 10

        Expected max flow: 15
        (One possible flow: 0->1:10, 0->2:5, then from 1->3:10, 2->3:5)
        """
        graph = [
            [0, 10,  5,  0],  # from node 0
            [0,  0, 15, 10],  # from node 1
            [0,  0,  0, 10],  # from node 2
            [0,  0,  0,  0]   # from node 3
        ]
        source = 0
        sink = 3
        self.assertEqual(edmonds_karp(graph, source, sink), 15)

    def test_precision_threshold(self):
        """
        Graph with small non-zero values.
        The BFS only considers edges with capacity > precision_threshold.
        If threshold = 0.5, edges with capacity 0.4 won't be used.
        """
        graph = [
            [0, 0.4, 0],
            [0, 0,   2],
            [0, 0,   0]
        ]
        source, sink = 0, 2
        # If threshold = 0, the path 0->1->2 has capacity min(0.4, 2) = 0.4
        # If threshold = 0.5, the 0->1 edge won't be used. The max flow becomes 0.
        self.assertAlmostEqual(edmonds_karp(graph, source, sink, precision_threshold=0), 0.4)
        self.assertAlmostEqual(edmonds_karp(graph, source, sink, precision_threshold=0.5), 0.0)

    def test_cycle_graph(self):
        graph = [
            [0, 10, 0],
            [0, 0, 10],
            [10, 0, 0]
        ]
        self.assertEqual(edmonds_karp(graph, 0, 2), 10)

    def test_large_capacity(self):
        graph = [
            [0, 1000000, 0],
            [0, 0, 1000000],
            [0, 0, 0]
        ]
        self.assertEqual(edmonds_karp(graph, 0, 2), 1000000)

    def test_parallel_paths(self):
        graph = [
            [0, 10, 10],
            [0, 0, 10],
            [0, 0, 0]
        ]
        self.assertEqual(edmonds_karp(graph, 0, 2), 20)

    def test_simple_sources(self):
        graph = [
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10],
            [0, 0, 0, 0]
        ]
        self.assertEqual(edmonds_karp(graph, 0, 3), 10)

    def test_parity_check(self):
        # test against networkx maximum_flow implementaiton
        for i in range(10):  # Run multiple iterations
            graph = generate_graph(10, edge_probability=0.3, max_capacity=20, precision=0)
            source, sink = 0, 9
            flow_ek = edmonds_karp(graph, source, sink)
            flow_nx = parity_check_with_networkx(graph, source, sink)
            self.assertEqual(flow_ek, flow_nx)




if __name__ == "__main__":
    test_edmonds_karp_precision(100, precision_range=range(0,20), graph_size=100)
    test_edmonds_karp(repetitions=50, sizes=[5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,])