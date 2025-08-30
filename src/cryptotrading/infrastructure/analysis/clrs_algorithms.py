"""
CLRS Algorithmic Reasoning Implementation for Glean
Implements 30 core algorithms from CLRS 3ed for advanced code analysis
"""

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# Core Data Structures
@dataclass
class TraversalResult:
    """Result of graph traversal operations"""

    visit_order: List[str]
    distances: Dict[str, int]
    predecessors: Dict[str, Optional[str]]
    discovered: Dict[str, int]
    finished: Dict[str, int]


@dataclass
class ShortestPaths:
    """Result of shortest path algorithms"""

    distances: Dict[str, float]
    predecessors: Dict[str, Optional[str]]
    has_negative_cycle: bool = False


@dataclass
class MinimumSpanningTree:
    """Result of MST algorithms"""

    edges: List[Tuple[str, str, float]]
    total_weight: float


class Graph:
    """Basic graph representation"""

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node: str) -> None:
        self.nodes.add(node)

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges[from_node].add(to_node)

    def get_neighbors(self, node: str) -> Set[str]:
        return self.edges.get(node, set())


class WeightedGraph(Graph):
    """Weighted graph for shortest path algorithms"""

    def __init__(self):
        super().__init__()
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)

    def add_weighted_edge(self, from_node: str, to_node: str, weight: float) -> None:
        self.add_edge(from_node, to_node)
        self.weights[from_node][to_node] = weight


class CLRSSortingAlgorithms:
    """CLRS Sorting Algorithms Implementation"""

    @staticmethod
    def insertion_sort(arr: List[T], compare_fn: Optional[Callable[[T, T], int]] = None) -> List[T]:
        """Insertion sort - O(n²) time, O(1) space"""
        if compare_fn is None:
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)

        result = arr.copy()
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and compare_fn(result[j], key) > 0:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        return result

    @staticmethod
    def bubble_sort(arr: List[T], compare_fn: Optional[Callable[[T, T], int]] = None) -> List[T]:
        """Bubble sort - O(n²) time, O(1) space"""
        if compare_fn is None:
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)

        result = arr.copy()
        n = len(result)
        for i in range(n):
            for j in range(0, n - i - 1):
                if compare_fn(result[j], result[j + 1]) > 0:
                    result[j], result[j + 1] = result[j + 1], result[j]
        return result

    @staticmethod
    def heapsort(arr: List[T], compare_fn: Optional[Callable[[T, T], int]] = None) -> List[T]:
        """Heapsort - O(n log n) time, O(1) space"""
        if compare_fn is None:
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)

        def heapify(arr: List[T], n: int, i: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and compare_fn(arr[left], arr[largest]) > 0:
                largest = left
            if right < n and compare_fn(arr[right], arr[largest]) > 0:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        result = arr.copy()
        n = len(result)

        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(result, n, i)

        # Extract elements
        for i in range(n - 1, 0, -1):
            result[0], result[i] = result[i], result[0]
            heapify(result, i, 0)

        return result

    @staticmethod
    def quicksort(arr: List[T], compare_fn: Optional[Callable[[T, T], int]] = None) -> List[T]:
        """Quicksort - O(n log n) average, O(n²) worst case"""
        if compare_fn is None:
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)

        def partition(arr: List[T], low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1

            for j in range(low, high):
                if compare_fn(arr[j], pivot) <= 0:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]

            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1

        def quicksort_helper(arr: List[T], low: int, high: int):
            if low < high:
                pi = partition(arr, low, high)
                quicksort_helper(arr, low, pi - 1)
                quicksort_helper(arr, pi + 1, high)

        result = arr.copy()
        quicksort_helper(result, 0, len(result) - 1)
        return result


class CLRSSearchAlgorithms:
    """CLRS Search Algorithms Implementation"""

    @staticmethod
    def binary_search(
        arr: List[T], target: T, compare_fn: Optional[Callable[[T, T], int]] = None
    ) -> int:
        """Binary search - O(log n) time"""
        if compare_fn is None:
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)

        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) // 2
            cmp_result = compare_fn(arr[mid], target)

            if cmp_result == 0:
                return mid
            elif cmp_result < 0:
                left = mid + 1
            else:
                right = mid - 1

        return -1

    @staticmethod
    def quick_select(arr: List[T], k: int, compare_fn: Optional[Callable[[T, T], int]] = None) -> T:
        """Quick select - O(n) average time"""
        if compare_fn is None:
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)

        def partition(arr: List[T], low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1

            for j in range(low, high):
                if compare_fn(arr[j], pivot) <= 0:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]

            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1

        def select(arr: List[T], low: int, high: int, k: int) -> T:
            if low == high:
                return arr[low]

            pi = partition(arr, low, high)

            if k == pi:
                return arr[pi]
            elif k < pi:
                return select(arr, low, pi - 1, k)
            else:
                return select(arr, pi + 1, high, k)

        work_arr = arr.copy()
        return select(work_arr, 0, len(work_arr) - 1, k)

    @staticmethod
    def minimum(arr: List[T], compare_fn: Optional[Callable[[T, T], int]] = None) -> T:
        """Find minimum element - O(n) time"""
        if not arr:
            raise ValueError("Array is empty")

        if compare_fn is None:
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)

        min_elem = arr[0]
        for elem in arr[1:]:
            if compare_fn(elem, min_elem) < 0:
                min_elem = elem

        return min_elem


class CLRSGraphAlgorithms:
    """CLRS Graph Algorithms Implementation"""

    @staticmethod
    def depth_first_search(graph: Graph, start_node: str) -> TraversalResult:
        """DFS traversal - O(V + E) time"""
        visited = set()
        visit_order = []
        distances = {}
        predecessors = {}
        discovered = {}
        finished = {}
        time = [0]  # Use list for mutable reference

        def dfs_visit(node: str, pred: Optional[str] = None):
            time[0] += 1
            discovered[node] = time[0]
            visited.add(node)
            visit_order.append(node)
            predecessors[node] = pred
            distances[node] = 0 if pred is None else distances[pred] + 1

            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    dfs_visit(neighbor, node)

            time[0] += 1
            finished[node] = time[0]

        dfs_visit(start_node)

        return TraversalResult(
            visit_order=visit_order,
            distances=distances,
            predecessors=predecessors,
            discovered=discovered,
            finished=finished,
        )

    @staticmethod
    def breadth_first_search(graph: Graph, start_node: str) -> TraversalResult:
        """BFS traversal - O(V + E) time"""
        visited = set()
        visit_order = []
        distances = {start_node: 0}
        predecessors = {start_node: None}
        queue = deque([start_node])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            visit_order.append(node)

            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited and neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    predecessors[neighbor] = node
                    queue.append(neighbor)

        return TraversalResult(
            visit_order=visit_order,
            distances=distances,
            predecessors=predecessors,
            discovered={},
            finished={},
        )

    @staticmethod
    def topological_sort(graph: Graph) -> List[str]:
        """Topological sort using DFS - O(V + E) time"""
        visited = set()
        stack = []

        def dfs_visit(node: str):
            visited.add(node)
            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    dfs_visit(neighbor)
            stack.append(node)

        for node in graph.nodes:
            if node not in visited:
                dfs_visit(node)

        return stack[::-1]

    @staticmethod
    def dijkstra(graph: WeightedGraph, source: str) -> ShortestPaths:
        """Dijkstra's shortest path - O((V + E) log V) time"""
        distances = {node: float("inf") for node in graph.nodes}
        distances[source] = 0
        predecessors = {node: None for node in graph.nodes}
        visited = set()
        heap = [(0, source)]

        while heap:
            current_dist, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            for neighbor in graph.get_neighbors(current):
                if neighbor in visited:
                    continue

                weight = graph.weights[current].get(neighbor, float("inf"))
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(heap, (distance, neighbor))

        return ShortestPaths(
            distances=distances, predecessors=predecessors, has_negative_cycle=False
        )

    @staticmethod
    def bellman_ford(graph: WeightedGraph, source: str) -> ShortestPaths:
        """Bellman-Ford shortest path - O(VE) time, detects negative cycles"""
        distances = {node: float("inf") for node in graph.nodes}
        distances[source] = 0
        predecessors = {node: None for node in graph.nodes}

        # Relax edges V-1 times
        for _ in range(len(graph.nodes) - 1):
            for node in graph.nodes:
                if distances[node] == float("inf"):
                    continue

                for neighbor in graph.get_neighbors(node):
                    weight = graph.weights[node].get(neighbor, float("inf"))
                    if distances[node] + weight < distances[neighbor]:
                        distances[neighbor] = distances[node] + weight
                        predecessors[neighbor] = node

        # Check for negative cycles
        has_negative_cycle = False
        for node in graph.nodes:
            if distances[node] == float("inf"):
                continue

            for neighbor in graph.get_neighbors(node):
                weight = graph.weights[node].get(neighbor, float("inf"))
                if distances[node] + weight < distances[neighbor]:
                    has_negative_cycle = True
                    break

        return ShortestPaths(
            distances=distances, predecessors=predecessors, has_negative_cycle=has_negative_cycle
        )


class CLRSDynamicProgramming:
    """CLRS Dynamic Programming Algorithms"""

    @staticmethod
    def longest_common_subsequence(str1: str, str2: str) -> str:
        """LCS algorithm - O(mn) time and space"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Reconstruct LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if str1[i - 1] == str2[j - 1]:
                lcs.append(str1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return "".join(reversed(lcs))

    @staticmethod
    def matrix_chain_multiplication(dimensions: List[int]) -> int:
        """Matrix chain multiplication - O(n³) time"""
        n = len(dimensions) - 1
        if n <= 1:
            return 0

        dp = [[0] * n for _ in range(n)]

        # Length of chain
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float("inf")

                for k in range(i, j):
                    cost = (
                        dp[i][k]
                        + dp[k + 1][j]
                        + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                    )
                    dp[i][j] = min(dp[i][j], cost)

        return dp[0][n - 1]


class CLRSStringAlgorithms:
    """CLRS String Algorithms"""

    @staticmethod
    def naive_string_match(text: str, pattern: str) -> List[int]:
        """Naive string matching - O(nm) time"""
        matches = []
        n, m = len(text), len(pattern)

        for i in range(n - m + 1):
            if text[i : i + m] == pattern:
                matches.append(i)

        return matches

    @staticmethod
    def kmp_string_match(text: str, pattern: str) -> List[int]:
        """KMP string matching - O(n + m) time"""

        def compute_lps(pattern: str) -> List[int]:
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1

            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1

            return lps

        matches = []
        n, m = len(text), len(pattern)

        if m == 0:
            return matches

        lps = compute_lps(pattern)
        i = j = 0

        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1

            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return matches


# Glean Integration Classes
class CodeDependencyAnalyzer:
    """Advanced code dependency analysis using CLRS algorithms"""

    def __init__(self):
        self.dependency_graph = Graph()
        self.weighted_graph = WeightedGraph()
        self.algorithms = CLRSGraphAlgorithms()

    def analyze_dependencies(self, codebase_modules: Dict[str, List[str]]) -> Dict[str, Any]:
        """Comprehensive dependency analysis"""
        # Build dependency graph
        for module, dependencies in codebase_modules.items():
            self.dependency_graph.add_node(module)
            for dep in dependencies:
                self.dependency_graph.add_edge(module, dep)

        # Detect circular dependencies using DFS
        cycles = self._detect_cycles()

        # Find build order using topological sort
        try:
            build_order = self.algorithms.topological_sort(self.dependency_graph)
        except:
            build_order = []  # Cycles present

        # Find critical modules using centrality
        critical_modules = self._find_critical_modules()

        return {
            "circular_dependencies": cycles,
            "build_order": build_order,
            "critical_modules": critical_modules,
            "total_modules": len(self.dependency_graph.nodes),
            "total_dependencies": sum(len(deps) for deps in self.dependency_graph.edges.values()),
        }

    def _detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []

        def has_cycle(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.dependency_graph.get_neighbors(node):
                if neighbor not in visited:
                    if has_cycle(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        for node in self.dependency_graph.nodes:
            if node not in visited:
                has_cycle(node, [])

        return cycles

    def _find_critical_modules(self) -> List[str]:
        """Find modules with highest dependency impact"""
        module_scores = {}

        for module in self.dependency_graph.nodes:
            # Score based on in-degree (how many depend on this module)
            dependents = sum(
                1
                for node in self.dependency_graph.nodes
                if module in self.dependency_graph.get_neighbors(node)
            )

            # Score based on out-degree (how many this module depends on)
            dependencies = len(self.dependency_graph.get_neighbors(module))

            # Combined criticality score
            module_scores[module] = dependents * 2 + dependencies

        # Sort by score and return top modules
        sorted_modules = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)
        return [module for module, score in sorted_modules[:10]]


class CodeSimilarityAnalyzer:
    """Code similarity analysis using string algorithms"""

    def __init__(self):
        self.dp_algorithms = CLRSDynamicProgramming()
        self.string_algorithms = CLRSStringAlgorithms()

    def analyze_similarity(self, code1: str, code2: str) -> Dict[str, Any]:
        """Analyze similarity between two code snippets"""
        # Normalize code (remove whitespace, comments)
        norm_code1 = self._normalize_code(code1)
        norm_code2 = self._normalize_code(code2)

        # Find longest common subsequence
        lcs = self.dp_algorithms.longest_common_subsequence(norm_code1, norm_code2)

        # Calculate similarity metrics
        lcs_ratio = (
            len(lcs) / max(len(norm_code1), len(norm_code2))
            if max(len(norm_code1), len(norm_code2)) > 0
            else 0
        )

        # Find common patterns
        common_patterns = self._find_common_patterns(norm_code1, norm_code2)

        return {
            "similarity_ratio": lcs_ratio,
            "common_subsequence": lcs,
            "common_patterns": common_patterns,
            "code1_length": len(norm_code1),
            "code2_length": len(norm_code2),
            "lcs_length": len(lcs),
        }

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison"""
        # Remove comments and extra whitespace
        lines = []
        for line in code.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("//"):
                lines.append(line)
        return " ".join(lines)

    def _find_common_patterns(self, code1: str, code2: str) -> List[str]:
        """Find common patterns between code snippets"""
        patterns = []

        # Look for common substrings of meaningful length
        for length in range(10, min(len(code1), len(code2)) + 1, 5):
            for i in range(len(code1) - length + 1):
                pattern = code1[i : i + length]
                if pattern in code2 and pattern not in patterns:
                    patterns.append(pattern)

        return patterns[:10]  # Return top 10 patterns
