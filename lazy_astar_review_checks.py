"""Assertion checks for the standard and lazy A* implementations."""

import io
import math
from contextlib import redirect_stdout

import numpy as np


def _run(search, task_map, start, goal, heuristic, tree):
    with redirect_stdout(io.StringIO()):
        return search(task_map, *start, *goal, heuristic, tree)


def _check_result(result, task_map, start, goal, expected_cost, true_evaluations):
    assert isinstance(result, tuple) and len(result) == 8, "Search must return eight values."
    found, end, steps, tree_size, evaluations, opened, expanded, edges = result
    assert isinstance(found, bool) and found is (expected_cost is not None)
    assert isinstance(steps, int) and steps >= 0
    assert isinstance(tree_size, int) and tree_size >= 1
    assert isinstance(evaluations, int) and evaluations == true_evaluations
    assert expanded is not None and isinstance(edges, dict)
    for (i1, j1, i2, j2), precise in edges.items():
        assert isinstance(precise, bool)
        assert (i2, j2) in task_map.get_neighbors(i1, j1), "An edge is not a legal move."

    if expected_cost is None:
        assert end is None and opened is None
        return evaluations

    assert end is not None and opened is not None
    assert (end.i, end.j) == goal
    assert math.isclose(end.g, expected_cost, rel_tol=1e-9, abs_tol=1e-9)
    path = []
    seen = set()
    node = end
    while node is not None:
        point = (node.i, node.j)
        assert point not in seen, "Parent pointers contain a cycle."
        assert task_map.in_bounds(*point) and task_map.traversable(*point)
        seen.add(point)
        path.append(node)
        node = node.parent
    path.reverse()
    assert (path[0].i, path[0].j) == start and path[0].g == 0

    cost = 0.0
    for first, second in zip(path, path[1:]):
        edge = (first.i, first.j, second.i, second.j)
        assert (second.i, second.j) in task_map.get_neighbors(first.i, first.j)
        assert edges.get(edge) is True, "Every final-path edge must be precisely evaluated."
        step_cost = math.sqrt(2) if first.i != second.i and first.j != second.j else 1.0
        cost += step_cost
        assert math.isclose(second.g, first.g + step_cost, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(cost, expected_cost, rel_tol=1e-9, abs_tol=1e-9)
    return evaluations


def review_lazy_astar(
    Map, Node, SearchTreePQD, diagonal_distance,
    astar, lazy_astar, compute_cost, estimate_cost,
):
    assert diagonal_distance(0, 0, 0, 0) == 0
    assert math.isclose(diagonal_distance(0, 0, 3, 5), 3 * math.sqrt(2) + 2)
    assert math.isclose(diagonal_distance(3, 5, 0, 0), 3 * math.sqrt(2) + 2)
    assert compute_cost(0, 0, 0, 1) == 1
    assert math.isclose(compute_cost(0, 0, 1, 1), math.sqrt(2))
    assert 0 <= estimate_cost(0, 0, 0, 1) <= 1
    assert 0 <= estimate_cost(0, 0, 1, 1) <= math.sqrt(2)

    tree = SearchTreePQD()
    first = Node(0, 0, f=1)
    tree.add_to_open(first)
    tree.add_to_open(Node(0, 0, f=2))
    assert tree.get_best_node_from_open() is first
    tree.add_to_closed(first)
    assert tree.get_best_node_from_open() is None
    assert tree.number_of_open_duplicates == 1

    # Remove the artificial delay and random variation while preserving
    # the exact move costs and an admissible, deliberately low estimate.
    calls = [0]

    def true_cost(i1, j1, i2, j2):
        calls[0] += 1
        di, dj = abs(i1 - i2), abs(j1 - j2)
        assert max(di, dj) == 1 and di + dj > 0
        return math.sqrt(2) if di and dj else 1.0

    def cheap_estimate(i1, j1, i2, j2):
        di, dj = abs(i1 - i2), abs(j1 - j2)
        assert max(di, dj) == 1 and di + dj > 0
        return 0.7 * (math.sqrt(2) if di and dj else 1.0)

    namespaces = {id(func.__globals__): func.__globals__ for func in (astar, lazy_astar)}
    originals = []
    for namespace in namespaces.values():
        originals.append((namespace, namespace["compute_cost"], namespace["estimate_cost"]))
        namespace["compute_cost"] = true_cost
        namespace["estimate_cost"] = cheap_estimate

    try:
        cases = (
            (np.zeros((1, 1), dtype=np.int8), (0, 0), (0, 0), 0),
            (np.zeros((1, 6), dtype=np.int8), (0, 0), (0, 5), 5),
            (np.zeros((10, 10), dtype=np.int8), (0, 0), (9, 9), 9 * math.sqrt(2)),
            (np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ], dtype=np.int8), (0, 0), (4, 4), 8),
            (np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ], dtype=np.int8), (0, 0), (2, 2), None),
        )
        open_grid_evaluations = {}
        for index, (cells, start, goal, expected_cost) in enumerate(cases):
            task_map = Map(cells)
            for name, search in (("A*", astar), ("Lazy A*", lazy_astar)):
                calls[0] = 0
                result = _run(search, task_map, start, goal, diagonal_distance, SearchTreePQD)
                evaluations = _check_result(
                    result, task_map, start, goal, expected_cost, calls[0]
                )
                if index == 2:
                    open_grid_evaluations[name] = evaluations
        assert open_grid_evaluations["Lazy A*"] < open_grid_evaluations["A*"], (
            "Lazy A* should evaluate fewer true edges on the open grid."
        )
    finally:
        for namespace, original_compute, original_estimate in originals:
            namespace["compute_cost"] = original_compute
            namespace["estimate_cost"] = original_estimate

    print("Lazy A* assertion checks passed.")
