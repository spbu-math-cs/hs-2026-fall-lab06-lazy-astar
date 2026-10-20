import traceback
from textwrap import dedent
from typing import Callable, Union
from random import randint
from utils.Map import Map
from utils.Node import BaseNode as Node
from utils.utils import make_path, convert_string_to_cells, draw


def simple_test(search_function: Callable, task: Union[int, None], heuristic, tree, draw_result=True):
    """
    Function `simple_test` runs `search_function` on one task. Use a number from 0 to 24 to choose a specific debug task
    on a simple map, or use None to select a random task from this pool. The function displays:
     - 'Path found!' and some statistics if a path was found.
     - 'Path not found!' if a path was not discovered.
     - 'Execution error' if an error occurred during the execution of the search_function.
    In the first case, the function also provides a visualization of the task.

    Parameters
    ----------
    search_function : Callable
        Implementation of the search method.
    task : int | None
        A number from 0 to 24 to choose a specific debug task on a simple map,
        or None to select a random task from this pool.
    *args
        Additional arguments passed to the search function.
    """

    def get_map_data():
        map_str = dedent(
            """
            . . . . . . . . . . . . . . . . . . . . . # # . . . . . . .  
            . . . . . . . . . . . . . . . . . . . . . # # . . . . . . . 
            . . . . . . . . . . . . . . . . . . . . . # # . . . . . . . 
            . . . # # . . . . . . . . . . . . . . . . # # . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . # # . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . # # # # # . . . . 
            . . . # # . . . . . . . . # # . . . . . . # # # # # . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . . . . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . . . . . . . . . . . # # . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . # # . . . . . . . . . . . . . . .
        """
        )
        cells = convert_string_to_cells(map_str)
        return Map(cells)

    task_map = get_map_data()
    starts = [(9, 0), (13, 0), (7, 28), (14, 29), (4, 1), (0, 17), (5, 6), (5, 20), (12, 2), (7, 28), (11, 9), (3, 2), (3, 17), (13, 20), (1, 1), (9, 10), (14, 6), (2, 0), (9, 28), (8, 6), (11, 6), (3, 0), (8, 9), (14, 7), (12, 4)]
    goals = [(11, 20), (2, 19), (6, 5), (4, 18), (9, 20), (7, 0), (2, 25), (12, 4), (3, 25), (0, 12), (4, 23), (2, 24), (9, 2), (1, 6), (13, 29), (14, 29), (2, 28), (14, 16), (13, 0), (1, 27), (14, 25), (10, 20), (12, 28), (2, 29), (1, 29)]
    lengths = [31.314, 24.728, 25.899, 15.728, 23.899, 21.657, 27.899, 21.728, 34.971, 18.899, 30.899, 32.899, 22.414, 21.899, 32.971, 28.385, 34.971, 26.828, 35.799, 30.142, 28.627, 24.899, 25.971, 34.971, 35.799]

    if (task is None) or not (0 <= task < 25):
        task = randint(0, 24)

    start = Node(*starts[task])
    goal = Node(*goals[task])
    length = lengths[task]
    try:
        (
            found,
            end_node,
            number_of_steps,
            search_tree_size,
            transitions_computed,
            open_set,
            closed_set,
            edges,
        ) = search_function(task_map, start.i, start.j, goal.i, goal.j, heuristic, tree)

        expansions = len(closed_set)

        if found:
            path, path_length = make_path(end_node)

            correct = abs(float(path_length) - float(length)) < 0.01
            if draw_result:
                draw(task_map, start, goal, path, open_set, closed_set, edges)
            print(
                f"Path found! Length: {round(path_length, 3)}. "
                f"Tree size: {search_tree_size}. "
                f"Number of expansions: {expansions}. "
                f"Transitions computed: {transitions_computed}. "
                f"Correct: {correct}."
            )
        else:
            print("Path not found!")
        return

    except Exception as e:
        print(f"Execution error: {e}")
        traceback.print_exc()


def massive_test(search_function: Callable, heuristic, tree):
    for i in range(25):
        simple_test(search_function, i, heuristic, tree, False)