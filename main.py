from typing import Dict, List, Tuple, Optional
from random import randint, choice
from pprint import pprint
from collections import Counter, deque
from copy import deepcopy
from math import exp

from visualize import generate_images, _clear

# Types
LevelType = List[List[int]]

# Game level variables
WIDTH: int = 20
HEIGHT: int = 10
N_CELL_TYPES: int = 5
CELL_TYPES: Dict[int, str] = {
    0: 'Floor',
    1: 'Space',
    2: 'Wall',
    3: 'Start',
    4: 'End',
}
FLOOR = 0
SPACE = 1
WALL = 2
START = 3
END = 4

# Evolutionary algorithm variables
POPULATION_SIZE: int = 10
NUM_GENERATIONS: int = 100


def flatten(l):
    """Flatten the 2D list to 1D list
    """
    return [item for sublist in l for item in sublist]


def random_child(amount: int = 1) -> List[LevelType]:
    """Generate the list of random levels
    """
    assert amount >= 0

    res: List[LevelType] = []
    for i in range(amount):
        res.append([])
        for j in range(HEIGHT):
            res[i].append([randint(0, N_CELL_TYPES - 3) for i in range(WIDTH)])

        # Start cell allocation
        res[i][randint(0, HEIGHT - 1)][randint(0, WIDTH - 1)] = START

        # End cell allocation
        x, y = randint(0, HEIGHT - 1), randint(0, WIDTH - 1)
        while res[i][x][y] == START:
            x, y = randint(0, HEIGHT - 1), randint(0, WIDTH - 1)

        res[i][x][y] = END

    return res


def fitness_function(level: LevelType) -> int:
    """The level evaluation function
    """
    assert level
    assert find_elem_position(level, START)
    assert find_elem_position(level, END)

    # Balance evaluation - the amount of each block should be approximately the same
    counter = Counter(flatten(level))
    del counter[START]
    del counter[END]

    counts = counter.values()
    total = sum(counts)

    balance = 1 - sum([abs(count / total - 1 / (N_CELL_TYPES - 2))
                       for count in counts])

    # The path have end
    has_end = 1 if find_elem_position(find_path(level), END) is not None else 0

    # Distance
    distance = moves_to_reach_end(level) if has_end else 0
    distance_score = 1 / (1 + exp(2.5 - 0.5 * distance))

    return balance + has_end + distance_score


def crossover(level1: LevelType, level2: LevelType) -> LevelType:
    """Generate the new level based on two others
    """
    path1 = find_path(level1)
    path2 = find_path(level2)

    level = deepcopy(path1)

    for i, rows in enumerate(zip(path1, path2)):
        for j, cells in enumerate(zip(rows[0], rows[1])):
            cell1, cell2 = cells
            if cell1 is None and cell2 is not None and cell2 not in [START, END]:
                level[i][j] = cell2

            if cell1 is None and cell2 is None and cell2 not in [START, END]:
                level[i][j] = choice((level1, level2))[i][j]
            else:
                level[i][j] = level1[i][j]

    return level


def mutate(level: LevelType) -> LevelType:
    """Mutate the level
    """
    res = deepcopy(level)

    n_mutations = randint(1, 10)
    while n_mutations > 0:
        x, y = randint(0, HEIGHT - 1), randint(0, WIDTH - 1)

        if res[x][y] == START or res[x][y] == END:
            continue

        res[x][y] = randint(0, 2)
        n_mutations -= 1

    return res


def get_best_levels(levels: List[LevelType], scores: List[int], amount: int = 2) -> List[LevelType]:
    """Get best levels based on scores
    """
    assert len(levels) == len(scores)
    assert len(levels) >= amount

    sorted_levels = sorted(zip(scores, levels), reverse=True)
    return [sorted_levels[i][1] for i in range(amount)]


def find_elem_position(level: LevelType, cell: int) -> Optional[Tuple[int, int]]:
    """Find first occure of cell in level
    """
    for i, row in enumerate(level):
        for j, el in enumerate(row):
            if el == cell:
                return (i, j)


def find_path(level: LevelType) -> LevelType:
    """Find path from start to connected to it free spaces and floors
    """
    assert find_elem_position(level, START)

    path = [[None for i in range(WIDTH)] for j in range(HEIGHT)]
    queue = deque()
    queue.append(find_elem_position(level, START))
    visited = set()

    while len(queue) > 0:
        x, y = queue.popleft()

        if x < 0 or y < 0 or x >= len(level) or y >= len(level[0]):
            continue

        if (x, y) in visited:
            continue

        if level[x][y] == WALL:
            continue

        if level[x][y] == SPACE:
            queue.append((x - 1, y))
            queue.append((x + 1, y))
            queue.append((x, y - 1))
            queue.append((x, y + 1))

        if level[x][y] in [FLOOR, START, END]:
            queue.append((x - 1, y))
            queue.append((x, y - 1))
            queue.append((x, y + 1))

        visited.add((x, y))
        path[x][y] = level[x][y]

    return path


def moves_to_reach_end(level: LevelType) -> int:
    """Get amount of cells between start and destination
    """
    assert level is not None

    queue = deque()

    x, y = find_elem_position(level, START)
    queue.append((x, y, 0))
    visited = set()

    while len(queue) > 0:
        x, y, count = queue.popleft()

        if x < 0 or y < 0 or x >= len(level) or y >= len(level[0]):
            continue

        if (x, y) in visited:
            continue

        if level[x][y] == WALL:
            continue

        if level[x][y] == END:
            return count

        if level[x][y] == SPACE:
            queue.append((x - 1, y, count + 1))
            queue.append((x + 1, y, count + 1))
            queue.append((x, y - 1, count + 1))
            queue.append((x, y + 1, count + 1))

        if level[x][y] in [FLOOR, START, END]:
            queue.append((x - 1, y, count + 1))
            queue.append((x, y - 1, count + 1))
            queue.append((x, y + 1, count + 1))

        visited.add((x, y))

    return -1


if __name__ == '__main__':
    population: List[LevelType] = random_child(POPULATION_SIZE)

    # Main evolutionary algoritm loop
    for i in range(NUM_GENERATIONS):
        # Evaluate each member of the population
        scores = list(map(fitness_function, population))

        print(f'Average score: {sum(scores) / len(scores)}')

        # Get 2 best levels
        level1, level2 = get_best_levels(population, scores)

        # Generate the new level based on 2 best
        new_level = crossover(level1, level2)

        # Mutate the generated level to POPULATION_SIZE
        population = list(map(mutate, [new_level
                                       for _ in range(POPULATION_SIZE)]))

    generate_images(population, width=WIDTH, height=HEIGHT)
