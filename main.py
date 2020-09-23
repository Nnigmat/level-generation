from typing import Dict, List
from random import randint
from pprint import pprint

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

# Evolutionary algorithm variables
POPULATION_SIZE: int = 10
NUM_GENERATIONS: int = 10


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
        res[i][randint(0, HEIGHT - 1)][randint(0, WIDTH - 1)] = 3

        # End cell allocation
        x, y = randint(0, HEIGHT - 1), randint(0, WIDTH - 1)
        while res[i][x][y] == 3:
            x, y = randint(0, HEIGHT - 1), randint(0, WIDTH - 1)

        res[i][x][y] = 4

    return res


def fitness_function(level: LevelType) -> int:
    """The level evaluation function
    """
    pass


def crossover(level1: LevelType, level2: LevelType) -> LevelType:
    """Generate the new level based on two others 
    """
    pass


def mutate(level: LevelType) -> LevelType:
    """Mutate the level
    """
    pass


def get_best_levels(levels: List[LevelType], scores: List[int], amount: int = 2) -> List[LevelType]:
    """Get best levels based on scores 
    """
    assert len(levels) == len(scores)
    assert len(levels) >= amount

    sorted_levels = sorted(zip(scores, levels), reverse=True)
    return [sorted_levels[i][1] for i in range(amount)]


if __name__ == '__main__':
    population: List[LevelType] = random_child(POPULATION_SIZE)
    generate_images(population, width=WIDTH, height=HEIGHT)

    '''
    # Main evolutionary algoritm loop
    for i in range(NUM_GENERATIONS):
        # Evaluate each member of the population
        scores = list(map(fitness_function, population))

        # Get 2 best levels
        level1, level2 = get_best_levels(population, scores)

        # Generate the new level based on 2 best
        new_level = crossover(level1, level2)

        # Mutate the generated level to POPULATION_SIZE
        population = list(map(mutate, [new_level] * POPULATION_SIZE))
    '''
