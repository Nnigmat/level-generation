from typing import Dict, List
from PIL import Image
from shutil import rmtree
from os import mkdir, getcwd

output_dir = getcwd() + './generated_images'

# Types
LevelType = List[List[int]]

# Cell sprites
CELL_SPRITES: Dict[int, str] = {
    0: 'Floor',
    1: 'Space',
    2: 'Wall',
    3: 'Start',
    4: 'End',
}


def generate_images(levels: List[LevelType]) -> None:
    """Generate images of levels based on the given levels
    and store them in output_dir
    """
    _clear()
    print('hello')


def _clear() -> None:
    """Remove output dir and create new
    """
    rmtree(output_dir)
    mkdir(output_dir)
