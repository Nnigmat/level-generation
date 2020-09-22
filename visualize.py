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

CELL_WIDTH: int = 32
CELL_HEIGHT: int = 67


def generate_images(levels: List[LevelType], width: int = 10, height: int = 10, amount: int = 10) -> None:
    """Generate images of levels based on the given levels
    and store them in output_dir
    """
    _clear()
    image_width, image_height = width * CELL_WIDTH, height * CELL_HEIGHT

    res: List[Image] = []
    for i in range(amount):
        image = Image.new('RGB', (image_width, image_height))
        res.append()


def _clear() -> None:
    """Remove output dir and create new
    """
    rmtree(output_dir)
    mkdir(output_dir)
