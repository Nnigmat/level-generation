from typing import Dict, List
from PIL import Image
from shutil import rmtree
from os import mkdir, getcwd

# Dirs paths
output_dir = getcwd() + '/generated_images'
cells_dir = getcwd() + '/cells'

# Types
LevelType = List[List[int]]

# Cell sprites
CELL_SPRITES: Dict[int, str] = {
    0: Image.open(f'{cells_dir}/floor.png').convert('RGB'),
    1: Image.open(f'{cells_dir}/free.png').convert('RGB'),
    2: Image.open(f'{cells_dir}/wall.png').convert('RGB'),
    3: Image.open(f'{cells_dir}/start.png').convert('RGB'),
    4: Image.open(f'{cells_dir}/end.png').convert('RGB'),
}

# Sprite resolution
CELL_WIDTH: int = 32
CELL_HEIGHT: int = 67


def generate_images(levels: List[LevelType], width: int = 10, height: int = 10) -> None:
    """Generate images of levels based on the given levels
    and store them in output_dir
    """
    _clear()
    image_width, image_height = width * CELL_WIDTH, height * CELL_HEIGHT

    for i, level in enumerate(levels):
        image = Image.new('RGBA', (image_width, image_height), (0, 0, 0, 0))
        for _i, row in enumerate(level):
            for _j, cell_type in enumerate(row):
                if cell_type is None:
                    continue

                left = _j * CELL_WIDTH
                upper = _i * CELL_HEIGHT
                right = left + CELL_WIDTH
                lower = upper + CELL_HEIGHT

                image.paste(CELL_SPRITES[cell_type],
                            box=(left, upper, right, lower))
        image.save(f'{output_dir}/{i}.png')


def _clear() -> None:
    """Remove output dir and create new
    """
    rmtree(output_dir, ignore_errors=True)
    mkdir(output_dir)
