import click
import os
import random
import numpy as np

from PIL import Image
from os.path import join
from itertools import islice
from collections import defaultdict

import pokemon_data
from config import grouped_ids


# region Utils

def random_in(a, b):
    return random.random() * (b - a) + a


def tiled_sample(sequence):
    while True:
        if len(sequence) == 0:
            yield None
        else:
            for x in random.sample(sequence, len(sequence)):
                yield x

# endregion


def load_pokemon_data():
    """
    Load the complete pokemon dataset (with info about types), filtered using the whitelist
    :return:
    """
    return [row for i, row in enumerate(pokemon_data.load()) if i+1 in whitelist_ids]


def get_crop_from_idx(renders, idx):
    """
    Generate a random crop on the fly for the pokemon with the specified idx.
    :param renders: preallocated dataset
    :param idx: pokedex number of the pokemon
    :return: random generated crop
    """
    if len(renders[idx]) == 0:
        return None

    data = random.sample(renders[idx], 1)[0]
    img: Image = data['img']
    w, h = img.size
    tw = random_in(200, 300)
    th = random_in(200, 300)
    pw = random_in(50, w - tw - 50)
    ph = random_in(50, h - th - 50)
    cropped = img.crop((pw, ph, pw+tw, ph+th))
    return np.array(cropped.resize((256, 256)))


def generator(renders):
    """
    Return an endless generator of random crops that can be iterated indefinitely.
    The result for each iteration is a dict, with pokemon types as keys and crops as values.
    The crops are numpy array with shape (256, 256, 3) and dtype np.uint8
    :param renders: dataset of renders
    :return: a random crop generator
    """
    ids_we_have = renders.keys()
    grouped_ids_we_have = {k: [i for i in ids if i in ids_we_have] for k, ids in grouped_ids.items()}

    types = [k for k, ids in grouped_ids_we_have.items() if len(ids) > 0]
    samplers = {t: tiled_sample(grouped_ids_we_have[t]) for t in types}
    ids_sampler = zip(*samplers.values())
    ids_sampler = (dict(zip(types, v)) for v in ids_sampler)

    for ids in ids_sampler:
        yield {t: get_crop_from_idx(renders, idx) for t, idx in ids.items()}


def load_renders(renders_path):
    """
    Allocate the full renders dataset into memory
    :param renders_path: path to renders
    :return: complete dataset of renders
    """
    images = defaultdict(list)

    subfolders = os.listdir(renders_path)
    for sf in subfolders:
        renders = os.listdir(join(renders_path, sf))
        for render in renders:
            render = join(renders_path, sf, render)
            img = Image.open(render)
            img.load()  # required for png.split()
            no_alpha = Image.new("RGB", img.size, (255, 255, 255))
            no_alpha.paste(img, mask=img.split()[3])  # 3 is the alpha channel

            images[int(sf)].append({'name': render, 'img': no_alpha})

    return images


def get_render_folders(renders_path):
    subfolders = os.listdir(renders_path)
    for sf in subfolders:
        renders = os.listdir(join(renders_path, sf))
        renders = [join(sf, render) for render in renders]
        yield int(sf), renders


def load_renders_iter(renders_path):
    """
    Allocate the full renders dataset into memory
    :param renders_path: path to renders
    :return: complete dataset of renders
    """
    subfolders = os.listdir(renders_path)
    for sf in subfolders:
        renders = os.listdir(join(renders_path, sf))
        for render in renders:
            render = join(renders_path, sf, render)
            img = Image.open(render)
            img.load()  # required for png.split()
            no_alpha = Image.new("RGB", img.size, (255, 255, 255))
            no_alpha.paste(img, mask=img.split()[3])  # 3 is the alpha channel

            yield int(sf), render, no_alpha


@click.command()
@click.argument('renders_path')
def test(renders_path):
    renders = load_renders(renders_path)
    gen = generator(renders)

    for vals in gen:
        for t, img in vals.items():
            im = Image.fromarray(img)
            im.show('Crop')
        input()


if __name__ == '__main__':
    test()
