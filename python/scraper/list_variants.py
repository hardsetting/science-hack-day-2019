import click
from os import listdir
from os.path import join
from glob import glob

from config import grouped_ids


def get_variants(folder):
    return glob(folder + "/**/*.FBX", recursive=True) + glob(folder + "/**/*.DAE", recursive=True)


def pick_variant(idx, variants):
    if len(variants) == 1:
        return variants[0]

    is_mega = lambda n: 'Mega' in n and '\\Mega' not in n
    is_collada = lambda n: 'Collada' in n
    is_female = lambda n: n[-5] == 'F' or n[-5] == 'f'
    is_male = lambda n: n[-5] == 'M' or n[-5] == 'm'
    has_genders = any([is_male(v) for v in variants]) and any([is_female(v) for v in variants])

    is_fbx = lambda n: n[-3:] == 'FBX' or n[-3:] == 'fbx'
    is_dae = lambda n: n[-3:] == 'DAE' or n[-3:] == 'dae'

    if len(variants) > 1:
        filtered_variants = [v for v in variants if not is_mega(v) and not (has_genders and is_female(v)) and is_dae(v) and not is_collada(v)]
        if len(filtered_variants) == 1:
            return filtered_variants[0]

    if len(variants) > 1:
        filtered_variants = [v for v in variants if not is_mega(v) and not (has_genders and is_female(v)) and is_fbx(v)]
        if len(filtered_variants) == 1:
            return filtered_variants[0]

    filtered_variants = [v for v in variants if not is_mega(v) and not is_collada(v)]
    # return filtered_variants
    return filtered_variants[0]


@click.command()
@click.argument('data_path', type=click.Path(file_okay=False, exists=True), default='pokemon')
def list_variants(data_path):
    folders = listdir(data_path)
    whitelisted = [f'{idx:03d}' for t, ids in grouped_ids.items() for idx in ids]

    pairs = []
    for idx in folders:
        if idx not in whitelisted:
            continue

        variants = get_variants(join(data_path, idx))
        variant = pick_variant(idx, variants)
        pairs.append((idx, variant))

    with open('model_paths.txt', 'w') as fh:
        for idx, path in pairs:
            fh.write(f'{idx} "{path}"\n')


if __name__ == '__main__':
    list_variants()
