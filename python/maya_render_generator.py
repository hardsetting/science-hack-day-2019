import maya.cmds as cmd
# import maya.mel.eval as eval_mel
from maya.mel import eval as eval_mel
from random import random
import math
from uuid import uuid4
from os.path import dirname, basename, join, splitext, exists
from os import makedirs, rename, unlink
import time
import csv
from itertools import islice


repo_path = 'C:\\Users\\gabriele\\workspace\\sfhack19\\python'

with open(join(repo_path, 'pokemon.csv')) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    csv_iter = iter(reader)
    header = next(csv_iter)
    pokemon_data = [dict(zip(header, row)) for row in csv_iter]
    # pokemon_data = {row['name']: row for row in pokemon_data}


def load_paths():
    with open(join(repo_path, 'scraper\\model_paths.txt')) as fh:
        lines = fh.readlines()

    pairs = [(l[:3], join(repo_path, 'scraper', l.rstrip()[5:-1])) for l in lines]
    return pairs


def clear_scene():
    cmd.file(newFile=True, save=False, force=True)


def import_model(model_path):
    model_path = model_path.replace('\\', '/')
    imp_cmd = 'file -import -ignoreVersion -ra true -mergeNamespacesOnClash false -pr  -importFrameRate true  "' + model_path + '";'
    eval_mel(imp_cmd)


def setup_renderer():
    cmd.setAttr("defaultRenderGlobals.imageFormat", 32)


def random_in(a, b):
    return random() * (b - a) + a

def polar_to_cart(rad, azim, elev):
    x = rad * math.cos(elev) * math.cos(azim)
    z = rad * math.cos(elev) * math.sin(azim)
    y = rad * math.sin(elev)
    return x, y, z

def random_in_sphere(rad):
    azim = random_in(0, 2 * math.pi)
    elev = random_in(-math.pi / 3, 2 * math.pi / 3)
    # elev = 0

    x = rad * math.cos(elev) * math.cos(azim)
    y = rad * math.cos(elev) * math.sin(azim)
    z = rad * math.sin(elev)

    return x, z, y


def render(cam, dst_path, width, height):
    tmp_path = cmd.render(cam, xresolution=width, yresolution=height)
    ext = splitext(tmp_path)

    parent_folder = dirname(dst_path)
    if not exists(parent_folder):
        makedirs(parent_folder)

    if exists(dst_path):
        unlink(dst_path)

    rename(tmp_path, dst_path)
    print('Saved to ' + dst_path)


def main():
    output_folder = 'C:\\Users\\gabriele\\Desktop\\renders'
    path_iterator = load_paths()
    path_iterator = islice(path_iterator, 0, 300)

    for idx, model_path in path_iterator:
        # model_path = "C:/Users/gabriele/Downloads/3DS - Pokemon Omega Ruby Alpha Sapphire - 018 Pidgeot/Pokemon XY/Pidgeot/Pidgeot.FBX"
        # data = pokemon_data['Pidgeot']

        # model_path = "C:/Users/gabriele/Downloads/3DS - Pokemon X Y - 487 Giratina/Pokemon XY/Giratina/Giratina.FBX"
        # data = pokemon_data['Giratina']

        # model_path = "C:/Users/gabriele/Downloads/3DS - Pokemon X Y - 487 Giratina/Pokemon XY/Giratina/Giratina.FBX"
        # data = pokemon_data['Giratina']

        # height = float(data['height_m'])
        # print(height)

        render_folder = join(output_folder, idx)
        if exists(render_folder):
            print('skipping ' + render_folder + ' as it already exists.')
            continue

        print('processing ' + model_path)

        n_idx = int(idx)
        data = pokemon_data[n_idx]
        name = data['name']
        #try:
        #    height = float(data['height_m'])
        #except ValueError as ex:
        #    print('Error parsing height ' + data['height_m'])
        #    continue
            
        print(data)

        off_per_m = 200 / 1.5
        rad_per_m = 5000 / 1.5
        # rad = rad_per_m * height
        # off = off_per_m * height
        rad = 1000
        off = 0

        clear_scene()
        setup_renderer()
        import_model(model_path)

        cam, cam_shape = cmd.camera(name='render_camera')
        print('Created camera ' + cam)

        cmd.select(cam)

        for i in range(40):
            x, y, z = random_in_sphere(rad)
            cmd.move(x, y, z)
            cmd.viewLookAt(cam, pos=(0, off, 0))
            print(x, y, z, off)
            cmd.viewFit(cam)
            # cmd.panZoom(cam_shape, zoomRatio=.1)

            dst_path = join(render_folder, idx + '_' + str(i) + '.png')
            render(cam, dst_path, 1000, 1000)


if __name__ == '__main__':
    main()
