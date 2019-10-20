import os
from os.path import join
import click
from PIL import Image


@click.command()
@click.argument('renders_path')
def main(renders_path):
    subfolders = os.listdir(renders_path)
    for sf in subfolders:
        renders = os.listdir(join(renders_path, sf))
        for render in renders:
            render = join(renders_path, sf, render)
            im = Image.open(render)
            w, h = im.size
            tw, th = 400, 400
            if w == tw and h == th:
                continue

            pw = (w - tw)//2
            ph = (h - th)//2
            res = im.crop((pw, ph, pw+tw, ph+th))
            print(render)
            res.save(render)


if __name__ == '__main__':
    main()
