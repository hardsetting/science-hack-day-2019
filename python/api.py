from flask import Flask, send_file, request
from flask_restful import Resource, Api

import click
from os.path import join
from PIL import Image
from pathlib import Path

from dataset import get_render_folders


app = Flask(__name__)
api = Api(app)


type_map = {
    'grass': 1,
    'water': 2
}


def pokemon_url(idx):
    base_url = request.url_root
    return join(base_url, 'static', f'{idx:03d}', f'{idx:03d}_1.png')


@click.command()
@click.argument('renders_path')
def main(renders_path):
    grouped_renders = get_render_folders(renders_path)
    grouped_first_renders = {idx: renders[0] for idx, renders in grouped_renders}

    class PokemonList(Resource):
        def get(self):
            return {idx: pokemon_url(idx) for idx, rel_path in grouped_first_renders.items()}

    class Pokemon(Resource):
        def get(self, idx):
            fh = open(join(renders_path, grouped_first_renders[idx]), 'rb')
            return send_file(fh, as_attachment=False,
                             attachment_filename='pokemon.png',
                             mimetype='image/png')

    class GeneratedPokemon(Resource):
        def get(self, idx, type_idx):
            fh = open(join(renders_path, grouped_first_renders[idx]), 'rb')
            return send_file(fh, as_attachment=False,
                             attachment_filename='pokemon.png',
                             mimetype='image/png')

    api.add_resource(PokemonList, '/pokemon/')
    api.add_resource(Pokemon, '/pokemon/<int:idx>')
    api.add_resource(GeneratedPokemon, '/pokemon/<int:idx>/as_type/<type_idx>')
    app.run(port=5000)


if __name__ == '__main__':
    main()
