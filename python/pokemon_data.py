from os.path import dirname, join
import csv
import io


def load():
    src_path = dirname(__file__)
    csv_path = join(src_path, 'pokemon.csv').replace('\\', '/')
    with io.open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        csv_iter = iter(reader)
        header = next(csv_iter)
        pokemon_data = [dict(zip(header, row)) for row in csv_iter]

    return pokemon_data
