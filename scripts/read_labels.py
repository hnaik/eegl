import json
from argparse import ArgumentParser
from pathlib import Path


def main(args):
    data = json.load(args.graph_input.open())
    labels = data['graph']['labels']
    with open(args.output, 'w') as f:
        f.write(','.join([str(i) for i in labels]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--graph-input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
