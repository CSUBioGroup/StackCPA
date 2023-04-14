import os
from model import CpaStacking
import pandas as pd

def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path


def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path

import argparse
parser = argparse.ArgumentParser(
    description='prediction',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--data', '-d', required=True, type=input_file, nargs='+',
                    help='data for prediction')
parser.add_argument('--model', '-m', default='../saved_model/model.pkl',
                    type=input_file,
                    help='model for prediction')
parser.add_argument('--output','-o',default='../predict/res.txt',
                    type=output_file,help='saving path for prediction')
args = parser.parse_args()


model = CpaStacking.load_model(file="../saved_model/pdbbind_save_model.pkl")
res = model.predict(args.data)

with open(args.output,'w') as file:
    file.write("\n".join(res))
