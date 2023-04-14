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
    description='Prepare molecular data for the network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog='''This script reads the structures of ligands and pocket(s),
    prepares them for the neural network and saves in a HDF file.
    It also saves affinity values as attributes, if they are provided.
    You can either specify a separate pocket for each ligand or a single
    pocket that will be used for all ligands. We assume that your structures
    are fully prepared.\n\n

    Note that this scripts produces standard data representation for our network
    and saves all required data to predict affinity for each molecular complex.
    If some part of your data can be shared between multiple complexes
    (e.g. you use a single structure for the pocket), you can store the data
    more efficiently. To prepare the data manually use functions defined in
    tfbio.data module.
    '''
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
