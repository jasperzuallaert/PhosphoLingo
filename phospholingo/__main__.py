"""Main command line interface to PhosphoLingo"""

# TODO copy to other
__author__ = ["Jasper Zuallaert"]
__credits__ = ["Jasper Zuallaert"] # TODO
__license__ = '' # TODO
__maintainer__ = ['Jasper Zuallaert']
__email__ = ['jasper.zuallaert@ugent.be']

import argparse
import sys
from train import run_training
from predict import run_predict
from visualize import run_visualize

def main(cl_input: list):
    """Main function for the CLI"""
    parser = argparse.ArgumentParser(prog='phospholingo')
    # parser.add_argument('func', help='the desired PhosphoLingo subprogram to run', required=True, choices=['train', 'predict', 'visualize'])
    subparsers = parser.add_subparsers(help='the desired PhosphoLingo subprogram to run', dest='prog')

    parser_train = subparsers.add_parser('train', help='train a new prediction model')
    parser_train.add_argument('json', help='the .json configuration file for training a new model', default='configs/default_config.json')

    parser_pred = subparsers.add_parser('predict', help='predict using an existing model')
    parser_pred.add_argument('model', help='the location of the saved model')
    parser_pred.add_argument('dataset', help='the dataset for which to make predictions')
    parser_pred.add_argument('out', help='the output file, will be written in a csv format')

    parser_vis = subparsers.add_parser('visualize', help='calculate SHAP values using an existing model')
    parser_vis.add_argument('model', help='the location of the saved model')
    parser_vis.add_argument('dataset', help='the dataset for which to visualize important features')
    parser_vis.add_argument('out', help='the output file, will be written in a txt format')

    args = parser.parse_args(cl_input)
    if args.prog == 'train':
        run_training(args.json)
    elif args.prog == 'predict':
        run_predict(args.model, args.dataset, args.out)
    elif args.prog == 'visualize':
        run_visualize(args.model, args.dataset, args.out)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main(sys.argv[1:])