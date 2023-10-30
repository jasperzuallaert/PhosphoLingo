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
from visualize_shuffling import run_visualize


def main():
    """Main function for the CLI"""
    print(sys.argv[1:])
    parser = argparse.ArgumentParser(prog='phospholingo')
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
    parser_vis.add_argument('out_values', help='the output SHAP scores file, will be written in a txt format')
    parser_vis.add_argument('out_img', help='the normalized average SHAP scores per position, as an image file')

    args = parser.parse_args()
    if args.prog == 'train':
        run_training(args.json)
    elif args.prog == 'predict':
        run_predict(args.model, args.dataset, args.out)
    elif args.prog == 'visualize':
        run_visualize(args.model, args.dataset, args.out_values, args.out_img)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()