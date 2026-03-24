"""Shared argument parsing for training scripts."""

import argparse
from config.datasets import KNOWN_DATASETS


def add_dataset_args(parser: argparse.ArgumentParser, required: bool = True):
    """Add common dataset selection arguments.
    
    Args:
        parser: ArgumentParser instance to add arguments to
        required: Whether dataset argument is required
    """
    parser.add_argument(
        '--dataset',
        type=str,
        required=required,
        choices=sorted(KNOWN_DATASETS),
        help='Dataset to train on'
    )


def add_preprocessing_args(parser: argparse.ArgumentParser):
    """Add common preprocessing arguments.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        '--feature-scaler',
        type=str,
        default='robust',
        choices=['minmax', 'standard', 'robust', 'quantile'],
        help='Scaler type for features (default: robust)'
    )
    parser.add_argument(
        '--label-scaler',
        type=str,
        default='robust',
        choices=['minmax', 'standard', 'robust', 'quantile'],
        help='Scaler type for labels (default: robust)'
    )
    parser.add_argument(
        '--use-log',
        action='store_true',
        default=True,
        help='Apply log1p transformation to labels (default: True)'
    )
    parser.add_argument(
        '--no-log',
        action='store_false',
        dest='use_log',
        help='Disable log1p transformation of labels'
    )
    parser.add_argument(
        '--use-area-root',
        action='store_true',
        default=False,
        help='Apply square root to Area label before scaling (isotherm only)'
    )


def add_device_args(parser: argparse.ArgumentParser):
    """Add device (CPU/GPU) arguments.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA (use CPU only)'
    )


def add_output_args(parser: argparse.ArgumentParser):
    """Add output directory arguments.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated in runs/)'
    )


def add_random_seed_args(parser: argparse.ArgumentParser):
    """Add random seed arguments for reproducibility.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=1,
        help='Number of seeds to run for mean±std (default: 1)'
    )
