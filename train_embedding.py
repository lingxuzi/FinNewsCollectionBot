from ai.embedding.train import run_training
import yaml

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train embedding model')
    parser.add_argument('--config', type=str, default='./ai/embedding/config.yml', help='Path to the configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        run_training(config)