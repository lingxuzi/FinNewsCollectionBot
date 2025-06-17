from ai.embedding.generate.generate_embedding import run as generate
import yaml

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train embedding model')
    parser.add_argument('--config', type=str, default='./ai/embedding/generate/config.yml', help='Path to the configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        generate(config)