from ai.embedding.search.search_embedding import EmbeddingQueryer
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Qurey embedding model')
    parser.add_argument('--config', type=str, default='./ai/embedding/search/config.yml', help='Path to the configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        queryer = EmbeddingQueryer(config)

        queryer.query('600318')
