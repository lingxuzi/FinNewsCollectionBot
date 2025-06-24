from backtrade.runner import do_backtrade
import yaml
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Qurey embedding model')
    parser.add_argument('--config', type=str, default='./backtrade/config/runner.yml', help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        do_backtrade(config)
