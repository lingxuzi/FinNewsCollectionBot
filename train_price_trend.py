from ai.vision.price_trend.train import run_training, run_eval
import yaml
import warnings
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train price trend model')
    parser.add_argument('--config', type=str, default='./ai/vision/price_trend/configs/config.yml', help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='train', help='Mode of operation: train or test')
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        if opts.mode == 'train':
            run_training(config)
        else:
            print("Running in test mode, no training will be performed.")
            run_eval(config)