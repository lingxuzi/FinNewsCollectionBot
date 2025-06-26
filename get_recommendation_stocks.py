from recommendation.manager import StockRecommendationManager
import yaml
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Qurey embedding model')
    parser.add_argument('--config', type=str, default='./ai/embedding/search/config.yml', help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    opts = parse_args()
    # Load configuration from YAML file
    with open(opts.config, 'r') as f:
        config = yaml.safe_load(f)
        manager = StockRecommendationManager(config)
        results = manager.get_recommendation_stocks(with_klines=True)
        print(results)