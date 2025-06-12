from ai.embedding.train import run_training
import yaml

if __name__ == '__main__':
    with open('./ai/embedding/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    run_training(config)