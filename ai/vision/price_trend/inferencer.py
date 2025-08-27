import torch
import joblib
import os
import ai.vision.price_trend.models.base
import torch.nn.functional as F
from datetime import datetime, timedelta
from datasource.stock_basic.baostock_source import BaoSource
from ai.vision.price_trend.models import create_model, get_model_config
from ai.vision.price_trend.dataset import PriceToImgae, normalize, get_image_with_price
from torchvision import transforms

class VisionInferencer:
    def __init__(self, config):
        self.config = config
        self.load_scaler_and_encoders()
        self.load_model()
        self.source = BaoSource()
        self.transforms = transforms.Compose([
            transforms.Resize(self.config['data']['image_size']),
            transforms.ToTensor()
        ])

    def load_scaler_and_encoders(self):
        encoder_path = self.config['data']['encoder_path']
        scaler_path = self.config['data']['scaler_path']
        if os.path.exists(encoder_path):
            print("Loading precomputed encoder...")
            encoder = joblib.load(encoder_path)
        if os.path.exists(scaler_path):
            print("Loading precomputed scaler...")
            scaler = joblib.load(scaler_path)

        self.encoder = encoder
        self.scaler = scaler

    def load_model(self):
        device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        model_config = get_model_config(self.config['training']['model'])
        model_config['stock_classes'] = len(self.encoder[1].classes_)
        model_config['industry_classes'] = len(self.encoder[0].classes_)
        model_config['ts_encoder']['ts_input_dim'] = len(self.config['data']['ts_features']['features']) + len(self.config['data']['ts_features']['temporal'])
        model_config['ts_encoder']['ctx_input_dim'] = len(self.config['data']['ts_features']['numerical'])

        model = create_model(self.config['training']['model'], model_config)
        model.build_ts()
        model.build_vision()
        model.build_fusion()
        model = model.to(device)
        state_dict = torch.load(self.config['training']['model_save_path'], map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model state dict: {e}")

        model.export()

        self.model = model

    def preprocess(self, df):
        df = self.source.calculate_indicators(df)
        ts_df = normalize(df, self.config['data']['ts_features']['features'], self.config['data']['ts_features']['numerical'])
        ts_df[self.config['data']['ts_features']['features'] + self.config['data']['ts_features']['numerical']] = self.scaler.transform(ts_df[self.config['data']['ts_features']['features'] + self.config['data']['ts_features']['numerical']])
        ts_featured_stock_data = ts_df[self.config['data']['ts_features']['features'] + self.config['data']['ts_features']['temporal']].to_numpy()
        ts_numerical_stock_data = ts_df[self.config['data']['ts_features']['numerical']].to_numpy()
        price_data = df[self.config['data']['features']].to_numpy()
        price_seq = price_data[-self.config['data']['sequence_length']:]
        ts_seq = ts_featured_stock_data[-self.config['data']['sequence_length']:]
        ctx_seq = ts_numerical_stock_data[-1]
        img = get_image_with_price(price_seq)

        return img, ts_seq, ctx_seq

    def inference(self, df):
        img, ts_seq, ctx_seq = self.preprocess(df)
        # convert to tensor
        img = self.transforms(img)
        ts_seq = torch.from_numpy(ts_seq).float()
        ctx_seq = torch.from_numpy(ctx_seq).float()
        img = img.unsqueeze(0)
        ts_seq = ts_seq.unsqueeze(0)
        ctx_seq = ctx_seq.unsqueeze(0)
        device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        ts_seq = ts_seq.to(device)
        ctx_seq = ctx_seq.to(device)
        # inference
        with torch.no_grad():
            trend_logits = self.model.fuse_logits(img, ts_seq, ctx_seq)
            trend_probs = F.softmax(trend_logits['fused_trend_logits'], dim=1).cpu().numpy()
            returns = trend_logits['returns'].cpu().numpy()
            up_prob = trend_probs[0][1]
            down_prob = trend_probs[0][0]
            
        return float(up_prob), float(down_prob), float(returns)

