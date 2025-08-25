# train.py
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import joblib
import pandas as pd
import copy
import random
import ai.vision.price_trend.models.base
from autoclip.torch import QuantileClip
from torch.utils.data import DataLoader, random_split
from ai.optimizer.muon import MuonClip
from ai.vision.price_trend.dataset import ImagingPriceTrendDataset, normalize
from ai.vision.price_trend.sampler.trend_sampler import TrendSampler
from ai.vision.price_trend.models import create_model, get_model_config
from tqdm import tqdm # 提供优雅的进度条
from utils.common import AverageMeter
from config.base import *
from ai.modules.multiloss import AutomaticWeightedLoss
from ai.modules.earlystop import EarlyStopping
from ai.vision.gradcam.gradcam import GradCAM, save_cam_on_image
from ai.embedding.train import HuberTrendLoss
from ai.optimizer import *
from ai.scheduler.sched import *
from ai.metrics import *
from ai.logs.log_agent import LogAgent
from utils.prefetcher import DataPrefetcher
from utils.common import ModelEmaV2
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def num_iters_per_epoch(loader, batch_size):
    return len(loader) // batch_size

def generate_encoder(hist_data_files, ts_features, ts_numerical, categorical):
    hist_data = []
    for hist_data_file in hist_data_files:
        df = pd.read_parquet(hist_data_file)
        df = normalize(df, ts_features, ts_numerical)
        hist_data.append(df)
    
    df = pd.concat(hist_data)

    scaler = StandardScaler()
    scaler.fit_transform(df[ts_features + ts_numerical])
    
    indus_encoder = LabelEncoder()
    indus_encoder.fit_transform(df[categorical[0]])

    code_encoder = LabelEncoder()
    code_encoder.fit_transform(df[categorical[1]])
    

    return (indus_encoder, code_encoder), scaler

def run_training(config):
    # set random seed
    """主训练函数"""
    # --- 1. 加载配置 ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder_path = config['data']['encoder_path']
    scaler_path = config['data']['scaler_path']
    if os.path.exists(encoder_path):
        print("Loading precomputed encoder...")
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
    else:
        encoder, scaler = generate_encoder(
        [
            config['data']['train']['hist_data_file'],
            config['data']['eval']['hist_data_file'],
            config['data']['test']['hist_data_file']
        ], config['data']['ts_features']['features'], config['data']['ts_features']['numerical'], config['data']['categorical'])
        joblib.dump(encoder, encoder_path)
        joblib.dump(scaler, scaler_path)

    os.makedirs(os.path.split(config['training']['model_save_path'])[0], exist_ok=True)

    # --- 2. 准备数据 ---
    if not config['training']['finetune']:
        train_dataset = ImagingPriceTrendDataset(
            db_path=config['data']['db_path'],
            img_caching_path=config['data']['train']['img_caching_path'],
            stock_list_file=config['data']['train']['stock_list_file'],
            hist_data_file=config['data']['train']['hist_data_file'],
            seq_length=config['data']['sequence_length'],
            features=config['data']['features'],
            ts_features=config['data']['ts_features'],
            image_size=config['data']['image_size'],
            encoder=encoder,
            scaler=scaler,
            tag='train'
        )
        eval_dataset = ImagingPriceTrendDataset(
            db_path=config['data']['db_path'],
            img_caching_path=config['data']['test']['img_caching_path'],
            stock_list_file=config['data']['test']['stock_list_file'],
            hist_data_file=config['data']['test']['hist_data_file'],
            seq_length=config['data']['sequence_length'],
            features=config['data']['features'],
            ts_features=config['data']['ts_features'],
            image_size=config['data']['image_size'],
            encoder=encoder,
            scaler=scaler,
            tag='test',
            is_train=False
        )
    else:
        train_dataset = ImagingPriceTrendDataset(
            db_path=config['data']['db_path'],
            img_caching_path=config['data']['eval']['img_caching_path'],
            stock_list_file=config['data']['eval']['stock_list_file'],
            hist_data_file=config['data']['eval']['hist_data_file'],
            seq_length=config['data']['sequence_length'],
            features=config['data']['features'],
            ts_features=config['data']['ts_features'],
            image_size=config['data']['image_size'],
            encoder=encoder,
            scaler=scaler,
            tag='eval',
            is_train=False
        )

        eval_dataset = ImagingPriceTrendDataset(
            db_path=config['data']['db_path'],
            img_caching_path=config['data']['test']['img_caching_path'],
            stock_list_file=config['data']['test']['stock_list_file'],
            hist_data_file=config['data']['test']['hist_data_file'],
            seq_length=config['data']['sequence_length'],
            features=config['data']['features'],
            ts_features=config['data']['ts_features'],
            image_size=config['data']['image_size'],
            encoder=encoder,
            scaler=scaler,
            tag='test',
            is_train=False
        )

    if config['data']['sampler']:
        train_sampler = TrendSampler(train_dataset, config['training']['batch_size'], tag='train' if not config['training']['finetune'] else 'finetune')
        train_loader = DataLoader(train_dataset, num_workers=config['training']['workers'], pin_memory=False, shuffle=False, drop_last=False, batch_sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=True, drop_last=True)

    print(f"Training data size: {len(train_dataset)}, Validation data size: {len(eval_dataset)}")
    if config['training']['awl']:
        awl = AutomaticWeightedLoss(len(config['training']['losses']) + 1)
        awl.to(device)

    # --- 3. 初始化模型、损失函数和优化器 ---

    model_config = get_model_config(config['training']['model'])
    model_config['stock_classes'] = len(encoder[1].classes_)
    model_config['industry_classes'] = len(encoder[0].classes_)
    model_config['ts_encoder']['ts_input_dim'] = len(config['data']['ts_features']['features']) + len(config['data']['ts_features']['temporal'])
    model_config['ts_encoder']['ctx_input_dim'] = len(config['data']['ts_features']['numerical'])
    model = create_model(config['training']['model'], model_config)

    log_agent = LogAgent(config['training']['model'], {
        'basic': config,
        'model': model_config
    }, name='vision_price_trend')

    if config['training']['load_pretrained']:
        try:
            state_dict = torch.load(config['training']['pretrained_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            print('pretrain loaded')
        except Exception as e:
            print(f'Load pretrained model failed: {e}')

    if config['training']['finetune']:
        try:
            state_dict = torch.load(config['training']['model_save_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            print('finetune model loaded')
        except Exception as e:
            print(f'Load finetune model failed: {e}')

    ema = ModelEmaV2(model, decay=0.9999, device=device)

    model = model.to(device)

    if config['training']['freeze'] == 'ts':
        model.freeze_ts()
    elif config['training']['freeze'] == 'vision':
        model.freeze_vision()
    elif config['training']['freeze'] == 'backbone':
        model.freeze_backbone()
    elif config['training']['freeze'] == 'all':
        model.freeze_vision()
        model.freeze_ts()

    criterion_trend = nn.CrossEntropyLoss() #ASLSingleLabel() #focal_loss(alpha=0.3, gamma=2, num_classes=model_config['trend_classes'])
    criterion_stock = nn.CrossEntropyLoss() #ASLSingleLabel()
    criterion_industry = nn.CrossEntropyLoss() #ASLSingleLabel()
    criterion_return = nn.HuberLoss(delta=0.1) #HuberTrendLoss(tildeq=True)

    parameters = []
    if config['training']['awl']:
        parameters += [{'params': awl.parameters(), 'weight_decay': 0, 'lr': 1e-2}]
    parameters += [{'params': [p for p in model.parameters() if p.requires_grad], 'weight_decay': config['training']['weight_decay'], 'lr': config['training']['min_learning_rate'] if config['training']['warmup_epochs'] > 0 else config['training']['learning_rate']}]
    # print(list(model.named_parameters()))
    if config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, momentum=0.9, nesterov=True)
    elif config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters)
    elif config['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(parameters)
        
    if config['training']['clip_norm'] == 0.01:
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=0.9, history_length=1000)
    early_stopper = EarlyStopping(patience=40, direction='up')
    
    if config['data']['sampler']:
        train_iter = DataPrefetcher(train_loader, config['device'], enable_queue=False, num_threads=1)
    scheduler = CosineWarmupLR(
        optimizer, config['training']['num_epochs'], config['training']['learning_rate'], config['training']['min_learning_rate'], warmup_epochs=config['training']['warmup_epochs'], warmup_lr=config['training']['min_learning_rate'])

    # --- 4. 训练循环 ---
    best_val_loss = float('inf') if early_stopper.direction == 'down' else -float('inf')
    for epoch in range(config['training']['num_epochs']):

        generate_gradcam(model, eval_dataset)

        if not config['data']['sampler']:
            train_iter = DataPrefetcher(train_loader, config['device'], enable_queue=False, num_threads=1)
        model.train()
        train_loss_meter = AverageMeter()
        trend_loss_meter = AverageMeter()
        stock_loss_meter = AverageMeter()
        industry_loss_meter = AverageMeter()
        returns_loss_meter = AverageMeter()

        trend_metric_meter = AverageMeter()
        stock_metric_meter = AverageMeter()
        industry_metric_meter = AverageMeter()
        returns_metric_meter = AverageMeter()

        # 使用tqdm显示进度条
        pbar = tqdm(range(num_iters_per_epoch(train_dataset if train_sampler is None else train_sampler, config['training']['batch_size'])), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training]")
        for _ in pbar:
            img, trend, returns, stock, industry, ts, ctx = train_iter.next()
            optimizer.zero_grad()
            trend_pred, ts_pred, trend_pred_fused, stock_pred, industry_pred, returns_pred = model(img, ts, ctx)

            losses = {}

            if 'trend' in config['training']['losses']:
                loss_trend = criterion_trend(trend_pred, trend.squeeze())
                loss_trend_fused = criterion_trend(trend_pred_fused, trend.squeeze())
                loss_ts = criterion_trend(ts_pred, trend.squeeze())
                losses['trend'] = (loss_trend + loss_trend_fused + loss_ts) / 3
                trend_loss_meter.update(losses['trend'].item())
                trend_metric_meter.update(balanced_accuracy_score(trend.squeeze().cpu().numpy(), trend_pred_fused.argmax(axis=1).cpu().numpy()))
            
            if 'stock' in config['training']['losses']:
                loss_stock = criterion_stock(stock_pred, stock.squeeze())
                losses['stock'] = loss_stock
                stock_loss_meter.update(loss_stock.item())
                stock_metric_meter.update(balanced_accuracy_score(stock.squeeze().cpu().numpy(), trend_pred.argmax(axis=1).cpu().numpy()))
            
            if 'industry' in config['training']['losses']:
                loss_industry = criterion_industry(industry_pred, industry.squeeze())
                losses['industry'] = loss_industry
                industry_loss_meter.update(loss_industry.item())
                industry_metric_meter.update(balanced_accuracy_score(industry.squeeze().cpu().numpy(), trend_pred.argmax(axis=1).cpu().numpy()))
            
            if 'returns' in config['training']['losses']:
                loss_returns = criterion_return(returns_pred, returns.squeeze())
                losses['returns'] = loss_returns
                returns_loss_meter.update(loss_returns.item())
                returns_metric_meter.update(r2_score(returns.squeeze().cpu().numpy(), returns_pred.detach().squeeze().cpu().numpy()))
            
            if config['training']['awl']:
                losses = list(losses.values())
                total_loss = awl(*losses)
            else:
                total_loss = sum([losses[lk] * w for w, lk in zip(config['training']['loss_weights'], config['training']['losses'])])
            total_loss.backward()

            clip_norm = config['training']['clip_norm']
            if clip_norm > 0.01:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            ema.update(model)

            train_loss_meter.update(total_loss.item())
            #| Pred Loss: {pred_loss_meter.avg}
            pbar.set_description(f"Total({epoch+1}/{config['training']['num_epochs']}): {train_loss_meter.avg:.4f} | Trend: {trend_loss_meter.avg:.4f} | Trend Metric: {trend_metric_meter.avg:.4f} | Stock: {stock_loss_meter.avg:.4f} | Industry: {industry_loss_meter.avg:.4f} | Returns: {returns_loss_meter.avg:.4f} | Returns Metric: {returns_metric_meter.avg:.4f}")
        
        scheduler.step()

        # --- 5. 验证循环 ---
        _model = copy.deepcopy(ema.module)
        mean_r2 = eval(_model, eval_dataset, config, log_agent)
        
        # --- 6. 保存最佳模型 ---
        # 只保存性能最好的模型，避免存储过多文件
        if mean_r2 > best_val_loss:
            best_val_loss = mean_r2
            # 我们只关心编码器的权重，也可以保存整个模型
            torch.save(_model.state_dict(), config['training']['model_save_path'])
            # torch.save({
            #     'encoder_state_dict': model.encoder.state_dict(),
            #     'encoder_fc_state_dict': model.encoder_fc.state_dict(),
            #     'config': config # 将配置也一并保存，便于推理时加载
            # }, config['training']['model_save_path'])
            print(f"Model improved and saved to {config['training']['model_save_path']}")

        early_stopper(mean_r2)
        if early_stopper.early_stop:
            break


def generate_gradcam(model, dataset):
    model.eval()

    indices = random.sample(range(len(dataset)), 10)

    os.makedirs('../stock_gradcams', exist_ok=True)

    for i in indices:
        img, trend, returns, stock, industry, ts, ctx = dataset[i]
        img = img.unsqueeze(0).cuda()
        stock = stock.cuda()
        industry = industry.cuda()
        ts = ts.cuda()
        ctx = ctx.cuda()
        img.requires_grad_()
        target_layer = model.gradcam_layer()
        gradcam = GradCAM(model=model, target_layer=target_layer, image_shape=img.shape[2:], forward_callback=gradcam_forward)
        cam = gradcam(input_tensor=(img, stock, industry, None, None))

        save_cam_on_image(img.detach().squeeze().cpu().numpy(), cam.squeeze(), f'../stock_gradcams/gradcam_{i}.png')

def gradcam_forward(input_tensor, model):
    img, stock, industry, ts, ctx = input_tensor
    trend_pred, ts_pred, trend_pred_fused, stock_pred, industry_pred, returns = model(img, None, None)
    return trend_pred

def eval(model, dataset, config, log_agent):
    _model = model#copy.deepcopy(ema.module)
    _model.eval()

    val_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=False, drop_last=True)
    with torch.no_grad():
        val_iter = DataPrefetcher(val_loader, config['device'], enable_queue=False, num_threads=1)
        trend_metric = ClsMetric('trend')
        ts_metric = ClsMetric('ts')
        vision_metric = ClsMetric('vision_trend')
        return_metric = Metric('returns')

        for _ in tqdm(range(num_iters_per_epoch(dataset, config['training']['batch_size'])), desc="[Validation]"):
            # ts_sequences = ts_sequences.to(device)
            # ctx_sequences = ctx_sequences.to(device)
            # y = y.to(device)
            img, trend, returns, stock, industry, ts, ctx = val_iter.next()

            trend_pred, ts_pred, trend_pred_fused, stock_pred, industry_pred, returns_pred = _model(img, ts, ctx)

            trend_metric.update(trend.squeeze().cpu().numpy(), trend_pred_fused.cpu().numpy())
            ts_metric.update(trend.squeeze().cpu().numpy(), ts_pred.cpu().numpy())
            vision_metric.update(trend.squeeze().cpu().numpy(), trend_pred.cpu().numpy())
            return_metric.update(returns.squeeze().cpu().numpy(), returns_pred.cpu().numpy())
    
    # --- 计算整体 R² --

    scores = []
    _, trend_score = trend_metric.calculate()
    _, ts_score = ts_metric.calculate()
    _, vision_score = vision_metric.calculate()
    _, return_score = return_metric.calculate()
    if config['training']['freeze'] == 'vision':
        scores.append(ts_score)
    elif config['training']['freeze'] == 'ts':
        scores.append(vision_score)
    else:
        scores.append(trend_score)

    log_agent.log({
        'eval_trend_score': trend_score,
        'eval_ts_score': ts_score,
        'eval_vision_score': vision_score,
        'eval_return_score': return_score
    })

    mean_r2 = sum(scores) / len(scores)

    return mean_r2

def run_eval(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    encoder_path = config['data']['encoder_path']
    scaler_path = config['data']['scaler_path']
    if os.path.exists(encoder_path):
        print("Loading precomputed encoder...")
        encoder = joblib.load(encoder_path)
    if os.path.exists(scaler_path):
        print("Loading precomputed scaler...")
        scaler = joblib.load(scaler_path)

    test_dataset = ImagingPriceTrendDataset(
        db_path=config['data']['db_path'],
        img_caching_path=config['data']['test']['img_caching_path'],
        stock_list_file=config['data']['test']['stock_list_file'],
        hist_data_file=config['data']['test']['hist_data_file'],
        seq_length=config['data']['sequence_length'],
        features=config['data']['features'],
        ts_features=config['data']['ts_features'],
        image_size=config['data']['image_size'],
        encoder=encoder,
        scaler=scaler,
        tag='test',
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=False)
    
    model_config = get_model_config(config['training']['model'])
    model_config['stock_classes'] = len(encoder[1].classes_)
    model_config['industry_classes'] = len(encoder[0].classes_)
    model_config['ts_encoder']['ts_input_dim'] = len(config['data']['ts_features']['features']) + len(config['data']['ts_features']['temporal'])
    model_config['ts_encoder']['ctx_input_dim'] = len(config['data']['ts_features']['numerical'])

    model = create_model(config['training']['model'], model_config).to(device)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(config['training']['model_save_path'], map_location=device)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading model state dict: {e}")

    model.eval()

    trend_metric = ClsMetric('trend')
    ts_metric = ClsMetric('ts')
    vision_metric = ClsMetric('vision_trend')
    return_metric = Metric('returns')
    
    with torch.no_grad():
        test_iter = DataPrefetcher(test_loader, config['device'], enable_queue=False, num_threads=1)
        for _ in tqdm(range(num_iters_per_epoch(test_dataset, config['training']['batch_size'])), desc=f"[Testing]"):
            # ts_sequences = ts_sequences.to(device)
            # ctx_sequences = ctx_sequences.to(device)
            # y = y.to(device)

            img, trend, returns, stock, industry, ts, ctx = test_iter.next()

            trend_pred, ts_pred, trend_pred_fused, stock_pred, industry_pred, returns_pred = model(img, ts, ctx)
            trend_metric.update(trend.squeeze().cpu().numpy(), trend_pred_fused.cpu().numpy())
            ts_metric.update(trend.squeeze().cpu().numpy(), ts_pred.cpu().numpy())
            vision_metric.update(trend.squeeze().cpu().numpy(), trend_pred.cpu().numpy())
            return_metric.update(returns.squeeze().cpu().numpy(), returns_pred.cpu().numpy())

        # --- 计算整体 R² ---
        _, trend_score = trend_metric.calculate()
        _, ts_score = ts_metric.calculate()
        _, vision_score = vision_metric.calculate()
        _, return_score = return_metric.calculate()
