# train.py
import yaml
import torch
import torch.nn as nn
import os
import joblib
import pandas as pd
import copy
import random
import ai.vision.price_trend.models.base
from ai.vision.price_trend.dataset import ImagingPriceTrendDataset
from ai.vision.price_trend.sampler.trend_sampler import TrendSampler
from autoclip.torch import QuantileClip
from ai.vision.price_trend.models import create_model, get_model_config
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # 提供优雅的进度条
from utils.common import AverageMeter
from config.base import *
from ai.modules.multiloss import AutomaticWeightedLoss
from ai.modules.earlystop import EarlyStopping
from ai.loss.focalloss import ASLSingleLabel
from ai.optimizer.tiger import Tiger
from ai.scheduler.sched import *
from ai.metrics import *
from utils.prefetcher import DataPrefetcher
from utils.common import ModelEmaV2
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder


def num_iters_per_epoch(loader, batch_size):
    return len(loader) // batch_size

def generate_encoder(hist_data_files, categorical):
    hist_data = []
    for hist_data_file in hist_data_files:
        df = pd.read_parquet(hist_data_file)
        hist_data.append(df)
    
    df = pd.concat(hist_data)
    
    indus_encoder = LabelEncoder()
    indus_encoder.fit_transform(df[categorical[0]])

    code_encoder = LabelEncoder()
    code_encoder.fit_transform(df[categorical[1]])
    

    return (indus_encoder, code_encoder)

def run_training(config):
    # set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    """主训练函数"""
    # --- 1. 加载配置 ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder_path = config['data']['encoder_path']
    if os.path.exists(encoder_path):
        print("Loading precomputed encoder...")
        encoder = joblib.load(encoder_path)
    else:
        encoder = generate_encoder(
        [
            config['data']['train']['hist_data_file'],
            config['data']['eval']['hist_data_file'],
            config['data']['test']['hist_data_file']
        ], config['data']['categorical'])
        joblib.dump(encoder, encoder_path)
        print(f"Scaler and encoder saved to {encoder_path}")

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
            image_size=config['data']['image_size'],
            encoder=encoder,
            tag='train'
        )
        eval_dataset = ImagingPriceTrendDataset(
            db_path=config['data']['db_path'],
            img_caching_path=config['data']['eval']['img_caching_path'],
            stock_list_file=config['data']['eval']['stock_list_file'],
            hist_data_file=config['data']['eval']['hist_data_file'],
            seq_length=config['data']['sequence_length'],
            features=config['data']['features'],
            image_size=config['data']['image_size'],
            encoder=encoder,
            tag='eval',
            is_train=False
        )

    if config['data']['sampler']:
        train_sampler = TrendSampler(train_dataset, config['training']['batch_size'])
        train_loader = DataLoader(train_dataset, num_workers=config['training']['workers'], pin_memory=False, shuffle=False, drop_last=False, batch_sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=True, drop_last=True)

    print(f"Training data size: {len(train_dataset)}, Validation data size: {len(eval_dataset)}")
    if config['training']['awl']:
        awl = AutomaticWeightedLoss(len(config['training']['losses']) + 1)
        awl.to(device)

    # --- 3. 初始化模型、损失函数和优化器 ---

    model_config = get_model_config(config['training']['model'])
    model_config['stock_classes'] = len(encoder[1].classes_)
    model_config['industry_classes'] = len(encoder[0].classes_)
    model = create_model(config['training']['model'], model_config)

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
            model.load_state_dict(state_dict, strict=False)
            print('finetune model loaded')
        except Exception as e:
            print(f'Load finetune model failed: {e}')

    ema = ModelEmaV2(model, decay=0.9999, device=device)

    model = model.to(device)
    criterion_trend = nn.CrossEntropyLoss() #ASLSingleLabel() #focal_loss(alpha=0.3, gamma=2, num_classes=model_config['trend_classes'])
    criterion_stock = nn.CrossEntropyLoss()#ASLSingleLabel()#focal_loss(alpha=0.3, gamma=2, num_classes=model_config['stock_classes'])
    criterion_industry = nn.CrossEntropyLoss()#ASLSingleLabel()#focal_loss(alpha=0.3, gamma=2, num_classes=model_config['industry_classes'])

    parameters = []
    if config['training']['awl']:
        parameters += [{'params': awl.parameters(), 'weight_decay': 0, 'lr': 1e-2}]
    parameters += [{'params': model.parameters(), 'weight_decay': config['training']['weight_decay'], 'lr': config['training']['min_learning_rate'] if config['training']['warmup_epochs'] > 0 else config['training']['learning_rate']}]
    # print(list(model.named_parameters()))
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
        if not config['data']['sampler']:
            train_iter = DataPrefetcher(train_loader, config['device'], enable_queue=False, num_threads=1)
        model.train()
        train_loss_meter = AverageMeter()
        trend_loss_meter = AverageMeter()
        stock_loss_meter = AverageMeter()
        industry_loss_meter = AverageMeter()

        trend_metric_meter = AverageMeter()

        # 使用tqdm显示进度条
        pbar = tqdm(range(num_iters_per_epoch(train_dataset, config['training']['batch_size'])), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training]")
        for _ in pbar:
            img, trend, stock, industry = train_iter.next()
            optimizer.zero_grad()
            trend_pred, stock_pred, industry_pred = model(img)

            losses = {}

            if 'trend' in config['training']['losses']:
                loss_trend = criterion_trend(trend_pred, trend.squeeze())
                losses['trend'] = loss_trend
                trend_loss_meter.update(loss_trend.item())

                trend_metric_meter.update(balanced_accuracy_score(trend.squeeze().cpu().numpy(), trend_pred.argmax(axis=1).cpu().numpy()))

            if 'stock' in config['training']['losses']:
                loss_stock = criterion_stock(stock_pred, stock.squeeze())
                losses['stock'] = loss_stock
                stock_loss_meter.update(loss_stock.item())
            
            if 'industry' in config['training']['losses']:
                loss_industry = criterion_industry(industry_pred, industry.squeeze())
                losses['industry'] = loss_industry
                industry_loss_meter.update(loss_industry.item())
            
            total_loss = sum(losses.values())
            total_loss.backward()

            clip_norm = config['training']['clip_norm']
            if clip_norm > 0.01:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            ema.update(model)

            train_loss_meter.update(total_loss.item())
            #| Pred Loss: {pred_loss_meter.avg}
            pbar.set_description(f"Total({epoch+1}/{config['training']['num_epochs']}): {train_loss_meter.avg:.4f} | Trend: {trend_loss_meter.avg:.4f} | Trend Metric: {trend_metric_meter.avg:.4f} | Stock: {stock_loss_meter.avg:.4f} | Industry: {industry_loss_meter.avg:.4f}")
        
        scheduler.step()

        # --- 5. 验证循环 ---
        _model = copy.deepcopy(ema.module)
        mean_r2 = eval(_model, eval_dataset, config)
        
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
    
    run_eval(config)

def eval(model, dataset, config):
    _model = model#copy.deepcopy(ema.module)
    _model.eval()

    val_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=False, drop_last=True)
    with torch.no_grad():
        val_iter = DataPrefetcher(val_loader, config['device'], enable_queue=False, num_threads=1)
        trend_metric = ClsMetric('trend')
        stock_metric = ClsMetric('stock')
        industry_metric = ClsMetric('industry')

        for _ in tqdm(range(num_iters_per_epoch(dataset, config['training']['batch_size'])), desc="[Validation]"):
            # ts_sequences = ts_sequences.to(device)
            # ctx_sequences = ctx_sequences.to(device)
            # y = y.to(device)
            img, trend, stock, industry = val_iter.next()
            trend_pred, stock_pred, industry_pred = _model(img)

            trend_metric.update(trend.squeeze().cpu().numpy(), trend_pred.cpu().numpy())
            stock_metric.update(stock.squeeze().cpu().numpy(), stock_pred.cpu().numpy())
            industry_metric.update(industry.squeeze().cpu().numpy(), industry_pred.cpu().numpy())

    # --- 计算整体 R² --

    scores = []
    if 'trend' in config['training']['losses']:
        _, trend_score = trend_metric.calculate()
        scores.append(trend_score)
    if 'stock' in config['training']['losses']:
        _, stock_score = stock_metric.calculate()
        scores.append(stock_score)
    if 'industry' in config['training']['losses']:
        _, industry_score = industry_metric.calculate()
        scores.append(industry_score)
    
    mean_r2 = sum(scores) / len(scores)

    return mean_r2

def run_eval(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    test_dataset = ImagingPriceTrendDataset(
        db_path=config['data']['db_path'],
        img_caching_path=config['data']['test']['img_caching_path'],
        stock_list_file=config['data']['test']['stock_list_file'],
        hist_data_file=config['data']['test']['hist_data_file'],
        seq_length=config['data']['sequence_length'],
        features=config['data']['features'],
        image_size=config['data']['image_size'],
        tag='test',
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=False)
    model_config = get_model_config(config['training']['model'])

    model = create_model(config['training']['model'], model_config).to(device)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(config['training']['model_save_path'], map_location=device)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading model state dict: {e}")

    trend_metric = ClsMetric('trend')
    
    with torch.no_grad():
        test_iter = DataPrefetcher(test_loader, config['device'], enable_queue=False, num_threads=1)
        for _ in tqdm(range(num_iters_per_epoch(test_dataset, config['training']['batch_size'])), desc=f"[Testing]"):
            # ts_sequences = ts_sequences.to(device)
            # ctx_sequences = ctx_sequences.to(device)
            # y = y.to(device)

            img, trend, code = test_iter.next()
            trend_pred = model(img)

            trend_metric.update(trend.squeeze().cpu().numpy(), trend_pred.cpu().numpy())

        # --- 计算整体 R² ---
        _, trend_score = trend_metric.calculate()
