# train.py
import yaml
import torch
import torch.nn as nn
import os
import joblib
import copy
import ai.embedding.models.base
from ai.embedding.samplers.trend_sampler import TrendSampler
from autoclip.torch import QuantileClip
from ai.loss.tildeq import tildeq_loss
from ai.embedding.models import create_model, get_model_config
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # 提供优雅的进度条
from utils.common import AverageMeter
from config.base import *
from ai.embedding.dataset.dataset import KlineDataset, generate_scaler_and_encoder
from ai.modules.multiloss import AutomaticWeightedLoss
from ai.modules.earlystop import EarlyStopping
from ai.modules.gradient import adaptive_clip_grad
from ai.scheduler.sched import *
from ai.metrics import *
from utils.prefetcher import DataPrefetcher
from utils.common import ModelEmaV2, calculate_r2_components, calculate_r2_components_recon


def num_iters_per_epoch(loader, batch_size):
    return len(loader) // batch_size

def kl_loss(latent_mean, latent_logvar, free_bits=5):
    return torch.clamp(-0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()), min=free_bits).mean()

def ale_loss(pred, target, reduction='mean', eps=1e-8, gamma=0):
    pred = pred.clamp(eps, 1-eps)
    target = target.clamp(eps, 1-eps)
    abs_error = torch.abs(pred - target)

    loss = -torch.log(1.0 - abs_error + eps) * torch.pow(torch.maximum(abs_erroDataLoadr, torch.tensor(eps).to(abs_error.device)), gamma)

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()

    return loss

class HuberTrendLoss:
    def __init__(self, delta=0.1, sim_weight=0.1, tildeq=False):
        self.delta = delta
        self.tildeq = tildeq
        self.epsilon = 1e-8
        self.sim_weight = sim_weight

    def _similarity(self, y_true, y_pred):
        y_true_mean = torch.mean(y_true, dim=-1, keepdim=True)
        y_pred_mean = torch.mean(y_pred, dim=-1, keepdim=True)
        y_true_centered = y_true - y_true_mean
        y_pred_centered = y_pred - y_pred_mean
        # 2. 计算分子和分母
        numerator = torch.sum(y_true_centered * y_pred_centered, dim=-1)
        denominator = torch.sqrt(torch.sum(y_true_centered ** 2, dim=-1) * torch.sum(y_pred_centered ** 2, dim=-1) + 1e-12)
        # 3. 计算皮尔逊相关系数
        pearson_corr = numerator / (denominator)  # 添加一个小的常数以避免除以零
        return pearson_corr

    def _ccc_similarity(self, y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        # 步骤 2: 计算均值
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)
        
        # 步骤 3: 计算方差
        var_true = torch.var(y_true, unbiased=False)  # 使用总体方差 (n)
        var_pred = torch.var(y_pred, unbiased=False) # 使用总体方差 (n)
        
        # 步骤 4: 计算协方差
        # Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
        cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
        
        # 步骤 5: 组装 CCC 公式
        numerator = 2 * cov
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        
        # 添加 epsilon 防止除以零
        ccc = numerator / (denominator + self.epsilon)
        
        return ccc

    def directional_consistency_loss(self, y_true, y_pred, type='pearson'):
        """
        计算方向一致性损失
        Args:
            y_true: 真实值 (torch.Tensor)
            y_pred: 预测值 (torch.Tensor)
        Returns:
            方向一致性损失 (torch.Tensor)
        """
        corr = self._similarity(y_true, y_pred) if type == 'pearson' else self._ccc_similarity(y_true, y_pred)
        # 4. 计算损失
        loss = 1 - corr
        return torch.mean(loss)

    def __call__(self, ytrue, ypred):
        direction_loss = tildeq_loss(ypred, ytrue) if self.tildeq else nn.functional.huber_loss(ypred, ytrue, delta=self.delta)
        if not self.tildeq:
            sim_loss = self.directional_consistency_loss(ytrue, ypred)
            similarity = (1 - sim_loss).item()
            return direction_loss + sim_loss * self.sim_weight, similarity
        else:
            # sim_loss = nn.functional.huber_loss(ypred, ytrue, delta=self.delta)
            with torch.no_grad():
                similarity = self._ccc_similarity(ytrue, ypred).mean().item()
            # corr_loss = self.directional_consistency_loss(ytrue, ypred, type='ccc')
            return direction_loss, similarity
        

def run_training(config):
    """主训练函数"""
    # --- 1. 加载配置 ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(os.path.split(config['training']['model_save_path'])[0], exist_ok=True)
    config['data']['db_path'] = os.path.join(BASE_DIR, config['data']['db_path'])
    scaler_path = os.path.join(config['data']['db_path'], 'scaler.joblib')
    encoder_path = os.path.join(config['data']['db_path'], 'encoder.joblib')
    os.makedirs(config['training']['processed_data_cache_path'], exist_ok=True)
    if os.path.exists(scaler_path) and os.path.exists(encoder_path):
        print("Loading precomputed scaler and encoder...")
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
    else:
        encoder, scaler = generate_scaler_and_encoder(
            config['data']['db_path'],
            [
            config['data']['train']['hist_data_file'],
            config['data']['eval']['hist_data_file'],
            config['data']['test']['hist_data_file']
        ], config['data']['features'], config['data']['numerical'], config['data']['categorical'])
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)
        print(f"Scaler and encoder saved to {scaler_path} and {encoder_path}")

    # --- 2. 准备数据 ---
    if not config['training']['finetune']:
        train_dataset = KlineDataset(
            cache=config['data']['cache'],
            db_path=config['data']['db_path'],
            stock_list_file=config['data']['train']['stock_list_file'],
            hist_data_file=config['data']['train']['hist_data_file'],
            seq_length=config['training']['sequence_length'],
            features=config['data']['features'],
            numerical=config['data']['numerical'],
            categorical=config['data']['categorical'],
            include_meta=config['data']['include_meta'],
            scaler=scaler,
            encoder=encoder,
            tag='train'
        )

        eval_dataset = KlineDataset(
            cache=config['data']['cache'],
            db_path=config['data']['db_path'],
            stock_list_file=config['data']['eval']['stock_list_file'],
            hist_data_file=config['data']['eval']['hist_data_file'],
            seq_length=config['training']['sequence_length'],
            features=config['data']['features'],
            numerical=config['data']['numerical'],
            categorical=config['data']['categorical'],
            include_meta=config['data']['include_meta'],
            scaler=scaler,
            encoder=encoder,
            is_train=False,
            tag='eval'
        )
    else:
        train_dataset = KlineDataset(
            cache=config['data']['cache'],
            db_path=config['data']['db_path'],
            stock_list_file=config['data']['eval']['stock_list_file'],
            hist_data_file=config['data']['eval']['hist_data_file'],
            seq_length=config['training']['sequence_length'],
            features=config['data']['features'],
            numerical=config['data']['numerical'],
            categorical=config['data']['categorical'],
            include_meta=config['data']['include_meta'],
            scaler=scaler,
            encoder=encoder,
            is_train=False,
            tag='eval'
        )
        eval_dataset = KlineDataset(
            cache=config['data']['cache'],
            db_path=config['data']['db_path'],
            stock_list_file=config['data']['test']['stock_list_file'],
            hist_data_file=config['data']['test']['hist_data_file'],
            seq_length=config['training']['sequence_length'],
            features=config['data']['features'],
            numerical=config['data']['numerical'],
            categorical=config['data']['categorical'],
            include_meta=config['data']['include_meta'],
            scaler=scaler,
            encoder=encoder,
            is_train=False,
            tag='test'
        )

    train_sampler = TrendSampler(train_dataset, config['training']['batch_size'])
    train_loader = DataLoader(train_dataset, num_workers=config['training']['workers'], pin_memory=False, shuffle=False, drop_last=False, batch_sampler=train_sampler)
    val_loader = DataLoader(eval_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=False, drop_last=True)

    
    print(f"Training data size: {len(train_dataset)}, Validation data size: {len(eval_dataset)}")
    if config['training']['awl']:
        awl = AutomaticWeightedLoss(len(config['training']['losses']) + 1)
        awl.to(device)

    # --- 3. 初始化模型、损失函数和优化器 ---

    model_config = get_model_config(config['training']['model'])
    model_config['ts_input_dim'] = len(config['data']['features'])
    model_config['ctx_input_dim'] = len(config['data']['numerical'] + config['data']['categorical'])
    model_config['trend_classes'] = train_dataset.trend_classes()

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

    if config['training']['reset_heads']:
        model.build_head(train_dataset.trend_classes())

    ema = ModelEmaV2(model, decay=0.9999, device=device)

    model = model.to(device)

    criterion_ts = HuberTrendLoss(delta=0.1, tildeq=True) # 均方误差损失
    criterion_ctx = nn.HuberLoss(delta=0.3) # 均方误差损失
    criterion_predict = HuberTrendLoss(delta=0.1, tildeq=False) # 均方误差损失
    criterion_trend = nn.CrossEntropyLoss() #HuberTrendLoss(delta=0.1, sim_weight=0.)
    criterion_return = nn.HuberLoss(delta=0.1) #HuberTrendLoss(delta=0.1, sim_weight=0.1)

    parameters = []
    if config['training']['awl']:
        parameters += [{'params': awl.parameters(), 'weight_decay': 0}]
    parameters += [{'params': model.parameters(), 'weight_decay': config['training']['weight_decay']}]
    optimizer = torch.optim.AdamW(parameters, lr=config['training']['min_learning_rate'] if config['training']['warmup_epochs'] > 0 else config['training']['learning_rate'])
    if config['training']['auto_grad_norm']:
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=0.9, history_length=1000)
    early_stopper = EarlyStopping(patience=40, direction='up')
    
    scheduler = CosineWarmupLR(
        optimizer, config['training']['num_epochs']*num_iters_per_epoch(train_dataset, config['training']['batch_size']), config['training']['learning_rate'], config['training']['min_learning_rate'], warmup_epochs=config['training']['warmup_epochs'] * num_iters_per_epoch(train_dataset, config['training']['batch_size']), warmup_lr=config['training']['min_learning_rate'])

    # --- 4. 训练循环 ---
    best_val_loss = float('inf') if early_stopper.direction == 'down' else -float('inf')
    train_iter = DataPrefetcher(train_loader, config['device'], enable_queue=False, num_threads=1)
    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss_meter = AverageMeter()
        ts_loss_meter = AverageMeter()
        ctx_loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        trend_loss_meter = AverageMeter()
        return_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()
        
        ts_sim_meter = AverageMeter()
        pred_sim_meter = AverageMeter()


        if not config['training']['awl']:
            if epoch == 0 or epoch % config['training']['kl_annealing_steps'] != 0:
                kl_weight = min(config['training']['kl_weight_initial'] + (config['training']['kl_target'] - config['training']['kl_weight_initial']) * epoch / config['training']['kl_annealing_steps'], config['training']['kl_target'])
            else:
                kl_weight = config['training']['kl_weight_initial']
        else:
            kl_weight = config['training']['kl_weight_initial']
        
        # 使用tqdm显示进度条
        pbar = tqdm(range(num_iters_per_epoch(train_dataset, config['training']['batch_size'])), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training]")
        for _ in pbar:
            ts_sequences, ctx_sequences, y, trend, _return = train_iter.next()
            optimizer.zero_grad()
            ts_reconstructed, ctx_reconstructed, pred, trend_pred, return_pred, _, latent_mean, latent_logvar = model(ts_sequences, ctx_sequences)

            losses = {}

            if 'ts' in config['training']['losses']:
                loss_ts, ts_sim = criterion_ts(ts_sequences, ts_reconstructed)
                ts_sim_meter.update(ts_sim)
                ts_loss_meter.update(loss_ts.item())
                
                losses['ts'] = loss_ts
            
            if 'ctx' in config['training']['losses']:
                loss_ctx = criterion_ctx(ctx_sequences, ctx_reconstructed)
                ctx_loss_meter.update(loss_ctx.item())
                losses['ctx'] = loss_ctx
            
            if 'pred' in config['training']['losses']:
                loss_pred, pred_sim = criterion_predict(y, pred)
                losses['pred'] = loss_pred
                pred_loss_meter.update(loss_pred.item())
                pred_sim_meter.update(pred_sim)
            
            if 'trend' in config['training']['losses']:
                loss_trend = criterion_trend(trend_pred, trend.squeeze())
                losses['trend'] = loss_trend
                trend_loss_meter.update(loss_trend.item())

            if 'return' in config['training']['losses']:
                loss_return = criterion_return(return_pred, _return)
                losses['return'] = loss_return
                return_loss_meter.update(loss_return.item())
            
            if kl_weight > 0:
                _kl_loss = kl_loss(latent_mean, latent_logvar, config['training']['kl_freebits'])
                kl_loss_meter.update(_kl_loss.item())
            

            if config['training']['awl']:
                losses = list(losses.values())
                if kl_weight > 0:
                    losses.append(_kl_loss)
                total_loss = awl(*losses)
            else:
                total_loss = sum([losses[lk] * w for w, lk in zip(config['training']['loss_weights'], config['training']['losses'])])
                if kl_weight > 0:
                    total_loss += kl_weight * _kl_loss

            total_loss.backward()

            if not config.get('auto_grad_norm', True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()

            ema.update(model)

            train_loss_meter.update(total_loss.item())
            #| Pred Loss: {pred_loss_meter.avg}
            pbar.set_description(f"({epoch+1}/{config['training']['num_epochs']})LR: {scheduler.learning_rate[0]:.4f} | Total: {train_loss_meter.avg:.4f} | KL: {kl_loss_meter.avg:.4f} | TS: {ts_loss_meter.avg:.4f} | TS Sim: {ts_sim_meter.avg:.4f} | CTX: {ctx_loss_meter.avg:.4f} | Pred: {pred_loss_meter.avg:.4f} | Pred Sim: {pred_sim_meter.avg:.4f} | Trend: {trend_loss_meter.avg:.4f} | Return: {return_loss_meter.avg:.4f}")
        

        # --- 5. 验证循环 ---
        _model = copy.deepcopy(ema.module)
        _model.eval()
        with torch.no_grad():
            val_iter = DataPrefetcher(val_loader, config['device'], enable_queue=False, num_threads=1)
            pred_metric = Metric('vwap', appends=True)
            ts_metric = Metric('ts')
            ctx_metric = Metric('ctx')
            trend_metric = ClsMetric('trend')
            return_metric = Metric('return')

            for _ in tqdm(range(num_iters_per_epoch(eval_dataset, config['training']['batch_size'])), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Validation]"):
                # ts_sequences = ts_sequences.to(device)
                # ctx_sequences = ctx_sequences.to(device)
                # y = y.to(device)
                ts_sequences, ctx_sequences, y, trend, _return = val_iter.next()
                ts_reconstructed, ctx_reconstructed, pred, trend_pred, return_pred, _ = _model(ts_sequences, ctx_sequences)
                
                y_cpu = y.cpu().numpy()
                pred_cpu = pred.cpu().numpy()

                pred_metric.update(y_cpu, pred_cpu)
                trend_metric.update(trend.squeeze().cpu().numpy(), trend_pred.cpu().numpy())
                return_metric.update(_return.cpu().numpy() + 1, return_pred.cpu().numpy() + 1)
                
                ts_sequences_cpu = ts_sequences.cpu().numpy()
                ts_reconstructed_cpu = ts_reconstructed.cpu().numpy()
                ctx_sequences_cpu = ctx_sequences.cpu().numpy()
                ctx_reconstructed_cpu = ctx_reconstructed.cpu().numpy()
                ctx_reconstructed_cpu[:, -1] = np.round(ctx_reconstructed_cpu[:, -1], 2)

                ts_metric.update(ts_sequences_cpu.reshape(-1, ts_sequences_cpu.shape[-1]), ts_reconstructed_cpu.reshape(-1, ts_reconstructed_cpu.shape[-1]))
                ctx_metric.update(ctx_sequences_cpu.reshape(-1, ctx_sequences_cpu.shape[-1]), ctx_reconstructed_cpu.reshape(-1, ctx_reconstructed_cpu.shape[-1]))

        # --- 计算整体 R² --

        scores = []
        if 'ts' in config['training']['losses']:
            _, ts_score = ts_metric.calculate()
            scores.append(ts_score)
        if 'ctx' in config['training']['losses']:
            _, ctx_score = ctx_metric.calculate()
            scores.append(ctx_score)
        if 'pred' in config['training']['losses']:
            _, vwap_score = pred_metric.calculate()
            scores.append(vwap_score)
        if 'trend' in config['training']['losses']:
            _, trend_score = trend_metric.calculate()
            scores.append(trend_score)
        if 'return' in config['training']['losses']:
            _, return_score = return_metric.calculate()
            scores.append(return_score)
        
        mean_r2 = sum(scores) / len(scores)
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

def run_eval(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    config['data']['db_path'] = os.path.join(BASE_DIR, config['data']['db_path'])
    scaler_path = os.path.join(config['data']['db_path'], 'scaler.joblib')
    encoder_path = os.path.join(config['data']['db_path'], 'encoder.joblib')
    if os.path.exists(scaler_path) and os.path.exists(encoder_path):
        print("Loading precomputed scaler and encoder...")
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
    else:
        encoder, scaler = generate_scaler_and_encoder(
            config['data']['db_path'],
            [
            config['data']['train']['hist_data_file'],
            config['data']['eval']['hist_data_file'],
            config['data']['test']['hist_data_file']
        ], config['data']['features'], config['data']['numerical'], config['data']['categorical'])
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)
        print(f"Scaler and encoder saved to {scaler_path} and {encoder_path}")

    test_dataset = KlineDataset(
        cache=config['data']['cache'],
        db_path=config['data']['db_path'],
        stock_list_file=config['data']['test']['stock_list_file'],
        hist_data_file=config['data']['test']['hist_data_file'],
        seq_length=config['training']['sequence_length'],
        features=config['data']['features'],
        numerical=config['data']['numerical'],
        categorical=config['data']['categorical'],
        include_meta=config['data']['include_meta'],
        scaler=scaler,
        encoder=encoder,
        is_train=False,
        tag='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=False, shuffle=False)

    model_config = get_model_config(config['training']['model'])
    model_config['ts_input_dim'] = len(config['data']['features'])
    model_config['ctx_input_dim'] = len(config['data']['numerical'] + config['data']['categorical'])
    model_config['trend_classes'] = test_dataset.trend_classes()

    model = create_model(config['training']['model'], model_config)
    
    model = model.to(device)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(config['training']['model_save_path'], map_location=device)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading model state dict: {e}")
    model.eval()
    
    pred_metric = Metric('vwap', appends=True)
    ts_metric = Metric('ts')
    ctx_metric = Metric('ctx')
    trend_metric = ClsMetric('trend')
    return_metric = Metric('return')
    
    with torch.no_grad():
        test_iter = DataPrefetcher(test_loader, config['device'], enable_queue=False, num_threads=1)
        for _ in tqdm(range(num_iters_per_epoch(test_dataset, config['training']['batch_size'])), desc=f"[Testing]"):
            # ts_sequences = ts_sequences.to(device)
            # ctx_sequences = ctx_sequences.to(device)
            # y = y.to(device)

            ts_sequences, ctx_sequences, y, trend, _return = test_iter.next()
            ts_reconstructed, ctx_reconstructed, pred, trend_pred, return_pred, _ = model(ts_sequences, ctx_sequences)
            
            y_cpu = y.cpu().numpy()
            pred_cpu = pred.cpu().numpy()
            trend_cpu = trend.cpu().numpy()
            return_cpu = _return.cpu().numpy()

            pred_metric.update(y_cpu, pred_cpu)
            trend_metric.update(trend_cpu.squeeze(), trend_pred.cpu().numpy())
            return_metric.update(return_cpu + 1, return_pred.cpu().numpy() + 1)
                
            ts_sequences_cpu = ts_sequences.cpu().numpy()
            ts_reconstructed_cpu = ts_reconstructed.cpu().numpy()
            ctx_sequences_cpu = ctx_sequences.cpu().numpy()
            ctx_reconstructed_cpu = ctx_reconstructed.cpu().numpy()
            ctx_reconstructed_cpu[:, -1] = np.round(ctx_reconstructed_cpu[:, -1], 2)

            ts_metric.update(ts_sequences_cpu.reshape(-1, ts_sequences_cpu.shape[-1]), ts_reconstructed_cpu.reshape(-1, ts_reconstructed_cpu.shape[-1]))
            ctx_metric.update(ctx_sequences_cpu.reshape(-1, ctx_sequences_cpu.shape[-1]), ctx_reconstructed_cpu.reshape(-1, ctx_reconstructed_cpu.shape[-1]))



        # --- 计算整体 R² ---
        _, vwap_score = pred_metric.calculate()
        _, ts_score = ts_metric.calculate()
        _, ctx_score = ctx_metric.calculate()
        _, trend_score = trend_metric.calculate()
        _, return_score = return_metric.calculate()
