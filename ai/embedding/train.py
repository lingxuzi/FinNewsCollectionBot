# train.py
import yaml
import torch
import torch.nn as nn
import os
import joblib
import copy
import ai.embedding.models.base
from ai.loss.quantileloss import QuantileLoss
from ai.embedding.models import create_model, get_model_config
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # 提供优雅的进度条
from utils.common import AverageMeter
from config.base import *
from ai.embedding.dataset.dataset import KlineDataset, generate_scaler_and_encoder
from ai.modules.multiloss import AutomaticWeightedLoss
from ai.modules.earlystop import EarlyStopping
from ai.modules.gradient import adaptive_clip_grad
from ai.scheduler.sched import *
from ai.metrics.series_metric import Metric
from utils.prefetcher import DataPrefetcher
from utils.common import ModelEmaV2, calculate_r2_components, calculate_r2_components_recon


def num_iters_per_epoch(loader, batch_size):
    return len(loader) // batch_size

def kl_loss(latent_mean, latent_logvar, free_bits=5):
    return torch.clamp(-0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()), min=free_bits).mean()

class HuberTrendLoss:
    def __init__(self, delta=0.1):
        self.delta = delta
        self.epsilon = 1e-8

    def directional_consistency_loss(self, y_true, y_pred):
        """
        计算方向一致性损失
        Args:
            y_true: 真实值 (torch.Tensor)
            y_pred: 预测值 (torch.Tensor)
        Returns:
            方向一致性损失 (torch.Tensor)
        """
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        
        # Reshape to treat the sequence of differences as a single vector for each batch item
        # Shape becomes (batch_size, (seq_len-1) * features)
        diff_pred_vec = diff_pred.contiguous().view(diff_pred.size(0), -1)
        diff_true_vec = diff_true.contiguous().view(diff_true.size(0), -1)

        # 中心化数据 (减去均值)
        diff_pred_vec_centered = diff_pred_vec - torch.mean(diff_pred_vec, dim=1, keepdim=True)
        diff_true_vec_centered = diff_true_vec - torch.mean(diff_true_vec, dim=1, keepdim=True)

        # Calculate cosine similarity
        # F.cosine_similarity handles the dot product and norms internally
        # It operates on a specified dimension, here dim=1 for batch-wise comparison
        cosine_sim = nn.functional.cosine_similarity(diff_pred_vec_centered, diff_true_vec_centered, dim=1, eps=self.epsilon)

        # The loss is 1 - cosine_similarity
        # We take the mean over the batch
        loss = 1.0 - cosine_sim
        return torch.mean(loss)

    def __call__(self, ytrue, ypred):
        direction_loss = self.directional_consistency_loss(ytrue, ypred)
        reconstruction_loss = nn.functional.huber_loss(ypred, ytrue, delta=self.delta)
        return direction_loss * 0.2 + reconstruction_loss, 1 - direction_loss.item()

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
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=2, pin_memory=False, shuffle=True, drop_last=True)
    val_loader = DataLoader(eval_dataset, batch_size=config['training']['batch_size'], num_workers=2, pin_memory=False, shuffle=False, drop_last=True)

    
    print(f"Training data size: {len(train_dataset)}, Validation data size: {len(eval_dataset)}")
    if config['training']['awl']:
        awl = AutomaticWeightedLoss(len(config['training']['losses']) + 1)
        awl.to(device)

    # --- 3. 初始化模型、损失函数和优化器 ---

    model_config = get_model_config(config['training']['model'])
    model_config['ts_input_dim'] = len(config['data']['features'])
    model_config['ctx_input_dim'] = len(config['data']['numerical'] + config['data']['categorical'])

    model = create_model(config['training']['model'], model_config)


    if config['training']['load_pretrained']:
        try:
            state_dict = torch.load(config['training']['pretrained_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
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
        for head in config['training']['reset_heads']:
            model.reset_prediction_head(head)

    ema = ModelEmaV2(model, decay=0.9999, device=device)

    model = model.to(device)

    criterion_ts = HuberTrendLoss(delta=0.1) # 均方误差损失
    criterion_ctx = nn.HuberLoss(delta=0.1) # 均方误差损失
    criterion_predict = HuberTrendLoss(delta=0.1) # 均方误差损失
    parameters = []
    if config['training']['awl']:
        parameters += [{'params': awl.parameters(), 'weight_decay': 0}]
    parameters += [{'params': model.parameters(), 'weight_decay': config['training']['weight_decay']}]
    optimizer = torch.optim.AdamW(parameters, lr=config['training']['min_learning_rate'] if config['training']['warmup_epochs'] > 0 else config['training']['learning_rate'])
    
    early_stopper = EarlyStopping(patience=40, direction='up')
    
    scheduler = CosineWarmupLR(
        optimizer, config['training']['num_epochs'], config['training']['learning_rate'], config['training']['min_learning_rate'], warmup_epochs=config['training']['warmup_epochs'], warmup_lr=config['training']['min_learning_rate'])

    # --- 4. 训练循环 ---
    best_val_loss = float('inf') if early_stopper.direction == 'down' else -float('inf')
    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss_meter = AverageMeter()
        ts_loss_meter = AverageMeter()
        ctx_loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()
        
        ts_sim_meter = AverageMeter()
        pred_sim_meter = AverageMeter()

        train_iter = DataPrefetcher(train_loader, config['device'], enable_queue=False, num_threads=1)

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
            ts_sequences, ctx_sequences, y = train_iter.next()
            optimizer.zero_grad()
            ts_reconstructed, ctx_reconstructed, pred, _, latent_mean, latent_logvar = model(ts_sequences, ctx_sequences)

            losses = {}

            if 'ts' in config['training']['losses']:
                loss_ts, ts_sim = criterion_ts(ts_reconstructed, ts_sequences)
                ts_sim_meter.update(ts_sim)
                ts_loss_meter.update(loss_ts.item())
                
                losses['ts'] = loss_ts
            
            if 'ctx' in config['training']['losses']:
                loss_ctx = criterion_ctx(ctx_reconstructed, ctx_sequences)
                ctx_loss_meter.update(loss_ctx.item())
                losses['ctx'] = loss_ctx
            
            if 'pred' in config['training']['losses']:
                loss_pred, pred_sim = criterion_predict(pred, y)
                losses['pred'] = loss_pred
                pred_loss_meter.update(loss_pred.item())
                pred_sim_meter.update(pred_sim)


            _kl_loss = kl_loss(latent_mean, latent_logvar, config['training']['kl_freebits'])
            kl_loss_meter.update(_kl_loss.item())

            if config['training']['awl']:
                total_loss = awl(*(list(losses.values()) + [_kl_loss]))
            else:
                total_loss = sum([losses[lk] * w for w, lk in zip(config['training']['loss_weights'], config['training']['losses'])]) + kl_weight * _kl_loss

            total_loss.backward()
            optimizer.step()

            if config.get('auto_grad_norm', True):
                adaptive_clip_grad(model.parameters())

            ema.update(model)

            train_loss_meter.update(total_loss.item())
            #| Pred Loss: {pred_loss_meter.avg}
            pbar.set_description(f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training] | Loss: {train_loss_meter.avg:.4f} | KL Loss: {kl_loss_meter.avg:.4f} | TS Loss: {ts_loss_meter.avg:.4f} | TS Sim: {ts_sim_meter.avg:.4f} | CTX Loss: {ctx_loss_meter.avg:.4f} | Pred Loss: {pred_loss_meter.avg:.4f} | Pred Sim: {pred_sim_meter.avg:.4f} ")
        
        scheduler.step()

        # --- 5. 验证循环 ---
        _model = copy.deepcopy(ema.module)
        _model.eval()
        with torch.no_grad():
            val_iter = DataPrefetcher(val_loader, config['device'], enable_queue=False, num_threads=1)
            pred_metric = Metric('vwap')
            ts_metric = Metric('ts')
            ctx_metric = Metric('ctx')

            for _ in tqdm(range(num_iters_per_epoch(eval_dataset, config['training']['batch_size'])), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Validation]"):
                # ts_sequences = ts_sequences.to(device)
                # ctx_sequences = ctx_sequences.to(device)
                # y = y.to(device)
                ts_sequences, ctx_sequences, y = val_iter.next()
                ts_reconstructed, ctx_reconstructed, pred, _ = _model(ts_sequences, ctx_sequences)

                
                y_cpu = y.cpu().numpy()
                pred_cpu = pred.cpu().numpy()

                pred_metric.update(y_cpu, pred_cpu)
                
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

        scores = []
        if 'ts' in config['training']['losses']:
            scores.append(ts_score)
        if 'ctx' in config['training']['losses']:
            scores.append(ctx_score)
        if 'pred' in config['training']['losses']:
            scores.append(vwap_score)
        
        mean_r2 = sum(scores)
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
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=4, pin_memory=False, shuffle=False)

    model_config = get_model_config(config['training']['model'])
    model_config['ts_input_dim'] = len(config['data']['features'])
    model_config['ctx_input_dim'] = len(config['data']['numerical'] + config['data']['categorical'])

    model = create_model(config['training']['model'], model_config)
    
    model = model.to(device)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(config['training']['model_save_path'], map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    pred_metric = Metric('vwap')
    ts_metric = Metric('ts')
    ctx_metric = Metric('ctx')
    
    with torch.no_grad():
        test_iter = DataPrefetcher(test_loader, config['device'], enable_queue=False, num_threads=1)
        for _ in tqdm(range(num_iters_per_epoch(test_dataset, config['training']['batch_size'])), desc=f"[Testing]"):
            # ts_sequences = ts_sequences.to(device)
            # ctx_sequences = ctx_sequences.to(device)
            # y = y.to(device)

            ts_sequences, ctx_sequences, y = test_iter.next()
            ts_reconstructed, ctx_reconstructed, pred, _ = model(ts_sequences, ctx_sequences)
            
            y_cpu = y.cpu().numpy()
            pred_cpu = pred.cpu().numpy()

            pred_metric.update(y_cpu, pred_cpu)
                
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
