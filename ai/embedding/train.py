# train.py
import yaml
import torch
import torch.nn as nn
import os
import joblib
import copy
import ai.embedding.models.base
from ai.embedding.models import create_model, get_model_config
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # 提供优雅的进度条
from utils.common import AverageMeter
from config.base import *
from ai.embedding.dataset.dataset import KlineDataset, generate_scaler_and_encoder
from ai.modules.multiloss import AutomaticWeightedLoss
from ai.modules.earlystop import EarlyStopping
from ai.scheduler.sched import *
from utils.prefetcher import DataPrefetcher
from utils.common import ModelEmaV2, calculate_r2_components, calculate_r2_components_recon


def num_iters_per_epoch(loader, batch_size):
    return len(loader) // batch_size

def kl_loss(latent_mean, latent_logvar):
    return -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

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
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=4, pin_memory=False, shuffle=True, drop_last=True)
    val_loader = DataLoader(eval_dataset, batch_size=config['training']['batch_size'], num_workers=4, pin_memory=False, shuffle=False, drop_last=True)

    
    print(f"Training data size: {len(train_dataset)}, Validation data size: {len(eval_dataset)}")

    if config['training']['awl']:
        awl = AutomaticWeightedLoss(len(config['training']['losses']), config['training']['loss_weights'])
        awl.to(device)

    # --- 3. 初始化模型、损失函数和优化器 ---

    model_config = get_model_config(config['training']['model'])
    model_config['ts_input_dim'] = len(config['data']['features'])
    model_config['ctx_input_dim'] = len(config['data']['numerical'] + config['data']['categorical'])

    model = create_model(config['training']['model'], model_config)

    ema = ModelEmaV2(model, decay=0.9999, device=device)

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

    model = model.to(device)

    criterion_ts = nn.HuberLoss(delta=0.1) # 均方误差损失
    criterion_ctx = nn.HuberLoss(delta=0.1) # 均方误差损失
    criterion_predict = nn.HuberLoss(delta=0.1) # 均方误差损失
    parameters = []
    if config['training']['awl']:
        parameters += [{'params': awl.parameters(), 'weight_decay': 0}]
    parameters += [{'params': model.parameters(), 'weight_decay': config['training']['weight_decay']}]
    optimizer = torch.optim.AdamW(parameters, lr=config['training']['min_learning_rate'] if config['training']['warmup_epochs'] > 0 else config['training']['learning_rate'])
    early_stopper = EarlyStopping(patience=10, direction='up')
    
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
        train_iter = DataPrefetcher(train_loader, config['device'], enable_queue=False, num_threads=1)


        if epoch == 0 or epoch % config['training']['kl_annealing_steps'] != 0:
            kl_weight = min(config['training']['kl_weight_initial'] + (config['training']['kl_target'] - config['training']['kl_weight_initial']) * epoch / config['training']['kl_annealing_steps'], config['training']['kl_target'])
        else:
            kl_weight = config['training']['kl_weight_initial']  # 重启
        
        # 使用tqdm显示进度条
        pbar = tqdm(range(num_iters_per_epoch(train_dataset, config['training']['batch_size'])), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training]")
        for _ in pbar:
            ts_sequences, ctx_sequences, y, _, _ = train_iter.next()
            optimizer.zero_grad()
            ts_reconstructed, ctx_reconstructed, pred, _, latent_mean, latent_logvar = model(ts_sequences, ctx_sequences)

            losses = {}

            if 'ts' in config['training']['losses']:
                loss_ts = criterion_ts(ts_reconstructed, ts_sequences)

                ts_loss_meter.update(loss_ts.item())
                
                losses['ts'] = loss_ts
            
            if 'ctx' in config['training']['losses']:
                loss_ctx = criterion_ctx(ctx_reconstructed, ctx_sequences)
                ctx_loss_meter.update(loss_ctx.item())
                losses['ctx'] = loss_ctx
            
            if 'pred' in config['training']['losses']:
                loss_pred = criterion_predict(pred, y)
                losses['pred'] = loss_pred
                pred_loss_meter.update(loss_pred.item())

            _kl_loss = kl_loss(latent_mean, latent_logvar)
            kl_loss_meter.update(_kl_loss.item())

            if config['training']['awl']:
                total_loss = awl(*list(losses.values))
            else:
                total_loss = sum([losses[lk] * w for w, lk in zip(config['training']['loss_weights'], config['training']['losses'])])

            total_loss += kl_weight * _kl_loss
            total_loss.backward()
            optimizer.step()

            ema.update(model)

            train_loss_meter.update(total_loss.item())
            #| Pred Loss: {pred_loss_meter.avg}
            pbar.set_description(f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training] | Loss: {train_loss_meter.avg:.4f} | KL Loss: {kl_loss_meter.avg:.4f} | TS Loss: {ts_loss_meter.avg:.4f} | CTX Loss: {ctx_loss_meter.avg:.4f} | Pred Loss: {pred_loss_meter.avg:.4f}")
        
        scheduler.step()

        # --- 5. 验证循环 ---
        _model = copy.deepcopy(ema.module)
        _model.eval()
        with torch.no_grad():
            val_iter = DataPrefetcher(val_loader, config['device'], enable_queue=False, num_threads=1)
            # truth = []
            # preds = []
            # ts_reconstructed_list = []
            # ctx_reconstructed_list = []
            # ts_sequence_list = []
            # ctx_sequence_list = []
            # 初始化 R² 相关变量
            overall_ssr = 0.0
            overall_sst = 0.0
            overall_ts_ssr = 0.0
            overall_ts_sst = 0.0
            overall_ctx_ssr = 0.0
            overall_ctx_sst = 0.0
            for _ in tqdm(range(num_iters_per_epoch(eval_dataset, config['training']['batch_size'])), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Validation]"):
                # ts_sequences = ts_sequences.to(device)
                # ctx_sequences = ctx_sequences.to(device)
                # y = y.to(device)
                ts_sequences, ctx_sequences, y, _, _ = val_iter.next()
                ts_reconstructed, ctx_reconstructed, pred, _ = _model(ts_sequences, ctx_sequences)

                
                y_cpu = y.cpu().numpy()
                pred_cpu = pred.cpu().numpy()
                ssr, sst = calculate_r2_components(y_cpu, pred_cpu)
                overall_ssr += ssr
                overall_sst += sst
                ts_sequences_cpu = ts_sequences.cpu().numpy()
                ts_reconstructed_cpu = ts_reconstructed.cpu().numpy()
                ts_ssr, ts_sst = calculate_r2_components_recon(ts_sequences_cpu.reshape(-1, ts_sequences_cpu.shape[-1]), ts_reconstructed_cpu.reshape(-1, ts_reconstructed_cpu.shape[-1]))
                overall_ts_ssr += ts_ssr
                overall_ts_sst += ts_sst
                ctx_sequences_cpu = ctx_sequences.cpu().numpy()
                ctx_reconstructed_cpu = ctx_reconstructed.cpu().numpy()
                ctx_reconstructed_cpu[:, -1] = np.round(ctx_reconstructed_cpu[:, -1], 2)
                ctx_ssr, ctx_sst = calculate_r2_components_recon(ctx_sequences_cpu.reshape(-1, ctx_sequences_cpu.shape[-1]), ctx_reconstructed_cpu.reshape(-1, ctx_reconstructed_cpu.shape[-1]))
                overall_ctx_ssr += ctx_ssr
                overall_ctx_sst += ctx_sst

        # r2 = r2_score(np.concatenate(truth), np.concatenate(preds))
        # ts_true = np.concatenate(ts_sequence_list)
        # ts_true = ts_true.reshape(-1, ts_true.shape[-1])

        # ts_recon = np.concatenate(ts_reconstructed_list)
        # ts_recon = ts_recon.reshape(-1, ts_recon.shape[-1])
        # r2_recon = r2_score(ts_true, ts_recon)

        # ctx_true = np.concatenate(ctx_sequence_list)
        # ctx_true = ctx_true.reshape(-1, ctx_true.shape[-1])

        # ctx_recon = np.concatenate(ctx_reconstructed_list)
        # ctx_recon = ctx_recon.reshape(-1, ctx_recon.shape[-1])
        # ctx_recon[:, -1] = np.round(ctx_recon[:, -1], 2)
        # r2_ctx_recon = r2_score(ctx_true, ctx_recon)

        # --- 计算整体 R² ---
        if overall_sst > 0:
            r2 = 1 - (overall_ssr / overall_sst)
        else:
            r2 = 0.0  # 避免除以零
        if overall_ts_sst > 0:
            r2_recon = 1 - (overall_ts_ssr / overall_ts_sst)
        else:
            r2_recon = 0.0
        if overall_ctx_sst > 0:
            r2_ctx_recon = 1 - (overall_ctx_ssr / overall_ctx_sst)
        else:
            r2_ctx_recon = 0.0
        
        print(f"Epoch {epoch+1}: R2 Score = {r2} R2 Recon Score = {r2_recon}, R2 CTX Recon Score = {r2_ctx_recon}")
        
        mean_r2 = (r2_ctx_recon + r2_recon + r2) / 3
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
    
    overall_ssr = 0.0
    overall_sst = 0.0
    overall_ts_ssr = 0.0
    overall_ts_sst = 0.0
    overall_ctx_ssr = 0.0
    overall_ctx_sst = 0.0
    with torch.no_grad():
        test_iter = DataPrefetcher(test_loader, config['device'], enable_queue=False, num_threads=1)
        for _ in tqdm(range(num_iters_per_epoch(test_dataset, config['training']['batch_size'])), desc=f"[Testing]"):
            # ts_sequences = ts_sequences.to(device)
            # ctx_sequences = ctx_sequences.to(device)
            # y = y.to(device)

            ts_sequences, ctx_sequences, y, _, _ = test_iter.next()
            ts_reconstructed, ctx_reconstructed, pred, _ = model(ts_sequences, ctx_sequences)
            
            y_cpu = y.cpu().numpy()
            pred_cpu = pred.cpu().numpy()
            ssr, sst = calculate_r2_components(y_cpu, pred_cpu)
            overall_ssr += ssr
            overall_sst += sst
            ts_sequences_cpu = ts_sequences.cpu().numpy()
            ts_reconstructed_cpu = ts_reconstructed.cpu().numpy()
            ts_ssr, ts_sst = calculate_r2_components_recon(ts_sequences_cpu.reshape(-1, ts_sequences_cpu.shape[-1]), ts_reconstructed_cpu.reshape(-1, ts_reconstructed_cpu.shape[-1]))
            overall_ts_ssr += ts_ssr
            overall_ts_sst += ts_sst
            ctx_sequences_cpu = ctx_sequences.cpu().numpy()
            ctx_reconstructed_cpu = ctx_reconstructed.cpu().numpy()
            ctx_reconstructed_cpu[:, -1] = np.round(ctx_reconstructed_cpu[:, -1], 2)
            ctx_ssr, ctx_sst = calculate_r2_components_recon(ctx_sequences_cpu.reshape(-1, ctx_sequences_cpu.shape[-1]), ctx_reconstructed_cpu.reshape(-1, ctx_reconstructed_cpu.shape[-1]))
            overall_ctx_ssr += ctx_ssr
            overall_ctx_sst += ctx_sst


        # --- 计算整体 R² ---
        if overall_sst > 0:
            r2 = 1 - (overall_ssr / overall_sst)
        else:
            r2 = 0.0  # 避免除以零
        if overall_ts_sst > 0:
            r2_recon = 1 - (overall_ts_ssr / overall_ts_sst)
        else:
            r2_recon = 0.0
        if overall_ctx_sst > 0:
            r2_ctx_recon = 1 - (overall_ctx_ssr / overall_ctx_sst)
        else:
            r2_ctx_recon = 0.0
        print(f"Test R2 Score = {r2} Test R2 Recon Score = {r2_recon}, R2 CTX Recon Score = {r2_ctx_recon}")
