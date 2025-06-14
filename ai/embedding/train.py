# train.py
import yaml
import torch
import torch.nn as nn
import os
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # 提供优雅的进度条
from utils.common import AverageMeter
from config.base import *
from ai.embedding.dataset import KlineDataset, generate_scaler_and_encoder
from ai.embedding.model import MultiModalAutoencoder
from ai.modules.earlystop import EarlyStopping
from ai.scheduler.sched import *

def run_training(config):
    """主训练函数"""
    # --- 1. 加载配置 ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(os.path.split(config['training']['model_save_path'])[0], exist_ok=True)
    config['data']['db_path'] = os.path.join(BASE_DIR, config['data']['db_path'])
    encoder = generate_scaler_and_encoder(
        config['data']['db_path'],
        [
        config['data']['train']['hist_data_file'],
        config['data']['eval']['hist_data_file'],
        config['data']['test']['hist_data_file']
    ], config['data']['features'], config['data']['numerical'], config['data']['categorical'])

    # --- 2. 准备数据 ---
    train_dataset = KlineDataset(
        db_path=config['data']['db_path'],
        stock_list_file=config['data']['train']['stock_list_file'],
        hist_data_file=config['data']['train']['hist_data_file'],
        seq_length=config['training']['sequence_length'],
        features=config['data']['features'],
        numerical=config['data']['numerical'],
        categorical=config['data']['categorical'],
        scaler=None,
        encoder=encoder
    )

    eval_dataset = KlineDataset(
        db_path=config['data']['db_path'],
        stock_list_file=config['data']['eval']['stock_list_file'],
        hist_data_file=config['data']['eval']['hist_data_file'],
        seq_length=config['training']['sequence_length'],
        features=config['data']['features'],
        numerical=config['data']['numerical'],
        categorical=config['data']['categorical'],
        scaler=train_dataset.scaler,
        encoder=encoder,
        is_train=False
    )

    test_dataset = KlineDataset(
        db_path=config['data']['db_path'],
        stock_list_file=config['data']['test']['stock_list_file'],
        hist_data_file=config['data']['test']['hist_data_file'],
        seq_length=config['training']['sequence_length'],
        features=config['data']['features'],
        numerical=config['data']['numerical'],
        categorical=config['data']['categorical'],
        scaler=train_dataset.scaler,
        encoder=encoder,
        is_train=False
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(eval_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    print(f"Training data size: {len(train_dataset)}, Validation data size: {len(eval_dataset)}, Teset data size: {len(test_dataset)}")

    # --- 3. 初始化模型、损失函数和优化器 ---
    model = MultiModalAutoencoder(
        ts_input_dim=len(config['data']['features']),
        ctx_input_dim=len(config['data']['numerical'] + config['data']['categorical']),# + len(train_dataset.encoder.categories_[0])
        ts_embedding_dim=config['training']['ts_embedding_dim'],
        ctx_embedding_dim=config['training']['ctx_embedding_dim'],
        hidden_dim=config['training']['hidden_dim'],
        num_layers=config['training']['num_layers'],
        predict_dim=config['training']['predict_dim'],
        attention_dim=config['training']['attention_dim']
    )

    if config['training']['load_pretrained']:
        state_dict = torch.load(config['training']['pretrained_path'], map_location='cpu')
        model.load_state_dict(state_dict)
        print('pretrain loaded')
    
    model = model.to(device)

    criterion_ts = nn.HuberLoss(delta=0.1) # 均方误差损失
    criterion_ctx = nn.HuberLoss(delta=0.1) # 均方误差损失
    criterion_predict = nn.HuberLoss(delta=0.1) # 均方误差损失
    alpha = 0.2
    beta = 0.2
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['min_learning_rate'], weight_decay=1e-5)
    early_stopper = EarlyStopping(patience=10, direction='up')
    
    scheduler = CosineWarmupLR(
        optimizer, config['training']['num_epochs'], config['training']['learning_rate'], config['training']['min_learning_rate'], warmup_epochs=10, warmup_lr=config['training']['min_learning_rate'])

    # --- 4. 训练循环 ---
    best_val_loss = 0.0#float('inf')
    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training]")
        for ts_sequences, ctx_sequences, y in pbar:
            ts_sequences = ts_sequences.to(device)
            ctx_sequences = ctx_sequences.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            ts_reconstructed, ctx_reconstructed, pred, _ = model(ts_sequences, ctx_sequences)
            loss_ts = criterion_ts(ts_reconstructed, ts_sequences)
            loss_ctx = criterion_ctx(ctx_reconstructed, ctx_sequences)
            loss_pred = criterion_predict(pred, y)
            total_loss = loss_ts + alpha * loss_ctx + beta * loss_pred
            total_loss.backward()
            optimizer.step()

            train_loss_meter.update(total_loss.item())
            pred_loss_meter.update(loss_pred.item())
            #| Pred Loss: {pred_loss_meter.avg}
            pbar.set_description(f"Epoch {epoch+1}/{config['training']['num_epochs']} [Training] | Loss: {train_loss_meter.avg} | Pred Loss: {pred_loss_meter.avg}")
        
        scheduler.step()

        # --- 5. 验证循环 ---
        model.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            truth = []
            preds = []
            ts_reconstructed_list = []
            ctx_reconstructed_list = []
            ts_sequence_list = []
            ctx_sequence_list = []
            for ts_sequences, ctx_sequences, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Validation]"):
                ts_sequences = ts_sequences.to(device)
                ctx_sequences = ctx_sequences.to(device)
                y = y.to(device)
                ts_reconstructed, ctx_reconstructed, pred, _ = model(ts_sequences, ctx_sequences)
                loss_ts = criterion_ts(ts_reconstructed, ts_sequences)
                loss_ctx = criterion_ctx(ctx_reconstructed, ctx_sequences)
                loss_pred = criterion_predict(pred, y)
                total_loss = loss_ts + alpha * loss_ctx + beta * loss_pred
                val_loss_meter.update(total_loss.item())

                truth.append(y.cpu().numpy())
                preds.append(pred.cpu().numpy())
                ts_reconstructed_list.append(ts_reconstructed.cpu().numpy())
                ctx_reconstructed_list.append(ctx_reconstructed.cpu().numpy())
                ts_sequence_list.append(ts_sequences.cpu().numpy())
                ctx_sequence_list.append(ctx_sequences.cpu().numpy())

        r2 = r2_score(np.concatenate(truth), np.concatenate(preds))
        ts_true = np.concatenate(ts_sequence_list)
        ts_true = ts_true.reshape(-1, ts_true.shape[-1])

        ts_recon = np.concatenate(ts_reconstructed_list)
        ts_recon = ts_recon.reshape(-1, ts_recon.shape[-1])
        r2_recon = r2_score(ts_true, ts_recon)

        ctx_true = np.concatenate(ctx_sequence_list)
        ctx_true = ctx_true.reshape(-1, ctx_true.shape[-1])

        ctx_recon = np.concatenate(ctx_reconstructed_list)
        ctx_recon = ctx_recon.reshape(-1, ctx_recon.shape[-1])
        ctx_recon[:, -1] = np.round(ctx_recon[:, -1], 2)
        r2_ctx_recon = r2_score(ctx_true, ctx_recon)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss_meter.avg:.6f}, Val Loss = {val_loss_meter.avg:.6f}, R2 Score = {r2} R2 Recon Score = {r2_recon}, R2 CTX Recon Score = {r2_ctx_recon}")
        
        mean_r2 = (r2_ctx_recon + r2_recon + r2) / 3
        # --- 6. 保存最佳模型 ---
        # 只保存性能最好的模型，避免存储过多文件
        if mean_r2 > best_val_loss:
            best_val_loss = mean_r2
            # 我们只关心编码器的权重，也可以保存整个模型
            torch.save(model.state_dict(), config['training']['model_save_path'])
            # torch.save({
            #     'encoder_state_dict': model.encoder.state_dict(),
            #     'encoder_fc_state_dict': model.encoder_fc.state_dict(),
            #     'config': config # 将配置也一并保存，便于推理时加载
            # }, config['training']['model_save_path'])
            print(f"Model improved and saved to {config['training']['model_save_path']}")

        
        early_stopper(mean_r2)
        if early_stopper.early_stop:
            break

    state_dict = torch.load(config['training']['model_save_path'], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    truth = []
    preds = []
    ts_reconstructed_list = []
    ctx_reconstructed_list = []
    ts_sequence_list = []
    ctx_sequence_list = []
    with torch.no_grad():
        for ts_sequences, ctx_sequences, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Testing]"):
            ts_sequences = ts_sequences.to(device)
            ctx_sequences = ctx_sequences.to(device)
            y = y.to(device)
            ts_reconstructed, ctx_reconstructed, pred, _ = model(ts_sequences, ctx_sequences)
            truth.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
            ts_reconstructed_list.append(ts_reconstructed.cpu().numpy())
            ctx_reconstructed_list.append(ctx_reconstructed.cpu().numpy())
            ts_sequence_list.append(ts_sequences.cpu().numpy())
            ctx_sequence_list.append(ctx_sequences.cpu().numpy())

        r2 = r2_score(np.concatenate(truth), np.concatenate(preds))
        ts_true = np.concatenate(ts_sequence_list)
        ts_true = ts_true.reshape(-1, ts_true.shape[-1])

        ts_recon = np.concatenate(ts_reconstructed_list)
        ts_recon = ts_recon.reshape(-1, ts_recon.shape[-1])
        r2_recon = r2_score(ts_true, ts_recon)

        ctx_true = np.concatenate(ctx_sequence_list)
        ctx_true = ctx_true.reshape(-1, ctx_true.shape[-1])

        ctx_recon = np.concatenate(ctx_reconstructed_list)
        ctx_recon = ctx_recon.reshape(-1, ctx_recon.shape[-1])
        r2_ctx_recon = r2_score(ctx_true, ctx_recon)
        print(f"Test R2 Score = {r2} Test R2 Recon Score = {r2_recon}, R2 CTX Recon Score = {r2_ctx_recon}")
