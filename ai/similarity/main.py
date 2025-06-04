import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from sklearn.cluster import KMeans
from ta import add_all_ta_features
from datetime import datetime, timedelta
import faiss
import akshare as ak
from tqdm import tqdm
import os
import io
from PIL import Image
import torchvision.transforms as transforms
from utils.cache import run_with_cache

# 配置参数
IMG_SIZE = 320  # 图像大小调整为192x192
FIG_SAVE_PATH = '../stock_embedding_train'
os.makedirs(FIG_SAVE_PATH, exist_ok=True)

# 1. 数据获取与预处理 (保持不变)
def get_stock_data(stock_code):
    """使用akshare获取股票数据"""
    df = run_with_cache(ak.stock_zh_a_hist, symbol=stock_code, period="daily", adjust='qfq')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期')
    df = df[['开盘', '最高', '最低', '收盘', '成交量']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def add_technical_indicators(df):
    """添加技术分析指标"""
    df = df.copy()
    df = add_all_ta_features(
        df, open="open", high="high", low="low", 
        close="close", volume="volume", fillna=True
    )
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    return df

def get_pattern_label(window_df):
    """识别K线形态"""
    closes = window_df['close'].values
    opens = window_df['open'].values
    if all(closes[i] >= closes[i-1] for i in range(1, len(closes))):
        return 'uptrend'
    elif all(closes[i] <= closes[i-1] for i in range(1, len(closes))):
        return 'downtrend'
    elif closes[-1] > opens[-1] and np.mean(closes[-3:]) > np.mean(opens[-3:]):
        return 'bullish'
    elif closes[-1] < opens[-1] and np.mean(closes[-3:]) < np.mean(opens[-3:]):
        return 'bearish'
    else:
        return 'neutral'

def preprocess_data(code, df, window_size=50):
    """数据预处理和滑动窗口切分（包含所有技术指标）"""
    # 添加所有技术指标
    # df = add_technical_indicators(df)
    non_numeric_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
    feat_cols = [col for col in df.columns if col not in non_numeric_cols]
    
    # 处理缺失值
    for col in feat_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0)
    
    # 准备多线程参数
    window_indices = range(len(df) - window_size + 1)
    n_windows = len(window_indices)
    
    # 创建线程安全的临时存储
    sequences = [None] * n_windows
    date_ranges = [None] * n_windows
    patterns = [None] * n_windows
    img_paths = [None] * n_windows
    
    # 定义每个窗口的处理函数
    def process_window(i, code, df, window_size):
        window_df = df.iloc[i:i+window_size]
        seq = window_df[feat_cols].to_numpy()
        date_range = (df.index[i], df.index[i+window_size-1])
        pattern = get_pattern_label(window_df)
        img_path = create_kline_chart(seq, code, f'{str(df.index[i])}-{str(df.index[i+window_size-1])}')
        return i, seq, date_range, pattern, img_path
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 使用partial固定部分参数
        process_func = partial(process_window, code=code, df=df, window_size=window_size)
        
        # 使用tqdm显示进度条
        futures = []
        for i in window_indices:
            futures.append(executor.submit(process_func, i))
        
        # 收集结果
        for future in tqdm(futures, desc=f"处理股票 {code}", total=n_windows):
            i, seq, date_range, pattern, img_path = future.result()
            sequences[i] = seq
            date_ranges[i] = date_range
            patterns[i] = pattern
            img_paths[i] = img_path
    
    return np.array(sequences), date_ranges, patterns, img_paths

# 2. K线图生成 (调整为192x192)
def create_kline_chart(data, code, ts_range):

    save_path = os.path.join(FIG_SAVE_PATH, code)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, ts_range + '.jpg')
    if os.path.isfile(save_path):
        return save_path
    
    """生成192x192的K线图"""
    fig, ax = plt.subplots(figsize=(IMG_SIZE/100, IMG_SIZE/100), dpi=100)
    
    colors = ['red' if data[i, 3] >= data[i-1, 3] else 'green' for i in range(1, len(data))]
    colors.insert(0, 'red')
    
    for i in range(len(data)):
        ax.plot([i, i], [data[i, 2], data[i, 1]], color=colors[i], linewidth=1)
        ax.plot([i-0.2, i+0.2], [data[i, 0], data[i, 0]], color=colors[i], linewidth=2)
        ax.plot([i-0.2, i+0.2], [data[i, 3], data[i, 3]], color=colors[i], linewidth=2)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img.save(save_path)
    return save_path

# 3. 轻量级深度学习模型
class LightweightKLineModel(nn.Module):
    def __init__(self, embedding_dim=64):
        super(LightweightKLineModel, self).__init__()
        
        # 轻量级CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 96x96
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 64x1x1
        )
        
        # 轻量级序列处理
        self.sequence_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, image, sequence):
        # 图像特征提取
        img_feat = self.cnn(image).squeeze()
        
        # 序列特征提取
        seq_feat = self.sequence_encoder(sequence.mean(dim=1))  # 使用均值简化
        
        # 特征融合
        combined = torch.cat([img_feat, seq_feat], dim=1)
        return self.fusion(combined)

class KLineDataset(Dataset):
    def __init__(self, sequences, transform=None):
        self.sequences = sequences
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        img = create_kline_chart(self.sequences[idx])
        return self.transform(img), torch.FloatTensor(self.sequences[idx])

def train_lightweight_model(sequences, epochs=20, batch_size=128):
    """训练轻量级模型"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = KLineDataset(sequences, transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LightweightKLineModel(embedding_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, seqs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            anchors = model(images, seqs)
            
            # 生成正负样本
            batch_size = anchors.size(0)
            pos_indices = torch.clamp(torch.arange(batch_size) + torch.randint(-3, 4, (batch_size,)), 0, batch_size-1)
            neg_indices = torch.randint(0, batch_size, (batch_size,))
            
            positives = model(images[pos_indices], seqs[pos_indices])
            negatives = model(images[neg_indices], seqs[neg_indices])
            
            loss = criterion(anchors, positives, negatives)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        scheduler.step()
        print(f"Loss: {total_loss/len(train_loader):.4f}")
    
    return model

# 4. 向量数据库与查询 (保持不变)
class TimeAwareVectorDatabase:
    def __init__(self, model, dim=64, decay_rate=0.99):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.model = model
        self.model.eval()
        self.decay_rate = decay_rate
        self.timestamps = []
        
    def add_embedding(self, sequence, metadata):
        with torch.no_grad():
            img = create_kline_chart(sequence)
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            embedding = self.model(img_tensor, seq_tensor).squeeze().numpy()
            
        self.index.add(embedding.reshape(1, -1))
        self.metadata.append(metadata)
        self.timestamps.append(metadata['date_range'][1])
        
    def search(self, query_sequence, k=5, time_weight=True):
        with torch.no_grad():
            query_img = create_kline_chart(query_sequence)
            query_img_tensor = transforms.ToTensor()(query_img).unsqueeze(0)
            query_seq_tensor = torch.FloatTensor(query_sequence).unsqueeze(0)
            query_embedding = self.model(query_img_tensor, query_seq_tensor).squeeze().numpy()
            
        distances, indices = self.index.search(query_embedding.reshape(1, -1), len(self.metadata))
        
        if time_weight:
            query_date = datetime.now()
            for i, idx in enumerate(indices[0]):
                days_passed = (query_date - self.timestamps[idx]).days
                time_decay = self.decay_rate ** days_passed
                distances[0][i] *= (1 / time_decay)
        
        results = []
        for i in range(min(k, len(indices[0]))):
            idx = indices[0][i]
            results.append({
                'similarity': 1 / (1 + distances[0][i]),
                'date_range': self.metadata[idx]['date_range'],
                'stock_code': self.metadata[idx]['stock_code'],
                'pattern': self.metadata[idx]['pattern'],
                'sequence': self.metadata[idx]['sequence'],
                'days_passed': (datetime.now() - self.timestamps[idx]).days
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
def load_data(stock_codes):
    # stock_codes = ["600000", "000858", "601318"]
    all_sequences = []
    all_date_ranges = []
    all_stock_codes = []
    all_patterns = []
    
    for code in stock_codes:
        print(f"处理股票 {code}...")
        df = get_stock_data(code)
        sequences, date_ranges, patterns, img_paths = preprocess_data(code, df)
        all_sequences.extend(sequences)
        all_date_ranges.extend(date_ranges)
        all_stock_codes.extend([code] * len(sequences))
        all_patterns.extend(patterns)

    return all_sequences, all_date_ranges, all_stock_codes, all_patterns

# 5. 主流程
def main():
    # 数据准备
    all_sequences, all_date_ranges, all_stock_codes, all_patterns = run_with_cache(load_data, ["600000", "000858", "601318"])
    
    # 训练轻量级模型
    print("训练轻量级模型...")
    model = train_lightweight_model(all_sequences)
    
    # 构建向量数据库
    print("构建向量数据库...")
    vdb = TimeAwareVectorDatabase(model)
    for seq, date_range, code in zip(all_sequences, all_date_ranges, all_stock_codes):
        pattern = get_pattern_label(pd.DataFrame(
            seq, columns=['open', 'high', 'low', 'close', 'volume']
        ))
        vdb.add_embedding(seq, {
            'date_range': date_range,
            'stock_code': code,
            'pattern': pattern,
            'sequence': seq
        })
    
    # 查询示例
    print("\n执行查询...")
    query_code = "000001"
    query_df = get_stock_data(query_code)
    query_seq, _, _ = preprocess_data(query_df)
    query_seq = query_seq[-1]  # 最近50天
    
    results = vdb.search(query_seq, k=5)
    
    # 可视化展示
    plt.figure(figsize=(20, 15))
    
    # 查询K线
    plt.subplot(3, 2, 1)
    create_kline_chart(query_seq)
    plt.title(
        f"查询K线 - {query_code}\n"
        f"{query_df.index[-50].date()} 至 {query_df.index[-1].date()}"
    )
    
    # 相似结果
    for i, result in enumerate(results):
        plt.subplot(3, 2, i+2)
        create_kline_chart(result['sequence'])
        plt.title(
            f"相似度: {result['similarity']:.3f}\n"
            f"{result['date_range'][0].date()} 至 {result['date_range'][1].date()}\n"
            f"股票: {result['stock_code']} | 形态: {result['pattern']}"
        )
    
    plt.tight_layout()
    plt.show()
    
    # 输出结果
    print("\n查询结果摘要:")
    print("\n最相似历史模式:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['stock_code']} {res['date_range'][0].date()}至{res['date_range'][1].date()} "
              f"(相似度:{res['similarity']:.3f}, 形态:{res['pattern']})")

if __name__ == "__main__":
    main()