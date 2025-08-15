# dataset.py
import torch
import pandas as pd
import numpy as np
import os
import traceback
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from utils.common import read_text
from utils.lmdb import LMDBEngine
from tqdm import tqdm
from diskcache import FanoutCache
from collections import OrderedDict


class PriceToImgae:
    def __init__(self, days, width, height, price_area_height, volume_area_height, pixelwidth, pixelheight):
        self.days = days
        self.width = width
        self.height = height
        self.price_area_height = price_area_height
        self.volume_area_height = volume_area_height
        self.pixelwidth = pixelwidth
        self.pixelheight = pixelheight
        self.price_area_logical_height = int(self.price_area_height / self.pixelheight)
        self.volume_area_logical_height = int(self.volume_area_height / self.pixelheight)

    def __drawPixel(self, x, y, pixel):
        logical_height = int(self.height / self.pixelheight)
        for i in range(self.pixelwidth):
            for j in range(self.pixelheight):
                self.img.putpixel((self.pixelwidth * x + i, self.pixelheight * (logical_height - 1 - y) + j), pixel)

    def __drawPrice(self, index, price, moving_average, volume, pixel):
        open_price = price[0]
        high_price = price[1]
        low_price = price[2]
        close_price = price[3]

        # 画OHLC表
        self.__drawPixel(3 * index + 0, self.volume_area_logical_height + 1 + open_price, pixel)

        for i in range(high_price - low_price + 1):
            self.__drawPixel(3 * index + 1, self.volume_area_logical_height + 1 + low_price + i, pixel)
        self.__drawPixel(3 * index + 2, self.volume_area_logical_height + 1 + close_price, pixel)

        # 画MA线
        self.__drawPixel(3 * index + 0, self.volume_area_logical_height + 1 + moving_average, pixel)
        self.__drawPixel(3 * index + 1, self.volume_area_logical_height + 1 + moving_average, pixel)
        self.__drawPixel(3 * index + 2, self.volume_area_logical_height + 1 + moving_average, pixel)

        # 画成交量柱
        for i in range(volume):
            self.__drawPixel(3 * index + 1, i, pixel)

    def getImg(self, price_array, moving_average_array, volume_array, background_pixel, color_pixel):
        self.img = Image.new("RGB", (self.width, self.height), background_pixel)
        for i in range(price_array.shape[0]):
            self.__drawPrice(i, price_array[i], moving_average_array[i], volume_array[i], color_pixel)
        return self.img


def image_loader(image):
    transform = transforms.Compose([
        transforms.ToTensor()])
    image = transform(image).squeeze(0)
    return image


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = unloader(image)
    return image


def get_image_with_price(price):
    '''
    price [time, feature]
    feature: open, close, low, high, vol, MA
    '''
    try:
        # 像素大小
        PIXEL_WIDTH = 1
        PIXEL_HEIGHT = 1
        # 宽度是时间序列的三倍长
        WIDTH = 3 * price.shape[0] * PIXEL_WIDTH

        # 价格占高度2/3，vol占1/3
        PRICE_LOGICAL_HEIGHT = 2 * price.shape[0]
        VOLUME_LOGICAL_HEIGHT = price.shape[0]

        # 计算区域各区域大小
        PRICE_AREA_HEIGHT = PRICE_LOGICAL_HEIGHT * PIXEL_HEIGHT
        V0LUME_AREA_HEIGHT = VOLUME_LOGICAL_HEIGHT * PIXEL_HEIGHT

        # 总高度还是加一个pixel大小分割
        HEIGHT = PRICE_AREA_HEIGHT + V0LUME_AREA_HEIGHT + PIXEL_HEIGHT

        # 放缩
        sclr1 = MinMaxScaler((0, PRICE_LOGICAL_HEIGHT - 1))
        sclr2 = MinMaxScaler((1, VOLUME_LOGICAL_HEIGHT))
        price_minmax = sclr1.fit_transform(price[:, :-1].reshape(-1, 1)).reshape(price.shape[0], -1).astype(int)
        volume_minmax = sclr2.fit_transform(price[:, -1].reshape(-1, 1)).reshape(price.shape[0]).astype(int)

        # 时间序列长度
        days = price_minmax.shape[0]

        # 转图片
        p2i = PriceToImgae(days, WIDTH, HEIGHT, PRICE_AREA_HEIGHT, V0LUME_AREA_HEIGHT, PIXEL_WIDTH, PIXEL_HEIGHT)
        background_pixel = (0, 0, 0, 100)
        color_pixel = (255, 255, 255, 100)
        image = p2i.getImg(price_minmax[:, :-1], price_minmax[:, -1], volume_minmax, background_pixel, color_pixel)
        # 转成黑白像素
        image = image.convert('1')
        return image
    except Exception as e:
        traceback.print_exc()

class ImagingPriceTrendDataset(Dataset):
    def __init__(self, db_path, img_caching_path, stock_list_file, hist_data_file, seq_length, features, ts_features, encoder, image_size, tag, is_train=True):
        super().__init__()
        self.image_size = image_size
        self.seq_length = seq_length
        self.features = features
        self.ts_features = ts_features
        self.is_train = is_train
        self.indus_encoder, self.stock_encoder = encoder
        self.img_caching_path = img_caching_path
        os.makedirs(self.img_caching_path, exist_ok=True)

        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        self.cache = FanoutCache(os.path.join(db_path, f'{tag}', 'cache'), shards=32, timeout=5, size_limit=3e11, eviction_policy='none')
        
        if self.cache.get('total_count', 0) == 0:
            # 1. 从数据库加载数据
            self.onehot = OrderedDict()
            all_data_df = pd.read_parquet(hist_data_file)
            stock_list = read_text(stock_list_file).split(',')

            all_data_df['month'] = all_data_df['date'].dt.month / 12
            all_data_df['day'] = all_data_df['date'].dt.day / 31
            all_data_df['weekday'] = all_data_df['date'].dt.weekday + 1 / 7

            images = []
            trends = []
            codes = []
            industries = []
            ts_sequences = []
            ctx_sequences = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.generate_sequence_imgs, all_data_df[all_data_df['code'] == code], code): code for code in stock_list}
                for future in tqdm(as_completed(futures), total=len(futures), ncols=120, desc='generate sequence imgs'):
                    code = futures[future]
                    if self.onehot.get(code) is None:
                        self.onehot[code] = len(self.onehot)
                    
                    code = self.stock_encoder.transform([code])[0]
                    try:
                        _images, _trends, industry, ts_sequences, ctx_sequences = future.result()
                        
                        images.extend(_images)
                        trends.extend(_trends)
                        codes.extend([code] * len(_images))
                        industries.extend([industry] * len(_images))
                        ts_sequences.extend(ts_sequences)
                        ctx_sequences.extend(ctx_sequences)
                    except Exception as e:
                        print(e)
            self.cache.set('total_count', len(images))
            for i, (img, trend, code, industry, ts_seq, ctx_seq) in tqdm(enumerate(zip(images, trends, codes, industries, ts_sequences, ctx_sequences))):
                self.cache.set(i, (img, trend, code, industry, ts_seq, ctx_seq))

    def accumulative_return(self, returns):
        return np.prod(1 + returns) - 1
    
    def generate_sequence_imgs(self, stock_data, code):
        print('preparing sequences for ', code)
        label_return_cols = []
        for i in range(5):
            label_return_cols.append(f'label_return_{i+1}')
        
        industry= self.indus_encoder.transform(stock_data['industry'].to_numpy())[0]
        price_data = stock_data[self.features].to_numpy()
        labels = stock_data[label_return_cols].to_numpy()

        ts_featured_stock_data = stock_data[self.ts_features['features'] + self.ts_features['temporal']].to_numpy()
        ts_numerical_stock_data = stock_data[self.ts_features['numerical']].to_numpy()

        returns = []
        imgs = []
        ts_sequences = [] # 时间序列部分
        ctx_sequences = [] # 上下文部分
        for i in range(0, len(stock_data) - self.seq_length + 1, 3):
            ts_seq = price_data[i:i + self.seq_length]
            if len(ts_seq) < self.seq_length:
                return None, None, None, None, None
            
            img_path = os.path.join(self.img_caching_path, code, f'{i}.png')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            if not os.path.isfile(img_path):
                image = get_image_with_price(ts_seq)
                image.save(img_path)
            
            # 时间序列部分
            ts_sequences.append(ts_featured_stock_data[i:i + self.seq_length])
            # 上下文部分
            ctx_sequences.append(ts_numerical_stock_data[i + self.seq_length - 1])

            returns.append(labels[i + self.seq_length - 1])
            imgs.append(img_path)
        return imgs, returns, industry, ts_sequences, ctx_sequences

    def __len__(self):
        return self.cache.get('total_count', 0)
    
    def parse_item(self, idx):
        image, returns, code, industry, ts_seq, ctx_seq = self.cache.get(idx)

        acu_return = self.accumulative_return(returns)
        # if acu_return > 0.01:
        #     _trend = 2
        # elif -0.01 < acu_return <= 0.01:
        #     _trend = 1
        # elif acu_return <= -0.01:
        #     _trend = 0
        if acu_return > 0.01:
            _trend = 1
        else:
            _trend = 0
        
        return image, _trend, acu_return, code, industry, ts_seq, ctx_seq
    
    def __getitem__(self, idx):
        image, _trend, returns, code, industry, ts_seq, ctx_seq = self.parse_item(idx)
        image = Image.open(image)
        image = self.transforms(image)

        ts_minmaxer = MinMaxScaler()
        ctx_minmaxer = MinMaxScaler()
        ts_seq = ts_minmaxer.fit_transform(ts_seq)
        ctx_seq = ctx_minmaxer.fit_transform(ctx_seq)

        return image, torch.LongTensor([_trend]), torch.FloatTensor([returns]), torch.LongTensor([code]), torch.LongTensor([industry]), torch.FloatTensor(ts_seq), torch.FloatTensor(ctx_seq)
