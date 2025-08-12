import torch
from diskcache import Cache
from ai.vision.price_trend.dataset import ImagingPriceTrendDataset
from tqdm import tqdm
import os
import random
import numpy as np

class TrendSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset: ImagingPriceTrendDataset, batch_size, tag):
        os.makedirs('../price_trend_sampler_cache', exist_ok=True)
        self._cache = Cache(f'../price_trend_sampler_cache_{tag}')
        self.batch_size = batch_size
        cls_indices = self._cache.get('trend_cls_indices', {})

        if len(cls_indices) == 0:
            for i in tqdm(range(len(dataset)), desc='Trend Sampler'):
                _, trend, _, _ = dataset.parse_item(i)

                if trend not in cls_indices:
                    cls_indices[trend] = []
                
                cls_indices[trend].append(i)
            
            self._cache.set('trend_cls_indices', cls_indices)
        self.cls_indices = cls_indices
        for _cls, indices in self.cls_indices.items():
            print(f'{_cls}: {len(indices)}')

        # self.samples_per_cls = batch_size // len(cls_indices)

        self.total = sum([len(indices) for indices in self.cls_indices.values()])

    def resample(self):
        indices = []
        min_len = sum([len(indices) for indices in self.cls_indices.values()]) // len(self.cls_indices)

        for trend, _indices in self.cls_indices.items():
            # print(_indices)
            if len(_indices) < min_len:
                indices.extend(_indices + random.sample(_indices, min_len - len(_indices)))
            else:
                indices.extend(random.sample(_indices, min_len))

        random.shuffle(indices)
        return np.asarray(indices).astype(int)

    def __iter__(self):
        # balanced sample indices from cls_indices
        indices = self.resample()
        while True:
            # batch = []
            # for trend in self.cls_indices:
            #     indices = random.sample(self.cls_indices[trend], self.samples_per_cls)
            #     batch.extend(indices)
            
            # random.shuffle(batch)
            if len(indices) < self.batch_size:
                _indices = self.resample()
                indices = np.concatenate([indices, _indices])
                print('resampled indices')
            
            select_indices = range(self.batch_size)
            batch = indices[select_indices]
            indices = np.delete(indices, select_indices)
            yield batch.tolist()

    def __len__(self):
        return self.total

            
