import torch
from diskcache import Cache
from ai.vision.price_trend.dataset import ImagingPriceTrendDataset
from tqdm import tqdm
import os
import random

class TrendSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset: ImagingPriceTrendDataset, batch_size):
        os.makedirs('../price_trend_sampler_cache', exist_ok=True)
        self._cache = Cache('../price_trend_sampler_cache')
        self.total = len(dataset)

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

        self.samples_per_cls = batch_size // len(cls_indices)

    def __iter__(self):
        # balanced sample indices from cls_indices
        while True:
            batch = []
            for trend in self.cls_indices:
                indices = random.sample(self.cls_indices[trend], self.samples_per_cls)
                batch.extend(indices)
            
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.total

            
