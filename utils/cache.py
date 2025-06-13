from diskcache import FanoutCache, Lock
from pandas import DataFrame
import re
import time
import random

cache = FanoutCache('../stock_cache', shards=16, timeout=5, size_limit=3e11, eviction_policy='none')

def run_with_cache(func, *args, **kwargs):
    args_str = re.sub(r'\<.*?\>', '', str(args))
    cache_key = f"{func.__qualname__}_{args_str}_{kwargs}"
    # if cache_key in cache:
    #     return cache[cache_key]
    result = cache.get(cache_key, None)
    if result is not None:
        return result
    time.sleep(random.random() / 2)
    
    expired = kwargs.get('expired', None)
    result = func(*args, **kwargs)
    if isinstance(result, DataFrame):
        if result.empty:
            return None
    else:
        if not result:
            return None
    t = time.time()
    if cache.set(cache_key, result, expired):
        print(f'{cache_key} cached -> {time.time() - t:.2f}s')
    else:
        print(f'{cache_key} cached failed')
    return result
    
def cache_decorate(func):
    def decorate(*args, **kwargs):
        return run_with_cache(func, *args, **kwargs)
    return decorate
