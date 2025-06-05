from diskcache import FanoutCache, Lock
from pandas import DataFrame
import time
import random

cache = FanoutCache('../stock_cache')

def run_with_cache(func, *args, **kwargs):
    cache_key = f"{func.__name__}_{args}_{kwargs}"
    # if cache_key in cache:
    #     return cache[cache_key]
    result = cache.get(cache_key, None)
    if result is not None:
        return result
    time.sleep(random.random() / 2)
    with Lock(cache, 'lock_' + cache_key, expire=60) as locker:
        result = cache.get(cache_key, None)
        if result is not None:
            return result
        expired = kwargs.get('expired', None)
        result = func(*args, **kwargs)
        if isinstance(result, DataFrame):
            if result.empty:
                return None
        else:
            if not result:
                return None
        cache.set(cache_key, result, expired)
        return result