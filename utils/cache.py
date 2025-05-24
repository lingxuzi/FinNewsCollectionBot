from diskcache import Cache, Lock
from pandas import DataFrame
import time
import random

cache = Cache('../stock_cache')

def run_with_cache(func, *args, **kwargs):
    cache_key = f"{func.__name__}_{args}_{kwargs}"
    if cache_key in cache:
        return cache[cache_key]
    time.sleep(random.random() / 2)
    with Lock(cache, 'lock_' + cache_key, expire=60) as locker:
        if cache_key in cache:
            return cache[cache_key]
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