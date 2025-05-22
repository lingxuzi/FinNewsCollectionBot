from diskcache import Cache, Lock
import time
import random

cache = Cache('../stock_cache')

def run_with_cache(func, *args, **kwargs):
    cache_key = f"{func.__name__}_{args}"
    if cache_key in cache:
        return cache[cache_key]
    time.sleep(random.random())
    with Lock(cache, 'lock_' + cache_key, expire=60) as locker:
        if cache_key in cache:
            return cache[cache_key]
        result = func(*args, **kwargs)
        expired = kwargs.get('expired', None)
        cache.set(cache_key, result, expired)
        return result