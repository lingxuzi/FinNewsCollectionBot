import warnings
import psutil
import lmdb
import numpy as np
import pickle

MAX_SIZE = int(1e12)

class LMDBEngine:
    def __init__(self, lmdb_path, readonly=False):
        if not readonly:
            self.env = lmdb.open(lmdb_path, map_size=MAX_SIZE, map_async=True, writemap=True, meminit=False)
        else:
            self.env = lmdb.open(lmdb_path, readonly=True, max_readers=10, lock=False, readahead=False, meminit=False)
        
        if readonly:
            self.txn = self.env.begin()
        else:
            self.txn = self.env.begin(write=True)
            
    @property
    def is_readonly(self):
        return self.env.readonly
    
    def commit(self):
        self.txn.commit()
        self.txn = None
        self.txn = self.env.begin(write=True)
    
    def put(self, key, val):
        self.txn.put(key.encode(), pickle.dumps(val))

    def get(self, key):
        return pickle.loads(self.txn.get(key.encode())) if self.txn.get(key.encode()) else None

    def close(self):
        if self.txn:
            self.txn.abort()
        self.env.close()