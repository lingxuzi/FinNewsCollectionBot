from concurrent.futures import thread
import torch
import queue
import time
import traceback
import numpy as np

class DataPrefetcher():
    def __init__(self, loader, device, enable_queue=False, num_threads=1):
        self.loader = iter(loader)
        self.device = device
        if self.device.startswith('cuda'):
            self.stream = torch.cuda.Stream()
        self.next_data = None
        self.enable_queue = enable_queue
        self.preload_time = 0


        self.preload()

        if enable_queue:
            pass
            # self.queue = queue.Queue(8)
            # self.locker = threading.Lock()
            # self.threads = []

            # for _ in range(num_threads):
            #     self.threads.append(threading.Thread(target=self.queue_process, args=()))

            # for t in self.threads:
            #     t.daemon = True
            #     t.start()
                
            # self.preload_worker = threading.Thread(target=self.preload, args=())
            # self.preload_worker.daemon = True
            # self.preload_worker.start()

    def queue_process(self):
        while True:
            if not self.queue.full():
                data = self.__next()
                if data is not None:
                    self.queue.put(data, block=True)
                else:
                    break
            else:
                time.sleep(1/30)

    def __next(self):
        if hasattr(self, 'locker'):
            with self.locker:
                try:
                    return next(self.loader)
                except Exception as e:
                    print(e)
                    return None
        else:
            try:
                return next(self.loader)
            except Exception as e:
                traceback.print_exc()
                return None

    def preload(self):
        t = time.time()
        self.next_data = None
        if self.enable_queue and not self.queue.empty():
            self.next_data = self.queue.get(block=True)
            if self.next_data is None:
                self.next_data = self.__next()
        else:
            self.next_data = self.__next()

        if self.next_data is not None:
            self.stream_data(self.next_data)
        self.preload_time = (time.time() - t) * 1000

    def size(self):
        if self.enable_queue:
            return self.queue.qsize()
        else:
            return 0
        
    def parse_data_to_cuda(self, data):
        if isinstance(data, list):
            for i, k in enumerate(data):
                if isinstance(k, torch.Tensor):
                    data[i] = k.cuda(non_blocking=True)
                # elif isinstance(k, list) or isinstance(k, tuple):
                #     data[i] = [kk.cuda(non_blocking=True) for kk in k]
                else:
                    data[i] = self.parse_data_to_cuda(data[i])
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda(non_blocking=True)
                else:
                    data[k] = self.parse_data_to_cuda(v)
        elif isinstance(data, tuple):
            data = tuple(self.parse_data_to_cuda(item) for item in data)
        elif isinstance(data, str):
            pass
        else:
            data = data.cuda(non_blocking=True)
        return data
    
    def stream_data(self, data):
        if hasattr(self, 'stream'):
            with torch.cuda.stream(self.stream):
                data = self.parse_data_to_cuda(data)
            
    def next(self):
        if self.device.startswith('cuda'):
            torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
    
    def clean(self):
        self.stream.close()