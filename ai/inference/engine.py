import torch
import threading
import queue
import time
import numpy as np
import GPUtil  # 用于 GPU 共享 (需要安装: pip install GPUtil)
import os

from utils.singleton_wrapper import Singleton
from collections import defaultdict
from ai.embedding.models import create_model

class AsynchronousInferenceEngine(Singleton):
    """
    异步推理引擎，支持动态批次大小、优先级队列、GPU 共享和模型缓存。
    """

    class _Request:  # 内部类，封装请求数据和 future
        def __init__(self, model_name, data, priority=0):
            self.data = data
            if isinstance(data, list) or isinstance(data, tuple):
                self.batch = 1
            self.model_name = model_name
            self.priority = priority
            self.future = queue.Queue(maxsize=1)

        def set_result(self, result):
            self.future.put(result)

        def get_result(self, timeout=None):
            try:
                result = self.future.get(timeout=timeout)
                return result
            except queue.Empty:
                raise TimeoutError("Timeout waiting for result.")

        def __lt__(self, other): #定义优先级
            return self.priority < other.priority

    def __init__(self, device='cpu', batch_size=32, num_workers=4, max_queue_size=128,
                 dynamic_batching=True, max_dynamic_batch_size=64, use_priority_queue=True,
                 gpu_sharing=False, model_cache_size=1): #模型缓存大小，默认为1
        """
        初始化异步推理引擎。
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.dynamic_batching = dynamic_batching
        self.max_dynamic_batch_size = max_dynamic_batch_size
        self.use_priority_queue = use_priority_queue
        self.gpu_sharing = gpu_sharing
        self.model_cache_size = model_cache_size

        self.model_cache = {} #模型缓存
        self.model_batch_collate_fn = {} #模型的batch collate函数
        self.model_cache_lock = threading.Lock() #保护模型缓存

        if self.use_priority_queue:
            self.request_queue = queue.PriorityQueue(maxsize=self.max_queue_size)
        else:
            self.request_queue = queue.Queue(maxsize=self.max_queue_size)

        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()
        self.lock = threading.Lock()

        self.available_gpu = None
        if self.gpu_sharing and self.device.type == 'cuda':
            self.available_gpu = self._get_available_gpu()

    def _get_available_gpu(self):
        """
        获取当前可用的 GPU。 (需要 GPUtil)
        """
        try:
            #GPUs = GPUtil.getGPUs()
            #if GPUs:
            #    return GPUs[0].id #返回第一个可用的GPU id
            device_id = os.environ.get("CUDA_VISIBLE_DEVICES")
            if device_id:
                return int(device_id)
            else:
                gpus = GPUtil.getAvailable(order='memory', limit=1)
                return gpus[0]
            #else:
            #    return None
        except Exception as e:
            print(f"Error getting available GPU: {e}")
            return None

    def _load_model(self, model_name, model_config, ckpt_path):
        """
        加载 PyTorch 模型。
        """
        try:
            # model = torch.load(model_path, map_location=self.device)
            model = create_model(model_name, model_config)
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt, strict=False)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {ckpt_path}: {e}")

    def _load_model_into_cache(self, model_name, model_config, ckpt_path, batch_collate_fn):
        """
        将模型加载到缓存中。
        """
        with self.model_cache_lock:
            if model_name not in self.model_cache:
                if len(self.model_cache) >= self.model_cache_size:
                    # 移除最久未使用的模型 (LRU)
                    oldest_model_path = next(iter(self.model_cache))
                    del self.model_cache[oldest_model_path]
                self.model_cache[model_name] = self._load_model(model_name, model_config, ckpt_path)
                self.model_batch_collate_fn[model_name] = batch_collate_fn

    def inference(self, model_name, data, priority=0):
        """
        提交推理请求。

        Args:
            data (np.ndarray): 输入数据。
            priority (int): 请求的优先级 (越大越高)。

        Returns:
            object: _Request 对象。
        """
        request = AsynchronousInferenceEngine._Request(model_name, data, priority=priority)
        try:
            if self.use_priority_queue:
                 self.request_queue.put(request)
            else:
                self.request_queue.put(request, timeout=1)  # 如果队列满了，等待1秒
        except queue.Full:
            raise RuntimeError("Request queue is full.  Increase max_queue_size or reduce request rate.")
        return request.get_result()

    def _prepare_tensor(self, data, device):
        """
        将输入数据转换为 PyTorch 张量。
        """
        # 在这里添加对不同数据类型的处理逻辑
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32).to(device)
        elif isinstance(data, list):  # 假设列表中的元素都是相同类型的 numpy 数组
            tensor_list = [torch.tensor(item, dtype=torch.float32).to(device) for item in data]
            return tensor_list #将list转换为tensor
        elif isinstance(data, tuple):
            tensor_list = [torch.tensor(item, dtype=torch.float32).to(device) for item in data]
            return tensor_list
        else:
            # 对于其他类型的数据，需要根据具体的模型输入格式进行处理
            raise ValueError(f"Unsupported data type: {type(data)}")
        
    def _process_batch(self, model_name, batch_requests):
        """
        处理单个模型的批次。
        """
        # 获取模型
        with self.model_cache_lock:
            model = self.model_cache[model_name]
        # 使用 _collate_fn 整理批次数据
        collated_data = self.model_batch_collate_fn[model_name]([batch_request.data for batch_request in batch_requests])
        # GPU 共享上下文
        if self.gpu_sharing and self.device.type == 'cuda' and self.available_gpu is not None:
            with torch.cuda.device(self.available_gpu):  # 设置当前gpu
                # 将数据转换为 PyTorch 张量
                input_tensor = self._prepare_tensor(collated_data, self.device)
                # 执行批量推理
                with torch.no_grad():
                    predictions = model(input_tensor)
                    predictions = predictions.cpu().numpy()
        else:
            # 将数据转换为 PyTorch 张量
            input_tensor = self._prepare_tensor(collated_data, self.device)
            # 执行批量推理
            with torch.no_grad():
                predictions = model(*input_tensor)
        # 将结果分发给相应的请求
        cursor = 0
        for i, request in enumerate(batch_requests):
            if isinstance(predictions, list) or isinstance(predictions, tuple):
                request.set_result([pred[cursor:cursor+request.batch] for pred in predictions])  # 将结果放入对应的queue
                cursor += request.batch
            else:
                request.set_result(predictions[cursor:cursor+request.batch])  # 将结果放入对应的queue
                cursor += request.batch
        # 防止内存泄漏
        del collated_data, input_tensor, predictions

    def _inference_worker(self):
        """
        推理工作线程。
        """
        while True:
            # 按模型路径分组请求
            model_batches = defaultdict(list)
            # 收集请求直到达到最大批次大小或队列为空
            total_requests = 0
            while total_requests < self.max_dynamic_batch_size if self.dynamic_batching else self.batch_size:
                try:
                    request = self.request_queue.get(timeout=0.1)  # 0.1秒超时
                    model_batches[request.model_name].append(request)
                    total_requests += 1
                except queue.Empty:
                    break  # 队列为空
            # 处理每个模型的批次
            for model_name, batch_requests in model_batches.items():
                self._process_batch(model_name, batch_requests)
            # 如果没有请求，则休眠一段时间
            if total_requests == 0:
                time.sleep(0.01)

    def shutdown(self):
        """
        关闭推理引擎。
        """
        self.worker_thread.join(timeout=1)  # 等待1秒
        print("Inference engine shutdown.")
