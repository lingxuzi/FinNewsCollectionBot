import torch
import torch.nn.functional as F
import cv2 # 使用OpenCV进行图像处理
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 假设 StockChartNet_Attention 和 weights_init 函数已经定义好
# ... (此处省略之前已提供的 StockChartNet_Attention 模型代码) ...

# --- 1. Grad-CAM 核心逻辑实现 ---
class GradCAM:
    def __init__(self, model, target_layer, image_shape, forward_callback):
        self.model = model
        self.target_layer = target_layer
        self.image_shape = image_shape
        self.forward_callback = forward_callback
        self.feature_maps = None
        self.gradients = None
        
        # 注册钩子 (Hooks) 来捕获正向和反向传播中的数据
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor):
        """
        生成热力图
        :param input_tensor: 模型的输入张量, shape: (1, 1, 60, 60)
        :return: 标准化后的热力图, numpy array, shape: (60, 60)
        """
        self.model.train() # 切换到评估模式
        
        # 步骤 1: 正向传播
        # output = self.model(input_tensor)
        output = self.forward_callback(input_tensor, self.model)
        
        # 步骤 2: 反向传播
        self.model.zero_grad()
        # 对于回归任务，我们直接对输出值进行反向传播
        if isinstance(output, tuple):
            output = output[0]
        score = output.sum()
        score.backward()
        
        # 步骤 3: 计算权重 (全局平均池化梯度)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 步骤 4: 计算加权特征图
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        
        # 步骤 5: ReLU激活和尺寸调整
        cam = F.relu(cam) # 只关心有积极影响的区域
        
        # 将热力图上采样到与输入图像相同的尺寸
        cam = F.interpolate(cam, 
                            size=(self.image_shape[0], self.image_shape[1]), 
                            mode='bilinear', 
                            align_corners=False)
        
        # 步骤 6: 标准化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

# --- 2. 可视化函数 ---
def save_cam_on_image(img, heatmap, save_path):
    """
    将热力图叠加到原始图像上
    :param img_path: 原始图像文件路径
    :param heatmap: Grad-CAM 生成的热力图 (numpy array)
    """
    
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    img = img.reshape(img.shape[0], img.shape[1], -1)
    if img.shape[-1] == 1:
        _img = np.zeros_like(heatmap)
        for i in range(3):
            _img[:, :, i] = img[:, :, 0]
    
    # 将热力图与原图融合
    superimposed_img = heatmap * 0.4 + _img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    cv2.imwrite(save_path, np.uint8(255 * superimposed_img))

# --- 3. 使用示例 ---
if __name__ == '__main__':
    # a. 准备模型和数据
    model = StockChartNet_Attention()
    # 假设模型已经训练好并加载了权重
    # model.load_state_dict(torch.load("best_model.pth"))
    
    # b. 选择目标层
    # 理想的目标层是最后一个卷积层，因为它包含了最丰富的语义信息
    # 在我们的模型中，这是 self.conv2
    target_layer = model.conv2
    
    # c. 实例化 Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # d. 准备一张待分析的图像
    # 注意: 这里的预处理需要和训练时完全一致
    img_path = 'path_to_your_stock_image.png' # <--- 修改为您的图片路径
    raw_img = Image.open(img_path).convert('L') # 转为灰度图
    input_tensor = torch.from_numpy(np.array(raw_img)).unsqueeze(0).unsqueeze(0).float()
    input_tensor = input_tensor / 255.0 # 假设训练时做了归一化
    
    # e. 生成热力图
    heatmap = grad_cam(input_tensor)
    
    # f. 可视化
    print("生成热力图并进行可视化...")
    show_cam_on_image(img_path, heatmap)
