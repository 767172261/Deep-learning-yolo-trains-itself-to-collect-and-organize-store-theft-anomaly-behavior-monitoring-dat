深度学习yolo训练自己收集整理的商店偷盗异常行为监测数据集 建立基于深度学习yolov8商店超市客户行为监测系统
@[toc]
以下文字及代码仅供参考。
商店异常行为检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41b7a01280b842c7a67eb27e9a6d8815.png)
数据集收集情况预览：

10445张，yolo和voc标注
6类，标注数量：
emptyhands - 空手 ： 5531
shoplifting - 商店盗窃 ： 3984
Looking around - 环顾四周 ： 1798
Picking-Holding - 拾取持有 ： 2605
Basket - 篮子 ： 472
Robbery - 抢劫 ： 131
image num -： 10445

针对商店异常行为检测数据集，我们可以通过YOLOv8进行训练和评估。该数据集包含10445张图像，涵盖了6类行为：空手、商店盗窃、环顾四周、拾取持有、篮子和抢劫。数据集标注使用了YOLO和VOC两种格式。

### 数据准备

首先确保你的数据集已准备好，并按照以下结构组织：

```
store_behavior_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── yolo_format/
    │   ├── train/
    │   └── val/
    └── voc_format/
        ├── train/
        └── val/
```

- `images/train/` 和 `images/val/` 分别存放训练和验证图像。
- `labels/yolo_format/train/` 和 `labels/yolo_format/val/` 存放对应图像的YOLO格式标签文件。
- `labels/voc_format/train/` 和 `labels/voc_format/val/` 存放对应图像的VOC格式标签文件。

### 数据配置

创建一个名为 `data.yaml` 的文件来描述数据集的路径和类别信息：

```yaml
train: ./store_behavior_dataset/images/train/
val: ./store_behavior_dataset/images/val/

nc: 6 # 类别数量
names: ['emptyhands', 'shoplifting', 'Looking around', 'Picking-Holding', 'Basket', 'Robbery'] # 类别名称
```

### 环境搭建

安装必要的库：

```bash
pip install ultralytics opencv-python
```

### 模型训练

使用YOLOv8训练模型：

```bash
yolo detect train model=yolov8n.pt data=data.yaml batch=16 epochs=100 imgsz=640 device=cuda
```

这里，`epochs` 设置为100，可以根据需要调整。`device=cuda` 表示使用GPU加速训练。

### 超参数配置

在训练过程中，可以通过修改 `cfg/training/yolov8n.yaml` 文件中的超参数来优化模型性能。例如，调整学习率、batch size等。

### 模型推理

训练完成后，可以使用以下命令对新图像进行推理：

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/test/images/ imgsz=640 device=cuda
```

### 批量推理代码示例

如果你想要编写自己的批量推理脚本，可以参考以下Python代码：

```python
from ultralytics import YOLO
import cv2
import os

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

def infer_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            results = model.predict(source=image_path, imgsz=640, save=True)

            for r in results:
                print(f"Prediction for {filename}:")
                print(r.boxes)  # 输出预测结果

# 定义测试图像目录
test_dir = 'path/to/test/images/'
infer_images_in_folder(test_dir)
```

### 性能评估

YOLOv8自带了性能评估工具，可以在训练结束后自动计算mAP（mean Average Precision）、Precision、Recall等指标。你也可以手动运行评估：

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640
```

。

使用YOLOv8构建一个针对商店异常行为检测的深度学习系统。。
