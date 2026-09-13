# trainhub

基于 PyQt6 的一站式「标注 → 训练」工作站。内置 labelme 兼容的标注引擎，以及 YOLO / nnU-Net 双训练后端，计算设备全平台自动适配（CUDA / MPS / CPU），训练器通过插件体系可自由扩展。

## 功能特性

- **数据集管理**：批量导入图像文件 / 文件夹；自动识别 labelme JSON 与 YOLO 数据集（`images/` + `labels/` + `dataset.yaml`）；类别分布统计与标注进度一目了然。
- **标注**：移植自 labelme 6.3.0，支持矩形框、多边形、定向矩形、圆形、点、线、折线等形状标注，产出与 labelme 完全兼容的 JSON。
- **训练**：YOLO（目标检测 / 实例分割 / 图像分类）与 nnU-Net（2D 医学图像分割），可视化调参，训练时实时绘制 loss / 指标 / 学习率曲线，自动收集日志与产物，可回看历史运行。
- **数据转换**：labelme → YOLO 一键导出；导入 YOLO 数据集时自动反向转换为 labelme 格式。
- **可扩展**：新增训练器只需实现 `BaseTrainer` 并 `@register_trainer` 注册，界面与调参表单自动生成。

## 界面预览

**数据集页**

![数据集页](docs/images/dataset.png)

**标注页**

![标注页](docs/images/annotate.png)

**训练页**

![训练页](docs/images/train.png)

## 平台与 torch 版本明细

| 平台 | 加速后端 | 推荐 torch | 推荐 torchvision | 说明 |
|------|---------|-----------|-----------------|------|
| macOS（Apple Silicon，M1–M5） | MPS | ≥ 2.2（本机实测 **2.14.0**） | ≥ 0.17（实测 **0.29.0**） | `pip install torch torchvision` 默认即含 MPS，无需 CUDA |
| Windows（NVIDIA 独显） | CUDA | ≥ 2.2 | ≥ 0.17 | 到 [pytorch.org](https://pytorch.org/get-started/locally/) 选择与驱动匹配的 CUDA wheel |
| Linux（NVIDIA 独显） | CUDA | ≥ 2.2 | ≥ 0.17 | 同上 |
| Windows / Linux（无独显） | CPU | ≥ 2.2 | ≥ 0.17 | `pip install torch torchvision` 默认 CPU 版即可 |

> 说明：
> - `torch` 与 `torchvision` 版本必须配套，一次 `pip install torch torchvision` 会自动解析；不要分别固定不同代版本。
> - 上表中的 **实测版本** 来自本项目开发机（macOS 25.6 / Apple Silicon M5 / Python 3.11.16）；已在 Windows 11 + RTX 4060 Laptop（CUDA 13.0 wheel，torch 2.14.0+cu130）上实测通过。
> - CUDA 平台安装命令形如 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`，`cu124` 请换成与显卡驱动匹配的 CUDA 版本。

## 环境安装

本项目推荐「conda 只提供 Python，其余依赖全部走 pip」，因为 macOS arm64 上带 MPS 的 PyTorch 只在 pip 提供。

```bash
# 1. 创建环境（Python 3.11 + 全部依赖，含 torch/ultralytics/nnunetv2）
conda env create -f environment.yml
conda activate trainhub

# 2. 安装 trainhub 本身
pip install -e . --no-deps

# 3. 启动
trainhub
# 或
python -m trainhub
```

> 国内下载加速：pip 段太慢可先执行
> `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`

只想跑 YOLO、不需要 nnU-Net 时，可把 `environment.yml` 里的 `nnunetv2 / nibabel / SimpleITK / scikit-learn` 等删掉，安装体积与时间都会小很多。

## 使用教程

### 1. 启动与项目管理

```bash
trainhub                    # 新建默认项目
python -m trainhub          # 同上
python -m trainhub /path/to/proj   # 打开（或新建）指定项目目录
```

也可以在界面里通过「项目 → 新建项目… / 打开项目…」切换。项目只是一个目录，内含 `trainhub.yaml`，整目录拷贝即可迁移。

### 2. 导入数据

在「数据集」页点击 **导入图像文件…** 或 **导入图像文件夹…**：

- 普通图像：按文件/文件夹批量导入。
- 已标注的 labelme 数据：导入图像时会自动带上同名的 `.json` 标注。
- 已标注的 YOLO 数据：若所选文件夹是 YOLO 结构（含 `images/`、`labels/`、`dataset.yaml`），会自动转换回 labelme 格式再导入。
- 导入新文件夹会清空当前项目数据（有确认提示），避免新旧数据混杂。

导入后，「数据集」页会显示图像总数、已标注数、对象总数以及各类别分布。

### 3. 标注

切换到「标注」页：

- 左侧选择标签，用 **矩形框 / 多边形 / 圆形 / 点 / 线 / 折线** 等工具在图像上标注。
- 支持删除当前图像、上/下一张、保存（Ctrl+S）。
- 标注结果保存为 labelme 兼容的 JSON，存于项目的 `annotations/` 目录。

### 4. 导出数据集（labelme2yolo）

在「数据集」页点击 **labelme2yolo 导出**，会把已标注图像转换成 Ultralytics YOLO 格式，输出到项目的 `datasets/yolo_detect/`（训练集/验证集自动按比例切分）。

### 5. 训练

切换到「训练」页：

1. 选择 **训练框架**（YOLO / nnU-Net）和 **任务类型**（如目标检测）。
2. 在左侧表单调整超参数（轮数、批大小、输入尺寸、学习率等）。
3. 点击 **开始训练**，右侧可实时查看 **训练曲线 / 训练日志 / 产物**。
4. 训练结束后，权重、日志、图表都保存在 `runs/<运行名>/`，可从「历史运行」下拉框回看任意一次训练的曲线与产物。

首次使用 YOLO 时会自动下载预训练权重到 `~/.cache/trainhub/weights`。

## 项目目录结构

```
<项目根目录>/
├── trainhub.yaml        # 项目元数据（标签、选中的训练器/任务、上次参数）
├── images/              # 原始图像
├── annotations/         # labelme 格式的 JSON 标注
├── datasets/            # 转换后可直接喂给训练器的数据集
└── runs/                # 每次训练的产物（权重、日志、图表）
```

## 训练后端说明

- **YOLO（Ultralytics）**：支持 `detect / segment / classify`；计算设备支持 `auto / cuda / mps / cpu`，`auto` 按平台自动选择（优先 CUDA，其次 MPS，最后 CPU），所选设备不可用时自动回退并提示。
- **nnU-Net**：2D 医学分割。程序启动时自动设置 `nnUNet_raw / nnUNet_preprocessed / nnUNet_results` 到项目目录，不污染全局环境。

## 可扩展性

新增一个训练器：

```python
from trainhub.core.trainer import BaseTrainer, TaskSpec
from trainhub.core.registry import register_trainer

@register_trainer
class MyTrainer(BaseTrainer):
    key = "mytrainer"
    display_name = "我的训练器"
    tasks = (TaskSpec("detect", "目标检测", "bbox", "2d", "矩形框标注"),)

    def is_available(self): ...
    def param_specs(self, task_key): ...
    def prepare(self, job, sink): ...
    def train(self, job, prepared, sink): ...
```

界面上的训练框架下拉框、调参表单、实时曲线会自动适配，无需改动 UI 代码。

## 已知问题

- 计算设备 `auto` 的优先级为 CUDA → MPS → CPU；在 Windows / Linux 无独显机器上会自动落到 CPU，属预期行为。
- MPS 后端对 AMP（混合精度）支持不完整，Apple 机器上建议保持关闭（默认关闭）；NVIDIA GPU 上可开启以加速训练。
- nnU-Net 部分算子 MPS 未实现，程序已自动开启 `PYTORCH_ENABLE_MPS_FALLBACK` 回退到 CPU。

## 致谢

标注引擎移植自 [labelme](https://github.com/wkentaro/labelme) 6.3.0（GPL-3.0），标注格式与其完全兼容。
