基于高斯径向基函数（Radial Basis Function, RBF）和 Softmax 分类器实现的多分类实验项目。项目提供统一的数据加载、预处理、训练和评估流程，适合用于机器学习课程实验、传统神经网络方法对比，以及 RBF 网络原理学习。

Features

- 使用 KMeans 选取 RBF 中心点
- 支持自动估计 `gamma` 参数
- 使用 Softmax 输出层完成多分类
- 使用 Adam 优化器训练输出层权重
- 支持标准化、PCA 降维和数据集级别配置
- 提供统一的训练与评估入口
- 支持多个常用分类数据集

Supported Datasets

| Dataset | Source | Notes |
| --- | --- | --- |
| `iris` | scikit-learn | 鸢尾花分类 |
| `wine` | scikit-learn | 葡萄酒分类 |
| `breast_cancer` | scikit-learn | 乳腺癌二分类 |
| `digits` | scikit-learn | 手写数字分类 |
| `pendigits` | UCI / OpenML | 笔迹数字分类，会优先读取本地文件 |
| `fashion_mnist` | torchvision | Fashion-MNIST 图像分类，默认使用 PCA 降维 |

Project Structure

text
RBF_Network/
├── src/
│   ├── rbf_network.py          # RBF 网络核心实现
│   ├── datasets.py             # 数据集加载与划分
│   ├── train_evaluate.py       # CPU 训练与评估入口
│   ├── train_evaluate_gpu.py   # GPU 入口占位，当前核心训练仍基于 CPU
│   └── download_datasets.py    # 下载或缓存数据集
├── config.yaml                 # 数据集与模型超参数配置
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
├── README_cn.md                # 中文简短说明
└── LICENSE                     # MIT License


Requirements

建议使用 Python 3.9 或更高版本。

主要依赖：

- NumPy
- SciPy
- scikit-learn
- PyYAML
- PyTorch
- torchvision

Installation

克隆项目后进入项目目录：

bash
git clone <your-repo-url>
cd RBF_Network


创建并激活虚拟环境：

bash
python -m venv .venv
source .venv/bin/activate


Windows 用户可使用：

bash
.venv\Scripts\activate


安装依赖：

bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


Quick Start

训练并评估 Iris 数据集：

bash
python src/train_evaluate.py --dataset iris


训练其他数据集：

bash
python src/train_evaluate.py --dataset wine
python src/train_evaluate.py --dataset breast_cancer
python src/train_evaluate.py --dataset digits
python src/train_evaluate.py --dataset pendigits
python src/train_evaluate.py --dataset fashion_mnist


指定自定义配置文件：

bash
python src/train_evaluate.py --config config.yaml --dataset digits


Dataset Preparation

`iris`、`wine`、`breast_cancer` 和 `digits` 由 scikit-learn 内置加载，通常不需要额外下载。

`pendigits` 和 `fashion_mnist` 可能需要联网下载。可以提前缓存所有数据集：

bash
python src/download_datasets.py --dataset all


也可以只准备单个数据集：

bash
python src/download_datasets.py --dataset pendigits
python src/download_datasets.py --dataset fashion_mnist


默认数据目录为 `data/`。如果需要自定义数据目录：

bash
python src/download_datasets.py --dataset all --data-dir data


Configuration

所有数据集相关参数集中在 `config.yaml` 中：

  yaml
project:
  data_dir: data
  random_state: 42

datasets:
  iris:
    standardize: true
    model:
      n_centers: 20
      gamma: 0.1
      lr: 0.01
      epochs: 500
      reg: 0.0001
      print_every: 20


常用参数说明：

| Parameter | Description |
| --- | --- |
| `data_dir` | 数据缓存目录 |
| `random_state` | 随机种子 |
| `standardize` | 是否使用 `StandardScaler` 标准化 |
| `pca_components` | PCA 降维后的维度，未设置时不启用 PCA |
| `train_size` | 训练集采样数量，主要用于 Fashion-MNIST |
| `test_size` | 测试集比例，默认 `0.2` |
| `n_centers` | RBF 中心点数量 |
| `gamma` | RBF 核宽度参数；设为 `null` 时自动估计 |
| `lr` | 学习率 |
| `epochs` | 训练轮数 |
| `reg` | L2 正则化系数 |
| `print_every` | 训练日志输出间隔 |

Output

训练脚本会输出：

- 数据集来源
- 训练集与测试集形状
- 每隔若干 epoch 的 loss 和训练准确率
- 训练集准确率
- 测试集准确率
- 自动估计的 `gamma`
- 分类报告
- 混淆矩阵

示例：

text
Dataset: iris
Source: scikit-learn iris
Train shape: (120, 4)
Test shape: (30, 4)

====================
TRAIN ACC: ...
TEST ACC: ...
AUTO GAMMA: ...
====================


Model Overview

模型核心位于 `src/rbf_network.py`，主要流程如下：

1. 使用 KMeans 从训练样本中得到 RBF 中心点。
2. 如果 `gamma` 为 `null`，根据中心点之间的距离自动估计核宽度。
3. 将输入样本映射到 RBF 特征空间。
4. 使用 Softmax 层输出类别概率。
5. 使用交叉熵损失、L2 正则化和 Adam 优化器训练分类层。

Notes

- `train_evaluate_gpu.py` 当前是兼容性入口，模型核心仍依赖 NumPy、SciPy 和 scikit-learn，实际训练在 CPU 上执行。
- Fashion-MNIST 数据量较大，默认只抽取部分训练样本并启用 PCA，以降低运行成本。
- `pendigits` 会优先尝试读取本地 `pendigits.tra` 和 `pendigits.tes` 文件；如果不存在，则尝试通过 OpenML 获取。

License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
