# EfficientNet-B3 训练脚本

## 用途

训练 EfficientNet-B3 模型用于图像分类任务，包含 28 个类别：
- 27 种 WikiArt 艺术风格
- 1 个 COCO 非艺术类别

脚本会自动平衡 WikiArt 和 COCO 数据集，保持 27:1 的比例。

## 用法

### 基本用法

```bash
python train_efficientnet_b3_28.py --epochs 100 --batch-size 32
```

### 完整参数列表

#### 数据集参数
- `--wikiart-root` (Path, 默认: `../datasets/wikiart`): WikiArt 数据集根目录
- `--coco-root` (Path, 默认: `../datasets/coco`): COCO 数据集根目录
- `--val-split` (float, 默认: `0.1`): 验证集比例（当没有显式分割时）
- `--subset-ratio` (float, 默认: `1.0`): 数据集使用比例（用于调试/快速测试）

#### 训练参数
- `--epochs` (int, 默认: `100`): 训练轮数
- `--batch-size` (int, 默认: `32`): 训练批次大小
- `--val-batch-size` (int, 默认: `None`): 验证批次大小（默认与训练批次大小相同）
- `--num-workers` (int, 默认: `8`): DataLoader 工作进程数
- `--seed` (int, 默认: `1337`): 随机种子
- `--image-size` (int, 默认: `300`): 输入图像尺寸

#### 优化器参数
- `--lr` (float, 默认: `None`): 学习率（可选，未设置时根据基础学习率自动计算）
- `--base-lr` (float, 默认: `0.256`): 基础学习率（按 `batch_size/256` 线性缩放）
- `--no-lr-scale` (flag): 禁用学习率随批次大小的线性缩放
- `--rmsprop-alpha` (float, 默认: `0.9`): RMSprop 的 alpha（decay）参数
- `--rmsprop-momentum` (float, 默认: `0.9`): RMSprop 动量
- `--rmsprop-eps` (float, 默认: `1e-3`): RMSprop epsilon
- `--weight-decay` (float, 默认: `1e-5`): 权重衰减系数

#### 学习率调度器参数
- `--lr-scheduler` (str, 默认: `cosine`): 学习率调度器类型，可选 `cosine`、`plateau` 或 `none`
- `--warmup-epochs` (int, 默认: `5`): Warmup 轮数（余弦退火前）
- `--warmup-lr-init` (float, 默认: `0.01`): Warmup 初始学习率比例
- `--min-lr` (float, 默认: `1e-5`): 余弦退火的最小学习率

#### 模型参数
- `--dropout` (float, 默认: `0.3`): 分类器 Dropout 概率
- `--freeze-backbone` (flag): 冻结特征提取器（用于线性探测）

#### 损失函数参数
- `--label-smoothing` (float, 默认: `0.1`): 标签平滑系数

#### 数据增强参数
- `--mixup-alpha` (float, 默认: `0.2`): MixUp 增强参数（设为 >0 启用）
- `--cutmix-alpha` (float, 默认: `1.0`): CutMix 增强参数（设为 >0 启用）
- `--use-randaugment` (flag): 启用 RandAugment 增强
- `--randaugment-num-ops` (int, 默认: `2`): RandAugment 操作数量
- `--randaugment-magnitude` (int, 默认: `9`): RandAugment 操作强度
- `--use-autoaugment` (flag): 启用 AutoAugment（ImageNet 策略，与 RandAugment 互斥）
- `--disable-autoaugment` (flag): 关闭默认 AutoAugment 策略
- 默认：未显式选择 RandAugment/AutoAugment 且未使用 `--disable-autoaugment` 时会自动启用 AutoAugment
- `--random-erasing` (flag): 启用 RandomErasing
- `--random-erasing-p` (float, 默认: `0.1`): RandomErasing 概率

#### 训练技巧参数
- `--grad-clip-norm` (float, 默认: `0.0`): 梯度裁剪阈值（<=0 禁用）
- `--ema-decay` (float, 默认: `0.9999`): EMA 衰减因子（<=0 禁用 EMA）
- `--ema-warmup-steps` (int, 默认: `100`): EMA 完全生效前的预热步数
- `--early-stop-patience` (int, 默认: `0`): 早停耐心值（基于验证准确率，0 禁用）
- `--no-amp` (flag): 禁用混合精度训练（默认启用 AMP）

#### 输出和日志参数
- `--output-dir` (Path, 默认: `None`): 输出目录（未指定时使用 `runs/YYYY_MM_DD_HH_MM_SS/`）
- `--save-every` (int, 默认: `5`): 每 N 轮保存一次检查点（除最佳模型外）
- `--log-tensorboard` (flag): 启用 TensorBoard 日志记录
- `--compile` (flag): 启用 `torch.compile` 加速（如果可用）
- `--resume` (Path, 默认: `None`): 从检查点恢复训练

### 输出文件

训练完成后，输出目录包含：
- `best.pt`: 最佳模型检查点
- `checkpoint_epoch_N.pt`: 定期保存的检查点
- `training_log.jsonl`: 训练日志
- `summary.json`: 训练摘要

