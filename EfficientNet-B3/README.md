# EfficientNet-B3（28类：27风格 + non_art）复现工程

本目录实现论文中 EfficientNet-B3 风格分类器：输出 28 类概率向量 `z`（softmax），并可选输出低维风格嵌入 `s`（供后续 BLIP-2 的风格先验注入使用）。

## 1. 安装

建议使用独立环境（conda/venv 均可）。

```bash
conda activate lavis2
pip install -r EfficientNet-B3/requirements.txt
```

## 2. 标签定义（强制一致）

- 28 类顺序固定在：`EfficientNet-B3/labels/classes_28.json`
- 27 风格的同义词/映射规则在：`EfficientNet-B3/labels/style_27.json`

你可以按自己的数据集风格命名习惯补充同义词，但不要改变 `classes_28.json` 的 id 顺序（否则训练/推理无法对齐）。

## 3. 数据准备（生成统一 manifest）

统一输出到一个 manifests 目录，得到 `train.csv / val.csv / test.csv`。每行至少包含：

- `image_path`：图片绝对路径（推荐）或相对路径
- `label_id`：0~27
- `split`：train/val/test
- （可选）`source`：wikiart/coco/art500k，便于排查数据混合比例

### 3.0 最小复现流程（建议按此顺序）

```bash
# 0) 规则要求：必须在 conda 环境 lavis2 下运行
conda activate lavis2

# 1) 生成各自来源的清单（wikiart.csv / coco_nonart.csv / art500k.csv）
#    注意：规则要求不要向 /data/... 写 outputs，建议把 manifests 放到项目 ./outputs 下
python EfficientNet-B3/scripts/prepare_wikiart.py --wikiart_root /path/to/WikiArt --out_dir ./outputs/effb3_28cls/manifests_raw
python EfficientNet-B3/scripts/prepare_coco_nonart.py --coco_root /path/to/COCO --out_dir ./outputs/effb3_28cls/manifests_raw
python EfficientNet-B3/scripts/prepare_art500k.py --art500k_root /path/to/ART500K --out_dir ./outputs/effb3_28cls/manifests_raw

# 2) 合并为训练直接读取的 train/val/test
python EfficientNet-B3/scripts/merge_manifests.py --in_dir ./outputs/effb3_28cls/manifests_raw --out_dir ./outputs/effb3_28cls/manifests

# 3) 修改 train.yaml 里的 manifests_dir / output_dir 后开训
python EfficientNet-B3/src/train.py --config EfficientNet-B3/configs/train.yaml
```

### 3.1 WikiArt（按风格目录或元数据）

```bash
python EfficientNet-B3/scripts/prepare_wikiart.py \
  --wikiart_root /path/to/WikiArt \
  --out_dir /path/to/manifests_effb3 \
  --seed 42
```

### 3.2 COCO（作为 non_art）

```bash
python EfficientNet-B3/scripts/prepare_coco_nonart.py \
  --coco_root /path/to/COCO \
  --out_dir /path/to/manifests_effb3 \
  --seed 42
```

### 3.3 ART500K（可选：扩充风格监督）

```bash
python EfficientNet-B3/scripts/prepare_art500k.py \
  --art500k_root /path/to/ART500K \
  --out_dir /path/to/manifests_effb3 \
  --seed 42
```

### 3.4 合并为统一 train/val/test（训练入口直接读取它们）

```bash
python EfficientNet-B3/scripts/merge_manifests.py \
  --in_dir /path/to/manifests_effb3 \
  --out_dir /path/to/manifests_effb3
```

> 说明：`prepare_*.py` 会分别生成 `wikiart.csv / coco_nonart.csv / art500k.csv`；`merge_manifests.py` 会把它们合并为 `train.csv / val.csv / test.csv`，供训练脚本读取。

## 4. 训练

```bash
python EfficientNet-B3/src/train.py --config EfficientNet-B3/configs/train.yaml
```

默认启用论文设定的采样比：**艺术:非艺术 = 27:1**。

输出目录（`train.yaml` 的 `paths.output_dir`）将包含：
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics/`（acc、每类acc、混淆矩阵等）
- `logs/`（包含数据检查结果 `data_check.json` 等）

### 4.1 Smoke test（规则要求）

- **快速小跑**：优先用 `--max_steps`，其次用 `--run_ratio`。

```bash
python EfficientNet-B3/src/train.py \
  --config EfficientNet-B3/configs/train.yaml \
  --run_ratio 1 \
  --max_steps 20
```

### 4.2 长任务运行规范（screen，规则要求）

```bash
exp=effb3_28cls
datestr=$(date +%Y%m%d)
screen -S train_${exp}_${datestr} -dm bash -lc \"conda activate lavis2 && python EfficientNet-B3/src/train.py --config EfficientNet-B3/configs/train.yaml > ./outputs/${exp}/logs/train.log 2>&1\"
```

查看日志：

```bash
tail -f ./outputs/effb3_28cls/logs/train.log
```

## 5. 评测

```bash
python EfficientNet-B3/src/eval.py --config EfficientNet-B3/configs/train.yaml --split test
```

小跑评测：

```bash
python EfficientNet-B3/src/eval.py --config EfficientNet-B3/configs/train.yaml --split test --run_ratio 1 --max_steps 20
```

## 6. 推理（输出 z / s）

```bash
python EfficientNet-B3/src/infer.py \
  --checkpoint /path/to/best.pt \
  --classes_json EfficientNet-B3/labels/classes_28.json \
  --image /path/to/img.jpg
```

## 7. 导出（可选 ONNX）

```bash
python EfficientNet-B3/src/export.py \
  --checkpoint /path/to/best.pt \
  --out_dir /path/to/exported
```

## 常见问题

- 如果你的 WikiArt/ART500K 风格名称与 `style_27.json` 不一致：请在 `style_27.json` 中补充同义词（大小写/下划线/连字符都可）。
- 如果你想把 manifest 写成绝对路径：脚本默认会尽量输出绝对路径，便于跨目录运行。
- ART500K 可能含超出 27 类集合的风格：`prepare_art500k.py` 默认会跳过无法映射到 `style_27.json` 的样本（避免污染标签空间）。


