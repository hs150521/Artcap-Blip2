# 数据准备脚本（scripts）

一句话用途：把 WikiArt/COCO/ART500K 整理为统一的 `csv` manifest，并合并为训练直接读取的 `train/val/test.csv`。

## prepare_wikiart.py

参数（核心）：`--wikiart_root`、`--out_dir`、可选 `--meta_csv`、`--seed`

示例：

```bash
conda activate lavis2
python EfficientNet-B3/scripts/prepare_wikiart.py --wikiart_root /data/artcap-blip2-4/datasets/wikiart --out_dir ./outputs/effb3_28cls/manifests_raw
```

## prepare_coco_nonart.py

参数（核心）：`--coco_root`、`--out_dir`、`--subdirs`

示例：

```bash
conda activate lavis2
python EfficientNet-B3/scripts/prepare_coco_nonart.py --coco_root /data/artcap-blip2-4/datasets/coco/images --out_dir ./outputs/effb3_28cls/manifests_raw --subdirs train2014,val2014
```

## prepare_art500k.py

参数（核心）：`--art500k_root`、`--out_dir`、可选 `--meta_csv`

示例：

```bash
conda activate lavis2
python EfficientNet-B3/scripts/prepare_art500k.py --art500k_root /path/to/ART500K --out_dir ./outputs/effb3_28cls/manifests_raw
```

## merge_manifests.py

一句话用途：把 `*_raw/*.csv` 合并输出为 `train.csv/val.csv/test.csv`。

示例：

```bash
conda activate lavis2
python EfficientNet-B3/scripts/merge_manifests.py --in_dir ./outputs/effb3_28cls/manifests_raw --out_dir ./outputs/effb3_28cls/manifests
```
















