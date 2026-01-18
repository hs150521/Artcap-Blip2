# train.py

一句话用途：训练 EfficientNet-B3 28类风格分类器（支持艺术:非艺术=27:1 采样），并保存 best/last checkpoint。

## 参数

- `--config`：训练配置 YAML（必填），例如 `EfficientNet-B3/configs/train.yaml`
- `--resume`：断点续训 checkpoint（可选）
- `--run_ratio`：1-100，使用部分数据小跑（固定 seed 可复现）
- `--max_steps`：最大训练 step 数（优先级高于 run_ratio；0 表示不限）

## 示例

```bash
conda activate lavis2
python EfficientNet-B3/src/train.py --config EfficientNet-B3/configs/train.yaml
```

Smoke test：

```bash
conda activate lavis2
python EfficientNet-B3/src/train.py --config EfficientNet-B3/configs/train.yaml --run_ratio 1 --max_steps 20
```















