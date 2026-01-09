# eval.py

一句话用途：加载训练好的 checkpoint，对指定 split（train/val/test）做分类评测并输出指标与混淆矩阵。

## 参数

- `--config`：配置 YAML（必填）
- `--checkpoint`：checkpoint 路径（可选；默认 `<output_dir>/checkpoints/best.pt`）
- `--split`：`train|val|test`（默认 test）
- `--run_ratio`：1-100，小跑评测
- `--max_steps`：最大评测 batch 数（0 不限）

## 示例

```bash
conda activate lavis2
python EfficientNet-B3/src/eval.py --config EfficientNet-B3/configs/train.yaml --split test
```

Smoke test：

```bash
conda activate lavis2
python EfficientNet-B3/src/eval.py --config EfficientNet-B3/configs/train.yaml --split test --run_ratio 1 --max_steps 20
```










