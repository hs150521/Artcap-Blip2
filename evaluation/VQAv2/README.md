# VQAv2 评测脚本

该模块提供一个用于在 VQAv2 数据集上评测 BLIP2 模型的**模块化** Python 脚本。它用更灵活的、基于函数的方式取代了原先基于 shell 的评测流程，支持数据采样与便捷的模型切换。

## 函数

1. **`load_config(config_path)`**：加载 YAML 配置文件，并加入对 `data_percentage` 的支持
2. **`load_vqav2_data(config, data_percentage)`**：加载 VQAv2 数据集，必要时按比例采样
3. **`run_model_inference(model, data_samples, config, ...)`**：对样本运行模型推理
4. **`evaluate_results(predictions, dataset_info, output_dir)`**：评估预测结果并计算指标

## 用法

### 基本用法

```bash
python VQAv2/evaluate_vqav2.py --cfg-path VQAv2/config_example.yaml
```

### 启用数据采样

仅使用 50% 的数据集：

```bash
python VQAv2/evaluate_vqav2.py --cfg-path VQAv2/config_example.yaml --data-percentage 0.5
```

或在配置文件中设置：

```yaml
run:
  data_percentage: 0.5  # 使用 50% 的数据
```

### 命令行选项

* `--cfg-path`：配置 YAML 文件路径（必填）
* `--data-percentage`：使用数据的比例（0–1），如提供则覆盖配置
* `--batch-size`：推理批大小，如提供则覆盖配置
* `--device`：使用的设备（cuda/cpu），如提供则覆盖配置

## 配置文件

完整示例见 `config_example.yaml`。

## 输出

脚本会生成：

* `vqa_predictions.json`：符合 VQA 格式的预测结果
* `evaluate.txt`：评测指标（总体准确率与按答案类型的准确率）

输出目录由配置文件中的 `run.output_dir` 指定。

## 依赖

TODO: conda activate lavis2
