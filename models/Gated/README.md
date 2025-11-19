# BLIP2-OPT + EfficientNet 门控融合

该目录提供了一个基于 **BLIP2-OPT-2.7B** 与 **EfficientNet-B3（WikiArt 预训练）** 的门控/FiLM 融合实现，通过在 Q-Former 跨注意力的 K/V 上施加门控参数，实现视觉风格调制与文本生成协同训练。

## 目录结构

```
models/Gated/
├── configs/            # YAML 配置，支持 default 与 artquest 预设
├── datasets/           # ArtQuest 数据加载器
├── modules/            # EfficientNet 适配、PromptMapper、Q-Former 门控等模块
├── trainers/           # 端到端训练控制器
├── scripts/            # CLI 入口（train/eval）
├── utils/              # Dataloader、checkpoint、风格指标等工具
└── checkpoints/        # 默认仅保存 best.pt
```

## 训练流程概览

1. **冻结组件**：视觉编码器（EVA CLIP）、OPT-2.7B 语言模型与 EfficientNet 主干，避免巨型参数更新。
2. **特征抽取**：使用 EfficientNet-B3 处理 BLIP2 归一化后的图像，得到 1536 维的 pooled 特征，并线性映射到 Q-Former 隐层维度。
3. **PromptMapper**：将 EfficientNet 特征与文本提示联合映射成控制 token，作为门控模块输入。
4. **门控/FiLM**：在 Q-Former 跨注意力的 K/V 上应用 γ(t)、β(t) 或门控因子 g(t)，支持逐头或逐通道控制。
5. **主任务 + 正则**：
   - 主损失：与原始 BLIP2 相同的语言建模损失。
   - 风格命中率：统计预测 token 中命中风格词表的比例。
   - 风格一致性 KL：EfficientNet 风格 logits 与文本风格 logits 的对称 KL。
6. **Checkpoint**：仅保存指标最优的 `best.pt`。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt  # 若已有 LAVIS/transformers 可跳过

# 训练（默认读取 configs/artquest.yaml）
python models/Gated/scripts/train_gated.py --config artquest

# 验证或推理
python models/Gated/scripts/eval_gated.py --config artquest --checkpoint models/Gated/checkpoints/best.pt
```

### 配置说明

主要参数集中在 `configs/default.yaml`：

- `data.*`：数据根目录、CSV 分割、图像根路径、字段名等。
- `model.*`
  - `visual_encoder`：BLIP2 视觉骨干（默认 eva_clip_g）。
  - `opt_model`：文本生成 LLM。
  - `efficientnet_output_dim`：EfficientNet pooled 特征映射维度（需与 Q-Former hidden size 对齐）。
  - `gating`：门控类型（film/gate）、逐头控制、初始缩放等。
  - `prompt_mapper`：控制 token 数量、隐藏层与 dropout。
  - `lora`：对 Q-Former 跨注意力 W_K/W_V 的 LoRA 设定（可选）。
- `training.*`：epoch、梯度累积、学习率调度、AMP 等。
- `loss.*`：主损失权重、风格命中与一致性 KL 权重。

> 如需切换其他 `/data/` 下的数据集，只需调整 `data.dataset_name`、CSV/图像路径及对应字段。

## 正则项细节

- **风格词表**：由训练集 `style` 字段收集所有风格标签，经 OPT tokenizer 映射成 token 集。
- **风格命中率**：`1 - hit_rate` 被加权到总损失，鼓励模型输出风格相关词。
- **一致性 KL**：EfficientNet pooled 特征通过线性头得到图像风格分布；文本端使用预测 logits 在风格 token 上的均值，二者通过对称 KL 约束。

## 仅保存 best.pt

`trainers/GatedTrainer` 调用 `utils.save_best_checkpoint`，每个 epoch 根据验证集 `-loss` 更新最优指标；若提升即覆盖写入 `checkpoint_path`。无需额外清理旧权重，磁盘压力极小。

## 评估与导出

`scripts/eval_gated.py` 会加载 `best.pt`，基于验证/测试集输出：

- 主损失 / 风格命中率 / KL 正则
- 可选：序列化预测结果（用于下游 VQA/VQG 评测）

如需对接 `evaluation/VQAv2` 等外部流程，可直接调用 `Blip2OPTGated.predict_answers`。

## 后续拓展

- 接入更多 `/data/` 下数据集：实现自定义 `Dataset` 并在 `utils/data.py` 中注册即可。
- LoRA 微调：在配置中启用 `model.lora.enabled` 并调整 rank/alpha。
- 正则增强：可在 `loss` 模块中增添更多风格或语义约束（例如对齐独立风格分类器）。

如有问题可参考 `models/KV` 目录中的 KV modulation 实现，两者结构保持一致，便于迁移与对比实验。



