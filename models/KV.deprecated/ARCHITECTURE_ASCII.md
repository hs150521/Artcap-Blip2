# BLIP2 KV Modulation 模型架构图

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BLIP2-OPT-KV-Modulated Model                        │
└─────────────────────────────────────────────────────────────────────────────┘

                                    Input Image
                                    (B, 3, 224, 224)
                                         │
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                        │
                    ▼                                        ▼
         ┌──────────────────────┐              ┌──────────────────────┐
         │   Vision Encoder     │              │    EfficientNet-B3   │
         │   (EVA-CLIP-G)      │              │     (Frozen)         │
         │   [Frozen]          │              │     [Frozen]         │
         └──────────────────────┘              └──────────────────────┘
                    │                                        │
                    │                                        │
                    ▼                                        ▼
         ┌──────────────────────┐              ┌──────────────────────┐
         │  Visual Features     │              │ EfficientNet Features│
         │  (B, 257, 1408)      │              │   (B, 1536)          │
         │  [257 = 256 patches  │              │   [Global Pooled]    │
         │   + 1 CLS token]     │              │                       │
         └──────────────────────┘              └──────────────────────┘
                    │                                        │
                    │                                        │
                    │                                        ▼
                    │                          ┌──────────────────────────┐
                    │                          │  KV-Prefix Generator    │
                    │                          │  (Trainable)            │
                    │                          │                          │
                    │                          │  ┌────────────────────┐ │
                    │                          │  │ Feature Projection │ │
                    │                          │  │ 1536 → 768        │ │
                    │                          │  │ MLP: Linear+GELU   │ │
                    │                          │  └────────────────────┘ │
                    │                          │           │             │
                    │                          │           ▼             │
                    │                          │  ┌────────────────────┐ │
                    │                          │  │ Prefix Tokens      │ │
                    │                          │  │ (Learnable)        │ │
                    │                          │  │ (B, 8, 768)        │ │
                    │                          │  └────────────────────┘ │
                    │                          │           │             │
                    │                          │           ▼             │
                    │                          │  ┌────────────────────┐ │
                    │                          │  │ K & V Projection   │ │
                    │                          │  │ K: (B,12,8,64)     │ │
                    │                          │  │ V: (B,12,8,64)     │ │
                    │                          │  └────────────────────┘ │
                    │                          └──────────────────────────┘
                    │                                        │
                    │                                        │
                    │                    ┌───────────────────┴───────────────────┐
                    │                    │                                       │
                    │                    ▼                                       ▼
                    │         ┌──────────────────┐                  ┌──────────────────┐
                    │         │   K Prefix       │                  │   V Prefix       │
                    │         │ (B, 12, 8, 64)   │                  │ (B, 12, 8, 64)   │
                    │         └──────────────────┘                  └──────────────────┘
                    │                    │                                       │
                    │                    └───────────┬───────────────────────────┘
                    │                                │
                    │                                │ Injected into
                    │                                │ Cross-Attention
                    │                                │
                    ▼                                ▼
         ┌─────────────────────────────────────────────────────────────┐
         │                    Qformer (KV-Modulated)                  │
         │                    (Trainable)                             │
         │                                                             │
         │  ┌──────────────────────────────────────────────────────┐  │
         │  │              Query Tokens                            │  │
         │  │         (B, 32, 768) [Learnable]                    │  │
         │  └──────────────────────────────────────────────────────┘  │
         │                      │                                      │
         │                      ▼                                      │
         │  ┌──────────────────────────────────────────────────────┐  │
         │  │         BertEncoderKVModulated                        │  │
         │  │                                                       │  │
         │  │  ┌──────────────────────────────────────────────┐    │  │
         │  │  │  Layer 0: Self-Attention                     │    │  │
         │  │  └──────────────────────────────────────────────┘    │  │
         │  │  ┌──────────────────────────────────────────────┐    │  │
         │  │  │  Layer 1: Self-Attention                     │    │  │
         │  │  └──────────────────────────────────────────────┘    │  │
         │  │  ┌──────────────────────────────────────────────┐    │  │
         │  │  │  Layer 2: Cross-Attention ⭐                │    │  │
         │  │  │  ┌──────────────────────────────────────┐   │    │  │
         │  │  │  │  Q = Query Tokens                    │   │    │  │
         │  │  │  │  K = [K_Prefix || Visual Features]   │   │    │  │
         │  │  │  │  V = [V_Prefix || Visual Features]   │   │    │  │
         │  │  │  │                                      │   │    │  │
         │  │  │  │  Attention(Q, K, V)                  │   │    │  │
         │  │  │  └──────────────────────────────────────┘   │    │  │
         │  │  └──────────────────────────────────────────────┘    │  │
         │  │  ┌──────────────────────────────────────────────┐    │  │
         │  │  │  Layer 3: Self-Attention                     │    │  │
         │  │  └──────────────────────────────────────────────┘    │  │
         │  │  ┌──────────────────────────────────────────────┐    │  │
         │  │  │  Layer 4: Cross-Attention ⭐                │    │  │
         │  │  │  (Same KV prefix injection)                 │    │  │
         │  │  └──────────────────────────────────────────────┘    │  │
         │  │  ... (12 layers total, cross-attention every 2)     │  │
         │  └──────────────────────────────────────────────────────┘  │
         │                                                             │
         │  Output: Query Features (B, 32, 768)                       │
         └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   OPT Projection Layer        │
                    │   768 → 2560 (OPT hidden)     │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      OPT-2.7B Model           │
                    │      (Frozen)                 │
                    │                               │
                    │  ┌─────────────────────────┐ │
                    │  │  Text Input Tokens       │ │
                    │  │  (Question)             │ │
                    │  └─────────────────────────┘ │
                    │            │                  │
                    │            ▼                  │
                    │  ┌─────────────────────────┐ │
                    │  │  Concatenate:            │ │
                    │  │  [Query Features ||     │ │
                    │  │   Text Tokens]          │ │
                    │  └─────────────────────────┘ │
                    │            │                  │
                    │            ▼                  │
                    │  ┌─────────────────────────┐ │
                    │  │  OPT Decoder Layers     │ │
                    │  │  (32 layers)            │ │
                    │  └─────────────────────────┘ │
                    │            │                  │
                    │            ▼                  │
                    │  ┌─────────────────────────┐ │
                    │  │  Language Model Head    │ │
                    │  └─────────────────────────┘ │
                    └───────────────────────────────┘
                                    │
                                    ▼
                            Generated Answer
                            (Text Output)
```

## 关键组件详解

### 1. 双视觉编码器架构

```
Input Image
    │
    ├───→ EVA-CLIP-G (Vision Encoder)
    │         │
    │         └───→ Visual Features (257, 1408)
    │                [256 image patches + 1 CLS token]
    │
    └───→ EfficientNet-B3
              │
              └───→ EfficientNet Features (1536)
                     [Global pooled features]
```

### 2. KV-Prefix Generator 详细结构

```
EfficientNet Features (B, 1536)
    │
    ▼
┌─────────────────────────────────────┐
│  Feature Projection                 │
│  ┌───────────────────────────────┐  │
│  │ Linear(1536 → 1536)           │  │
│  │ GELU()                        │  │
│  │ Linear(1536 → 768)            │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Add to Learnable Prefix Tokens    │
│  ┌───────────────────────────────┐  │
│  │ Prefix Tokens (B, 8, 768)     │  │
│  │ [Learnable Parameter]         │  │
│  │                               │  │
│  │ + Projected Features          │  │
│  │   (Broadcast to 8 tokens)    │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ K Proj   │    │          │    │ V Proj   │
│ Linear   │    │          │    │ Linear   │
│(768→768) │    │          │    │(768→768) │
└──────────┘    │          │    └──────────┘
    │           │          │         │
    ▼           │          │         ▼
┌──────────┐    │          │    ┌──────────┐
│ Reshape  │    │          │    │ Reshape  │
│ to heads │    │          │    │ to heads │
│(B,8,12,64)│   │          │    │(B,8,12,64)│
└──────────┘    │          │    └──────────┘
    │           │          │         │
    ▼           │          │         ▼
┌──────────┐    │          │    ┌──────────┐
│ Transpose│    │          │    │ Transpose│
│(B,12,8,64)│   │          │    │(B,12,8,64)│
└──────────┘    │          │    └──────────┘
    │           │          │         │
    └───────────┴──────────┴─────────┘
                │
                ▼
    K Prefix: (B, 12, 8, 64)
    V Prefix: (B, 12, 8, 64)
```

### 3. Qformer Cross-Attention with KV Modulation

```
┌─────────────────────────────────────────────────────────────┐
│         Cross-Attention Layer (KV-Modulated)                │
│                                                             │
│  Query: Query Tokens (B, 32, 768)                          │
│    │                                                        │
│    ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Q = Linear(Query Tokens)                            │  │
│  │  Shape: (B, 32, 768) → (B, 12, 32, 64)             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Key: Visual Features + K Prefix                           │
│    │                                                        │
│    ├───→ Visual Features (B, 257, 1408)                    │
│    │         │                                              │
│    │         ▼                                              │
│    │    ┌────────────────────────────────────────────┐     │
│    │    │  Linear Projection                         │     │
│    │    │  (1408 → 768)                              │     │
│    │    └────────────────────────────────────────────┘     │
│    │         │                                              │
│    │         ▼                                              │
│    │    Visual K: (B, 12, 257, 64)                         │
│    │                                                        │
│    └───→ K Prefix: (B, 12, 8, 64)                          │
│              │                                              │
│              ▼                                              │
│    ┌────────────────────────────────────────────┐          │
│    │  Concatenate along sequence dimension      │          │
│    │  K = [K_Prefix || Visual K]                │          │
│    │  Shape: (B, 12, 8+257, 64) = (B,12,265,64)│          │
│    └────────────────────────────────────────────┘          │
│                                                             │
│  Value: Visual Features + V Prefix                         │
│    │                                                        │
│    ├───→ Visual Features (B, 257, 1408)                    │
│    │         │                                              │
│    │         ▼                                              │
│    │    ┌────────────────────────────────────────────┐     │
│    │    │  Linear Projection                         │     │
│    │    │  (1408 → 768)                              │     │
│    │    └────────────────────────────────────────────┘     │
│    │         │                                              │
│    │         ▼                                              │
│    │    Visual V: (B, 12, 257, 64)                         │
│    │                                                        │
│    └───→ V Prefix: (B, 12, 8, 64)                          │
│              │                                              │
│              ▼                                              │
│    ┌────────────────────────────────────────────┐          │
│    │  Concatenate along sequence dimension      │          │
│    │  V = [V_Prefix || Visual V]                │          │
│    │  Shape: (B, 12, 8+257, 64) = (B,12,265,64)│          │
│    └────────────────────────────────────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Attention Computation                               │  │
│  │  Attention(Q, K, V) = softmax(QK^T / √d_k) V        │  │
│  │                                                      │  │
│  │  Q: (B, 12, 32, 64)                                 │  │
│  │  K: (B, 12, 265, 64)  [8 prefix + 257 visual]     │  │
│  │  V: (B, 12, 265, 64)  [8 prefix + 257 visual]     │  │
│  │                                                      │  │
│  │  Output: (B, 12, 32, 64)                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Output: (B, 32, 768)                                      │
└─────────────────────────────────────────────────────────────┘
```

## 数据流总结

```
1. 图像输入 (B, 3, 224, 224)
   │
   ├─→ EVA-CLIP-G → Visual Features (B, 257, 1408)
   │
   └─→ EfficientNet-B3 → EfficientNet Features (B, 1536)
       │
       └─→ KV-Prefix Generator
           │
           ├─→ K Prefix (B, 12, 8, 64)
           └─→ V Prefix (B, 12, 8, 64)

2. Query Tokens (B, 32, 768) [Learnable]
   │
   └─→ Qformer Cross-Attention
       │
       ├─→ Q: Query Tokens (B, 12, 32, 64)
       ├─→ K: [K_Prefix (8) || Visual K (257)] (B, 12, 265, 64)
       └─→ V: [V_Prefix (8) || Visual V (257)] (B, 12, 265, 64)
       │
       └─→ Query Output (B, 32, 768)

3. Query Output (B, 32, 768)
   │
   └─→ OPT Projection (768 → 2560)
       │
       └─→ OPT-2.7B Decoder
           │
           ├─→ Input: [Query Features || Text Tokens]
           └─→ Output: Generated Answer
```

## 关键创新点

1. **双视觉编码器融合**: 
   - EVA-CLIP-G 提供主要视觉特征
   - EfficientNet-B3 提供补充特征用于KV调制

2. **KV前缀注入机制**:
   - EfficientNet特征 → KV-Prefix Generator → K/V前缀
   - 在Qformer的cross-attention层中，将KV前缀与视觉特征的KV拼接
   - 使模型能够同时关注两种视觉编码器的信息

3. **参数效率**:
   - Vision Encoder (EVA-CLIP-G): Frozen
   - EfficientNet: Frozen
   - OPT-2.7B: Frozen
   - 仅训练: KV-Prefix Generator + Qformer + OPT Projection

4. **注意力机制增强**:
   - Query tokens可以同时关注:
     - EfficientNet的补充信息 (通过KV前缀)
     - EVA-CLIP的视觉特征 (通过标准KV)
   - 实现了多层次的视觉信息融合

## 模型参数统计

```
Frozen Components:
  - EVA-CLIP-G Vision Encoder: ~1B parameters
  - EfficientNet-B3: ~12M parameters  
  - OPT-2.7B: ~2.7B parameters

Trainable Components:
  - KV-Prefix Generator: ~2M parameters
    - Feature Projection: 1536→1536→768
    - K/V Projection: 768→768
    - Prefix Tokens: (8, 768)
  - Qformer: ~188M parameters
    - 12 layers with cross-attention every 2 layers
  - OPT Projection: 768→2560: ~2M parameters

Total Trainable: ~192M parameters
Total Model: ~3.9B parameters
```

