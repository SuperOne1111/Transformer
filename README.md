#本文档用简单的数学描述和图示解释 Transformer 的 Encoder 和 Decoder 模块，适合初学者快速理解核心机制。

## 目录
- [Encoder 阶段](./Transformer_Encoder.md) 
  - [自注意力机制]  
  - [前馈网络] 
- [Decoder 阶段](./Transformer_Decoder.md) 
  - [掩码自注意力] 
  - [交叉注意力]
---
# 简要说明
**[Encoder阶段](./Transformer_Encoder.md)**
Encoder 将输入序列转换为隐藏表示，核心是**自注意力机制**和**位置感知前馈网络**。


**[Decoder阶段](./Transformer_Decoder.md)**
Decoder 通过**掩码自注意力**和**交叉注意力**逐步生成输出序列。
