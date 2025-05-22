# Transformer Decoder 机制详解

## 输入处理
### 目标序列预处理
- **训练阶段**：  
  给定目标序列 `["<START>", "I", "love", "NLP", "<END>"]`，通过右移操作添加起始符：  
  输入序列：`["<START>", "I", "love", "NLP"]`  
  目标序列：`["I", "love", "NLP", "<END>"]`
- **推理阶段**：  
初始输入为 `["<START>"]`，逐步预测直至遇到 `<END>` 或达到最大长度。

### 嵌入与位置编码
1. 通过查表将输入序列转换为嵌入向量：
$`
 \text{Embedding}(x_i) \in \mathbb{R}^{d_{\text{model}}}
 `$
2. 添加位置编码（以正弦函数为例）：
 $`P(pos, 2i)`$ &= $`\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)`$, $`P(pos, 2i+1) `$&= $`\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)`$ 

 
3. 最终输入矩阵：
 $`
 T = \text{Embedding}(X) + P \in \mathbb{R}^{n \times d_{\text{model}}}
 `$

## Decoder 自注意力机制
### 权重矩阵初始化
- 专属权重矩阵：
$`
W_Q^d, W_K^d, W_V^d \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
`$
- 计算 Q/K/V 矩阵：
$`
Q = T W_Q^d, \quad K = T W_K^d, \quad V = T W_V^d
`$

### 多头注意力拆分
将 Q/K/V 拆分为 $`h`$ 个头（$`h`$ 为头数）：
  $`
\text{head}_i = \text{Attention}(Q_i, K_i, V_i), \quad Q_i \in \mathbb{R}^{n \times (d_{\text{model}}/h)}
`$

# Masked Multi-Head Self-Attention 机制详解

## 1. 掩码机制（Masking）

### 核心目的
- **防止信息泄露**  
  确保模型在预测时仅依赖合法信息（如语言模型只能根据上文预测下文）。若不掩码，模型可能直接"作弊"利用未来信息，导致训练失效。
- **支持自回归生成**  
  在推理时逐步生成序列（如逐词生成文本），掩码保证生成过程的因果性。
- **处理变长序列**  
  结合填充掩码（Padding Mask），可忽略无效的填充位置（如短句末尾的`<pad>`符号）。

### 数学实现
生成下三角矩阵 $M \in \mathbb{R}^{n \times n}$：
$`
M_{i,j} = \begin{cases} 
0 & \text{if } j \leq i \quad \text{(允许访问当前位置及历史)} \\ 
-\infty & \text{if } j > i \quad \text{(屏蔽未来位置)} 
\end{cases}
`$

### 注意力计算
$`
\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + M \right) V
`$
其中 $d_k = \frac{d_{\text{model}}}{h}$

---

## 2. 多头注意力机制

### 多头拆分
将Q/K/V通过线性投影拆分为$`h`$个头：
$$\text{Head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
- 投影矩阵维度：  
  $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$  
  $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$  
  （其中 $d_k = d_v = \frac{d_{\text{model}}}{h}$）

### 并行计算
每个头独立计算掩码注意力，输出形状为 `[batch_size, seq_len, d_v]`。

### 2.1 多头合并机制

#### 拼接与投影
1. **拼接（Concatenation）**：\
   $$MultiHead(Q,K,V) = Concat(Head_1, \ldots, Head_h) W^O$$
   - 投影权重矩阵 $W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$  
   - 输出维度恢复为 `[batch_size, seq_len, d_model]`

2. **维度变化示例**：
   | 阶段                | 张量形状（示例）       | 说明                          |
   |---------------------|-----------------------|-------------------------------|
   | 单头输出            | `[1, 3, 64]`          | 假设`batch=1`, `seq_len=3`     |
   | 多头拼接（8个头）   | `[1, 3, 512]`         | $8 \times 64 = 512$           |
   | 最终输出            | `[1, 3, 512]`         | 与输入维度一致                |

---

### 残差连接与层归一化
```math 
T_{new} = LayerNorm( Attention(Q, K, V) + T )
```

*多头情形* 
$`T_{new} = LayerNorm( MultiHead(Q, K, V) + T ) `$ 

## Encoder-Decoder 交叉注意力
对于Decoder-Only模型无此步骤。
### 权重矩阵与计算
- 专属权重矩阵：
$`
W_Q^{\text{cross}}, W_K^{\text{cross}}, W_V^{\text{cross}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
`$
- 计算 Q/K/V：
$`
Q_{\text{cross}} = T W_Q^{\text{cross}}, \quad K_{\text{cross}} = Z W_K^{\text{cross}}, \quad V_{\text{cross}} = Z W_V^{\text{cross}}
`$
其中 $Z \in \mathbb{R}^{m \times d_{\text{model}}}$ 为 Encoder 输出

### 注意力计算与归一化
$`
\begin{aligned}
\text{CrossAttention} &= \text{Softmax}\left( \frac{Q_{\text{cross}}K_{\text{cross}}^\top}{\sqrt{d_k}} \right) V_{\text{cross}} \\
T_{\text{new}} &= \text{LayerNorm}( \text{CrossAttention} + T )
\end{aligned}
`$

## 前馈神经网络（FFN）
### 网络结构
```math
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2 \\
W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}, \quad W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}
```
其中 $d_{\text{ff}}$ 通常为 $4 \times d_{\text{model}}$

### 残差连接
$$
T_{\text{final}} = \text{LayerNorm}( \text{FFN}(T) + T )
$$

---

## 输出映射与预测
### 词表投影
$`
\text{Logits} = T_{\text{final}} W_{\text{vocab}}, \quad W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times V}
`$

## 概率生成过程

### 1. 输出层计算
- **输入隐藏状态**  
  $`H \in \mathbb{R}^{n \times d_{\text{model}}}`$  （$`n`$: 当前序列长度，$`d_{\text{model}}`$: 模型维度）

- **词表空间投影**  
  $$\text{Logits} = H W_{\text{vocab}}, \quad W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times V}$$  
  **形状变换**：  
  $$(n \times d_{\text{model}}) \times (d_{\text{model}} \times V) \rightarrow \mathbb{R}^{n \times V}$$

### 2. 概率归一化
$$
\begin{aligned}
P_{\text{token}} &= \text{Softmax}(\text{Logits}) \\
\text{其中} \quad P_{\text{token}}[t,j] &= \frac{e^{\text{Logits}[t,j]}}{\sum_{k=1}^V e^{\text{Logits}[t,k]}}
\end{aligned}
$$

### 3. 条件概率输出
- **联合概率分解**  
  $$p(y_{1:n}) = \prod_{t=1}^n p(y_t|y_{<t})$$

- **当前步预测**  
  $$\boxed{p(y_t|y_{<t}) = P_{\text{token}}[t,:] \in \mathbb{R}^V}$$
  依照当前步预测的概率分布，根据采样或过滤算法选择下一个 token

## 训练阶段

### 交叉熵损失计算
交叉熵损失计算（假设是平均损失）
$$\mathcal{L} = -\frac{1}{n} \sum_{t=1}^n \log P_{\text{token}}[t, y_t^*]
$$
  - $n$：序列长度  
  - $y_t^*$ ：第 \( t \) 步的真实 token（Ground Truth）  
  - $P_{\text{tokens}}[t, y_t^*]$ ：模型对真实 token 的预测概率

**梯度计算**：  

1. **反向传播**：  
   - 计算损失 \( L \) 对模型参数 \( \theta \) 的梯度：  
 ```math
     \nabla_\theta L = -\frac{1}{n} \sum_{t=1}^{n} \nabla_\theta \log P_{\text{tokens}}[t, y_t^*]
``` 
   - 梯度方向指示参数更新的方向。  

2. **参数更新**：  
   - 使用优化器（如 Adam）更新参数：  
 ```math
     \theta \leftarrow \theta - \eta \cdot \nabla_\theta L
``` 
$\eta$ ：学习率  


## 推理阶段

### 自回归生成流程
1. **初始化**  
   $`y_0 = \text{[<START>]} `$,$`H_0 \in \mathbb{R}^{1 \times d_{\text{model}}}`$
3. **迭代步骤**（第$`t`$步）：
即新预测的token加入目标序列，然后继续下一轮预测
- **输入序列**：  
  $y_{0:t-1} $
- **更新隐藏状态**：  
  $H_t = \text{Decoder}(y_{0:t-1}) $ 
- **计算 logits**：  
  $\text{Logits} = H_t[-1,:] \, W_{\text{vocab}} \in \mathbb{R}^V $  
- **采样下一个 token**：  
  $y_t \sim \text{softmax}(\text{Logits})$  
  （可通过贪心、随机采样等方式）  
- **更新序列**：  
  $y_{0:t} = [y_{0:t-1}, y_t] $  


5. **终止条件**  
   $y_t = \text{<END>}$ 或 $`t \geq t_{\text{max}}`$
