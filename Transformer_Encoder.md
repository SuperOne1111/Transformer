# Transformer Encoder 详细流程

## 1. 输入预处理
### Tokenization & Embedding
- **输入**：自然语言文本（如句子）。
- **分词**：通过分词算法（如BPE、WordPiece）将文本拆分为Token序列。
- **映射为ID**：将Token转换为词汇表中的整数ID。
- **词向量映射**：通过Embedding层将ID转为词向量，组成初始矩阵 $X$（形状：（序列长度, 模型维度 $d_{model}$））。
- **位置编码**：对 $X$ 添加位置编码（Positional Encoding），得到最终的输入矩阵：
  \[
  X = $\text{Embedding}$($\text{Token}$ ) + $\text{PositionalEncoding}$
  \]

---

## 2. Encoder Layer（重复N次）
### 2.1 自注意力机制（Self-Attention）
#### (1) 计算Q/K/V矩阵
- 通过线性变换生成 $\text{Query (Q)}、\text{Key (K)}、\text{Value (V)}$：
  
  $$Q = X  \cdot  W_Q^T, \quad K = X \cdot W_K^T, \quad V = X \cdot W_V^T$$
 
  - 权重矩阵 $W_Q$, $W_K$, $W_V$ 形状为 $[d_{model}, d_k]$（单头注意力时 $d_k = d_{model}$，多头时$`d_k = d_v=d_{model}/h`$,h是头的个数）。
  - 若为**多头注意力**，每个头有独立的 $W_Q^i$, $W_K^i$, $W_V^i$，输出拼接后通过 $W_O$ 融合。

#### (2) 缩放点积注意力
- 计算注意力分数并缩放：
  
  $A =  \frac{Q \cdot K^T}{\sqrt{d_k}} $
  
  - $d_k$ 为注意力头维度（如64），用于防止梯度消失。

#### (3) Softmax与加权求和
- 对 $A$ 的每一行应用Softmax（沿Key方向）：
 
   $A_{\text{softmax}}  = \text{Softmax}(A)$
 
- 加权求和得到输出：
 
  $Output_{\text{attention}}= A_{\text{softmax}} \cdot V$
  
  

#### (4) 残差连接 & 层归一化
- 残差连接：
 
  $$Z = X + \text{Output}_{\text{attention}}$$
 
- 层归一化（LayerNorm）：
  
  $$Z_{\text{norm}} = \text{LayerNorm}(Z)$$
 

---

### 2.2 前馈神经网络（FFN）
#### (1) 第一层（升维 + 激活函数ReLU作为样例）
$$
\text{Intermediate} = \text{ReLU}(Z_{\text{norm}} \cdot W_1 + b_1)
$$
- $W_1$ 形状：[$`d_{model}`$, $d_{ff}$]（如 $d_{\text{ff}}$=2048），$`b_1`$ 形状[$`d_{\text{ff}}`$]。

#### (2) 第二层（降维）
$$
FFN(Z_{\text{norm}}) = \text{Intermediate} \cdot W_2 + b_2
$$
- $W_2$ 形状：[$`d_{ff}`$, $d_{model}$]，$`b_2`$ 形状 [$`d_{model}`$]。

#### (3) 残差连接 & 层归一化
$$
Output_{\text{FFN}} = LayerNorm(Z_{\text{norm}} + FFN(Z_{\text{norm}}))
$$

---

## 3. 输出
- 重复N次Encoder Layer后，得到最终的上下文感知表示矩阵：
  
  $`Encoder Output=[Output_{FFN}]N`$
  

---

## 关键参数说明
| 符号              | 含义                          | 典型值         |
|-------------------|-------------------------------|---------------|
| $d_{model}$       | 模型维度（如词向量维度）       | 512, 768      |
| $d_k$             | 注意力头的Key/Query维度        | 64            |
| $d_v$             | 注意力头的Value维度            | 64            |
| $d_{ff}$          | FFN中间层维度                 | 2048          |
| $N$               | Encoder层重复次数             | 6, 12         |

---

## 流程图（简化版）
```mermaid
graph TD
    A[输入文本] --> B[Tokenization]
    B --> C[Embedding + 位置编码]
    C --> D[Encoder Layer × N]
    D -->|自注意力| E[计算Q/K/V → Scaled Dot-Product → Softmax → Output]
    E --> F[残差连接 + LayerNorm]
    F --> G[FFN: ReLU → 线性层]
    G --> H[残差连接 + LayerNorm]
    H --> I[输出]
