# Split-KV Attention：Online Softmax 与 LSE Reduce

Split-KV Attention 将同一个 query 对应的 KV 序列切成多个互不重叠的分片，分别计算局部 attention，再合并成全局结果。它解决的是长序列 attention 的并行度问题：多个计算单元可以同时处理不同 KV 区间，而不必由一个计算单元串行扫描完整序列。

虽然 softmax 需要对全部 token 做全局归一化，但它仍然可以被分片计算。关键在于：每个分片不仅要输出局部 attention 结果，还要输出该分片的 softmax 总质量。这个总质量通常用 Log-Sum-Exp（LSE）表示，以保证数值稳定。

---

## 1. Split-KV 为什么数学上成立

固定一个 query head，令第 `j` 个 KV token 的 attention score 为：

$$
s_j = q \cdot k_j \times \text{scale}
$$

标准 attention 的分母、分子和输出分别为：

$$
Z = \sum_j e^{s_j}
$$

$$
N = \sum_j e^{s_j}v_j
$$

$$
O = \frac{N}{Z}
$$

将全部有效 KV token 划分成互不重叠的集合 $S_0,S_1,\ldots,S_{R-1}$，要求这些集合的并集恰好覆盖完整 KV 序列：

$$
S = \bigcup_r S_r, \qquad S_a \cap S_b = \varnothing \quad (a \ne b)
$$

对第 `r` 个分片定义：

$$
Z_r = \sum_{j \in S_r} e^{s_j}
$$

$$
N_r = \sum_{j \in S_r} e^{s_j}v_j
$$

$$
O_r = \frac{N_r}{Z_r}
$$

由于求和对分区可加：

$$
Z = \sum_r Z_r
$$

$$
N = \sum_r N_r = \sum_r Z_r O_r
$$

所以全局输出可以由局部输出重建：

$$
O = \frac{\sum_r Z_r O_r}{\sum_r Z_r}
$$

因此，KV 切分只改变计算和归约顺序，不改变 attention 所覆盖的 token 集合，也不改变精确算术下的最终结果。

这里不能只保存 $O_r$。局部输出已经除掉了自己的分母，丢失了该分片在全局 softmax 中所占的质量；要正确合并，还必须保存 $Z_r$ 或它的稳定表示。

---

## 2. 分片内部：Online Softmax

一个 KV 分片通常仍然很长，需要再按 block 或 tile 顺序处理。Online Softmax 不保存全部 score，而是维护三个状态：

$$
m = \max_j s_j
$$

$$
l = \sum_j e^{s_j-m}
$$

$$
acc = \sum_j e^{s_j-m}v_j
$$

其中：

- $m$ 是当前已处理 token 的最大 score；
- $l$ 是减去最大值后的 softmax 分母；
- $acc$ 是同一指数基准下尚未归一化的 attention 分子。

当前集合的 attention 输出为（在这个操作后，减去max的操作被抵消了，相当于没有减去max的操作的结果，所以就等于上面的$O_r$）：

$$
O = \frac{acc}{l}
$$

### 2.1 两组 Online Softmax 状态如何合并

假设旧集合 A 的状态为 $(m_a,l_a,acc_a)$，新集合 B 的状态为 $(m_b,l_b,acc_b)$。合并后的最大值是：

$$
m = \max(m_a,m_b)
$$

两组状态必须转换到同一个指数基准：

$$
l = e^{m_a-m}l_a + e^{m_b-m}l_b
$$

$$
acc = e^{m_a-m}acc_a + e^{m_b-m}acc_b
$$

最终输出仍然是 $O=acc/l$。

实际逐块计算时，新 tile 的指数权重可以直接基于新的最大值 $m$ 计算，因此通常只需显式缩放旧状态：

$$
\alpha = e^{m_{old}-m_{new}}
$$

$$
l_{new} = \alpha l_{old} + l_{tile}
$$

$$
acc_{new} = \alpha acc_{old} + acc_{tile}
$$

### 2.2 为什么可以在 PV 之后缩放旧结果

当最大值发生变化时，旧 token 的所有 softmax 权重都乘以同一个标量 $\alpha$。矩阵乘法对这种统一标量缩放是线性的：

$$
(\alpha P_{old})V_{old} = \alpha(P_{old}V_{old})
$$

等价地：

$$
\sum_j (\alpha p_j)v_j
= \alpha\sum_j p_jv_j
= \alpha * acc_{old}
$$

因此，即使旧的概率与 value 已经完成乘加，也可以直接缩放累计结果 $acc_{old}$，不需要重新计算旧 token。分母 $l_{old}$ 必须乘以同一个 $\alpha$，这样分子和分母才继续处在相同的指数基准下。

这个等价要求 $\alpha$ 对同一个 query row 的全部旧 token 是统一标量。如果缩放因子随 token 或输出维度变化，就不能这样穿过 PV 运算。

---

## 3. 分片摘要：为什么使用 LSE

对第 `r` 个分片，原始 softmax 总质量为：

$$
Z_r = \sum_{j\in S_r}e^{s_j}
$$

Online Softmax 保存的缩放分母为：

$$
l_r = \sum_{j\in S_r}e^{s_j-m_r}
$$

因为 $m_r$ 对分片内所有 token 都是同一个常数：

$$
Z_r = e^{m_r}l_r
$$

取自然对数得到：

$$
LSE_r = \log Z_r = m_r + \log l_r
$$

因此，每个分片只需输出两个量：

$$
PartialOut_r = \frac{acc_r}{l_r} = O_r
$$

$$
PartialLSE_r = m_r + \log l_r = \log Z_r
$$

`PartialOut` 表示局部归一化后的 value，`PartialLSE` 表示该分片在全局 softmax 中的总质量。使用 LSE 而不是直接保存 $Z_r$，可以避免直接计算大指数造成上溢。

对于空分片，可以定义 $LSE_r=-\infty$，这样它在后续归约中的权重自然为零。

---

## 4. 分片之间：LSE Reduce

理论上，全局输出可以直接写成：

$$
O = \frac{\sum_r e^{LSE_r}O_r}{\sum_r e^{LSE_r}}
$$

但直接计算 $e^{LSE_r}$ 仍可能溢出。稳定做法是先求：

$$
L_{max} = \max_r LSE_r
$$

再定义每个分片的相对权重：

$$
w_r = e^{LSE_r-L_{max}}
$$

最终结果为：

$$
O = \frac{\sum_r w_rO_r}{\sum_r w_r}
$$

因为：

$$
w_r = \frac{Z_r}{e^{L_{max}}}
$$

所有分片的质量都除以同一个公共因子，该因子会在最终分子和分母中抵消：

$$
\frac{\sum_r w_rO_r}{\sum_r w_r}
= \frac{\sum_r Z_rO_r}{\sum_r Z_r}
= \frac{\sum_r N_r}{\sum_r Z_r}
= O
$$

这证明了 LSE Reduce 与对完整 KV 序列一次性执行 softmax 和 PV 在数学上等价。

### 为什么不能平均局部输出

不能使用：

$$
O \ne \frac{1}{R}\sum_r O_r
$$

不同分片的 token 数量、score 分布和 softmax 总质量通常不同。即使两个分片的最大 score 接近，包含更多中等 score 的分片也可能拥有更大的 $Z_r$。LSE 同时编码了最大 score 和其余 token 的累计贡献，因此必须用 LSE 恢复权重。

---

## 5. 分片内合并与分片间合并的关系

两级归约本质上是同一个可结合的 softmax 合并操作，只是状态表示不同：

| 对比项 | 分片内 Online Softmax | 分片间 LSE Reduce |
|---|---|---|
| 合并对象 | 当前分片中的多组 token | 多个 KV 分片 |
| 状态 | $(m,l,acc)$ | $(LSE,O)$ |
| 输出是否已归一化 | 否，$acc$ 尚未除以 $l$ | 是，$O$ 是局部输出 |
| 稳定基准 | token score 的最大值 $m$ | 分片总质量对数的最大值 $\max(LSE)$ |

两种表示之间可以互相对应：

$$
LSE = m + \log l
$$

$$
O = \frac{acc}{l}
$$

$(m,l,acc)$ 适合持续接收新的 token tile；$(LSE,O)$ 更适合把已完成的局部 attention 压缩成稳定、紧凑的摘要，再进行全局归约。

减去最大值只是数值稳定技巧，不会改变结果。对于任意常数 $c$：

$$
\frac{\sum_j e^{s_j-c}v_j}{\sum_j e^{s_j-c}}
= \frac{e^{-c}\sum_j e^{s_j}v_j}{e^{-c}\sum_j e^{s_j}}
= \frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}
$$

公共缩放因子会在归一化时抵消。

---

## 6. PyTorch 参考实现

下面的代码抽取了 Split-KV kernel 的核心计算顺序，并用纯 PyTorch 表达。为了突出数学原理，示例使用连续的 K/V 张量，不包含 PagedAttention 的逻辑块到物理块映射，也不包含设备调度。

张量形状约定：

```text
q: [H, D]       # 一个 sequence 的 H 个 query heads
k: [N, D]       # N 个 KV token，共享一组 K/V（MQA 形式）
v: [N, D]
```

其中每个 KV split 又按 `tile_size` 逐块处理，因此代码同时展示了两级归约：

1. `online_softmax_partial`：对应分片内的 Online Softmax；
2. `lse_reduce`：对应分片间的 LSE Reduce；
3. `split_kv_attention`：负责切分 KV 并串起以上两个阶段。

```python
import math

import torch


def online_softmax_partial(q, k, v, scale, tile_size=128):
    """计算一个 KV split 的 (PartialOut, PartialLSE)。

    q: [H, D]
    k: [N_split, D]
    v: [N_split, D]

    返回：
        partial_out: [H, D]
        partial_lse: [H]
    """
    q = q.float()
    k = k.float()
    v = v.float()

    num_heads, head_dim = q.shape
    num_tokens = k.shape[0]

    # 与 kernel 相同，Online Softmax 状态使用 FP32。
    m = torch.full(
        (num_heads,), float("-inf"), device=q.device, dtype=torch.float32
    )
    l = torch.zeros(num_heads, device=q.device, dtype=torch.float32)
    acc = torch.zeros(
        (num_heads, head_dim), device=q.device, dtype=torch.float32
    )

    for start in range(0, num_tokens, tile_size):
        k_tile = k[start : start + tile_size]  # [T, D]
        v_tile = v[start : start + tile_size]  # [T, D]

        scores = q @ k_tile.T * scale          # [H, T]

        # 先把旧状态和当前 tile 转换到共同的新最大值基准。
        m_new = torch.maximum(m, scores.max(dim=-1).values)  # [H]
        p = torch.exp(scores - m_new[:, None])               # [H, T]
        alpha = torch.exp(m - m_new)                          # [H]

        # alpha 同时缩放旧分母和旧 PV 累计结果。
        l = l * alpha + p.sum(dim=-1)
        acc = acc * alpha[:, None] + p @ v_tile
        m = m_new

    # 空 split 对 reduce 的贡献应为 0：Out=0，LSE=-inf。
    empty = l == 0
    safe_l = torch.where(empty, torch.ones_like(l), l)
    partial_out = torch.where(
        empty[:, None], torch.zeros_like(acc), acc / safe_l[:, None]
    )
    partial_lse = torch.where(
        empty, torch.full_like(m, float("-inf")), m + torch.log(safe_l)
    )
    return partial_out, partial_lse


def lse_reduce(partial_out, partial_lse):
    """按 PartialLSE 合并多个 split。

    partial_out: [R, H, D]
    partial_lse: [R, H]
    返回:        [H, D]
    """
    valid = partial_lse != float("-inf")
    lse_max = partial_lse.max(dim=0).values       # [H]

    # 所有 split 都为空时，避免计算 -inf - (-inf)。
    has_value = valid.any(dim=0)
    safe_lse_max = torch.where(
        has_value, lse_max, torch.zeros_like(lse_max)
    )
    delta = torch.where(
        valid,
        partial_lse - safe_lse_max[None, :],
        torch.zeros_like(partial_lse),
    )
    weights = torch.where(valid, torch.exp(delta), torch.zeros_like(delta))

    denominator = weights.sum(dim=0)              # [H]
    output_acc = (weights[:, :, None] * partial_out).sum(dim=0)
    safe_denominator = torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        denominator,
    )
    output = output_acc / safe_denominator[:, None]
    return torch.where(
        (denominator == 0)[:, None], torch.zeros_like(output), output
    )


def split_kv_attention(q, k, v, num_splits, scale=None, tile_size=128):
    """Split-KV Attention 的两阶段参考实现。"""
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")
    if k.shape != v.shape or q.shape[1] != k.shape[1]:
        raise ValueError("q, k, v shapes are incompatible")

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # tensor_split 会形成无遗漏、无重叠的连续 KV 分区；当
    # num_splits > N 时可能产生空 split，partial 函数也能处理。
    k_splits = torch.tensor_split(k, num_splits, dim=0)
    v_splits = torch.tensor_split(v, num_splits, dim=0)

    partial_outs = []
    partial_lses = []
    for k_split, v_split in zip(k_splits, v_splits):
        out_r, lse_r = online_softmax_partial(
            q, k_split, v_split, scale, tile_size
        )
        partial_outs.append(out_r)
        partial_lses.append(lse_r)

    partial_out = torch.stack(partial_outs, dim=0)  # [R, H, D]
    partial_lse = torch.stack(partial_lses, dim=0)  # [R, H]
    return lse_reduce(partial_out, partial_lse)


def direct_attention(q, k, v, scale=None):
    """不切分的完整 softmax，用于对照。"""
    q = q.float()
    k = k.float()
    v = v.float()
    if k.shape[0] == 0:
        return torch.zeros_like(q)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.T * scale
    return torch.softmax(scores, dim=-1) @ v


# 可运行的等价性检查。
torch.manual_seed(0)
H, N, D = 4, 37, 16
q = torch.randn(H, D)
k = torch.randn(N, D)
v = torch.randn(N, D)

out_split = split_kv_attention(
    q, k, v,
    num_splits=5,  # KV 先切成 5 个 split
    tile_size=3,   # 每个 split 内再按 3 个 token 做 Online Softmax
)
out_direct = direct_attention(q, k, v)

torch.testing.assert_close(out_split, out_direct, atol=1e-5, rtol=1e-5)
print("max error:", (out_split - out_direct).abs().max().item())
```

### 6.1 代码与公式的对应关系

| 代码 | 数学含义 |
|---|---|
| `m`, `l`, `acc` | Online Softmax 状态 $(m,l,acc)$ |
| `m_new = maximum(...)` | 为旧状态与新 tile 选择统一指数基准 |
| `alpha = exp(m - m_new)` | 将旧状态转换到新最大值基准 |
| `acc * alpha + p @ v_tile` | 利用线性性在 PV 后 rescale |
| `m + log(l)` | 当前 split 的 $LSE=\log Z_r$ |
| `exp(lse - lse_max)` | 恢复各 split 的相对 softmax 质量 |
| `weighted sum / denominator` | 重建完整 attention 输出 |

这份参考代码让各 split 串行执行，目的是展示数据依赖；真实 Split-KV kernel 可以并行计算各个 partial，因为每个 split 只读取自己的 K/V 区间，直到 LSE Reduce 阶段才需要合并。

---

## 7. 数值示例

设四个 token 的 score 和二维 value 为：

```text
score = [1, 2, 0, -1]

V0 = [1,  0]
V1 = [0,  2]
V2 = [3,  1]
V3 = [2, -1]
```

切成两个分片：

```text
Split A: token 0, 1
Split B: token 2, 3
```

### Split A

```text
m_A   = 2
l_A   = exp(1 - 2) + exp(2 - 2)
      = 1.367879

acc_A = exp(-1) * [1, 0] + 1 * [0, 2]
      = [0.367879, 2.000000]

O_A   = acc_A / l_A
      = [0.268941, 1.462117]

LSE_A = 2 + log(1.367879)
      = 2.313262
```

### Split B

```text
m_B   = 0
l_B   = exp(0) + exp(-1)
      = 1.367879

acc_B = 1 * [3, 1] + exp(-1) * [2, -1]
      = [3.735759, 0.632121]

O_B   = acc_B / l_B
      = [2.731059, 0.462117]

LSE_B = log(1.367879)
      = 0.313262
```

### LSE Reduce

```text
L_max = 2.313262

w_A = exp(LSE_A - L_max) = 1
w_B = exp(LSE_B - L_max) = exp(-2) = 0.135335

O = (w_A * O_A + w_B * O_B) / (w_A + w_B)
  = [0.562433, 1.342914]
```

直接在四个 token 上计算完整 attention，也会得到：

```text
O_direct = [0.562433, 1.342914]
```

因此 $O_{split}=O_{direct}$。

---

## 8. 正确性条件与精度边界

### 数学等价成立的条件

1. 各分片使用相同的 query、scale、mask 和 attention 语义。
2. 所有分片对有效 KV token 构成无遗漏、无重叠的完整分区。
3. 每个分片同时保留局部输出和 LSE，不能只保留局部输出。
4. 空分片必须映射为零权重，例如使用 $LSE=-\infty$。
5. 每个 sequence 和 head 必须独立归约，不能混合不同 query 的状态。

### 有限精度下的差异

Split-KV 在实数精确算术下与完整 attention 等价，但浮点实现通常不保证逐 bit 相同，原因包括：

- 分片改变了点积和加法的累加顺序；
- 中间概率、value 和输出可能发生 dtype 转换；
- 多级归约引入了不同的舍入路径。

这些通常表现为合理范围内的浮点误差，而不是算法不等价。

如果中间概率经过低比特非线性量化，严格等价性可能不再成立。一般来说，设量化算子为 $Q$：

$$
Q(\alpha P) \ne \alpha Q(P)
$$

因此，“先量化再缩放”和“统一指数基准后再量化”可能得到不同结果。普通浮点路径的线性 rescale 证明不能直接推广到这类量化路径；此时 Split-KV 应被视为近似计算，并单独评估误差。

---

## 9. 核心结论

- KV 可以切分，是因为 softmax 的分子和分母都能按 token 分区求和。
- 每个分片的充分摘要是局部输出 $O_r$ 和局部总质量的对数 $LSE_r$。
- 分片间不能等权平均，必须按 $e^{LSE_r}$ 对局部输出加权。
- Online Softmax 的 $(m,l,acc)$ 与 LSE Reduce 的 $(LSE,O)$ 是同一合并规律的两种状态表示。
- 将旧概率的统一 rescale 放到 PV 之后仍然等价，来源于矩阵乘法的线性性。
- 减去最大值只用于数值稳定，公共指数因子会在归一化时抵消。
- 普通浮点路径数学等价但未必 bitwise 一致；非线性低比特量化可能破坏严格等价性。
