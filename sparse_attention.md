# Sparse Attention Notes

本文整理三类 sparse attention / long-context attention 结构：

1. 基础 DSA：DeepSeek-V3.2 / GLM-5 / GLM-5.1 的 MLA + Lightning Indexer + Sparse Attention。
2. GLM-5.2 DSA：在基础 DSA 上加入 IndexShare / idxcache，多层复用 top-k indices。
3. DeepSeek V4 attention：sliding window KV + HCA/CSA compressed KV。

重点关注 attention 中数据如何流动、shape 如何变化、mask 或 sparse index 如何约束可见性。

---

## 1. 基础 DSA：DeepSeek-V3.2 / GLM-5 / GLM-5.1

### 核心结论

基础 DSA 可以理解成：

```text
MLA projection + Lightning Indexer + Sparse Attention
```

它和 DeepSeek V4 的 compressed KV 方式不同。DeepSeek V4 是把 compressed KV 追加到 KV 序列后面；基础 DSA 则是先用 learned indexer 选出 top-k 历史 token，然后主 attention 只对这些 token 计算。

核心链路：

```text
hidden_states
  -> MLA 得到 q / k / v
  -> Lightning Indexer 给历史 token 打分
  -> topk_indices: [B, T, K]
  -> gather selected K/V
  -> sparse attention over K selected tokens
  -> o_proj
```

在 vLLM-Ascend 中，这条链路通常对应：

```text
npu_lightning_indexer
  -> topk_indices
  -> npu_sparse_flash_attention
```

DeepSeek-V3.2、GLM-5、GLM-5.1 主结构相同，差异主要在模型配置、RoPE 风格、GLM 的 `glm_moe_dsa` 分支以及工程算子细节。

### MLA 主路径

设：

```text
B = batch
T = query length / current sequence length
S = key length
H = main attention heads
K = top_k selected tokens
Dq = qk_nope_dim + qk_rope_dim
Dv = v_head_dim
R_q = q_lora_rank
R_kv = kv_lora_rank
```

输入：

```text
hidden_states: [B, T, hidden_size]
```

Q 路径：

```text
q_a(hidden_states)       -> [B, T, R_q]
q_a_norm                 -> [B, T, R_q]
q_b                      -> [B, T, H * Dq]
view                     -> [B, T, H, Dq]
split q_nope / q_rope    -> [B, T, H, qk_nope_dim], [B, T, H, qk_rope_dim]
RoPE(q_rope)             -> [B, T, H, qk_rope_dim]
```

KV 路径：

```text
kv_a(hidden_states)      -> [B, S, R_kv + qk_rope_dim]
split kv_c / k_rope      -> [B, S, R_kv], [B, S, qk_rope_dim]
kv_a_norm(kv_c)          -> [B, S, R_kv]
RoPE(k_rope)             -> [B, S, qk_rope_dim]
kv_b(kv_c)               -> [B, S, H * (qk_nope_dim + Dv)]
view                     -> [B, S, H, qk_nope_dim + Dv]
split k_nope / value     -> [B, S, H, qk_nope_dim], [B, S, H, Dv]
```

### Lightning Indexer

Lightning Indexer 用轻量的 indexer query/key 来选 token，而不是直接用主 attention 的 Q/K 做全量 attention。

常见形状：

```text
q_c: [B, T, R_q]
kv_c: [B, S, R_kv]
```

Indexer query：

```text
q_c -> wq_b                  -> [B, T, index_heads * index_dim]
view                         -> [B, T, index_heads, index_dim]
RoPE                         -> [B, T, index_heads, index_dim]
```

Indexer key 和 head weights：

```text
kv_c -> wk_weights_proj      -> [B, S, index_dim + index_heads]
split k_li / weights         -> [B, S, index_dim], [B, S, index_heads]
k_norm + RoPE(k_li)          -> [B, S, index_dim]
```

打分：

```text
q_li:       [B, T, index_heads, index_dim]
k_li:       [B, S, index_dim]
weights:    [B, S, index_heads]

head_scores = q_li @ k_li^T
head_scores: [B, T, index_heads, S]

weighted sum over index_heads
scores:      [B, T, S]
```

然后加 causal mask，选 top-k：

```text
topk_indices:   [B, T, K]
selected_valid: [B, T, K]
```

这里的 sparse 不是 block sparse mask，而是 token index sparse：每个 query token 只保留 K 个历史 token 位置。

### Sparse Attention

主 attention 只 gather `topk_indices` 指向的 K/V。

```text
k_nope:       [B, S, H, qk_nope_dim]
k_rope:       [B, S, qk_rope_dim]
value:        [B, S, H, Dv]
topk_indices: [B, T, K]
```

gather 后：

```text
k_nope_sel: [B, T, K, H, qk_nope_dim]
k_rope_sel: [B, T, K, H, qk_rope_dim]
v_sel:      [B, T, K, H, Dv]
```

拼出 selected key：

```text
q:     [B, T, H, Dq]
k_sel: [B, T, K, H, Dq]
```

attention：

```text
scores = q · k_sel
scores: [B, T, H, K]
prob:   [B, T, H, K]
out:    [B, T, H, Dv]
```

最后：

```text
reshape -> [B, T, H * Dv]
o_proj  -> [B, T, hidden_size]
```

### 数据流图

```mermaid
flowchart TD
    A["hidden_states<br/>[B, T, hidden]"] --> B["q_a + norm<br/>q_c [B, T, Rq]"]
    B --> C["q_b<br/>main q [B, T, H, Dq]"]

    A --> D["kv_a<br/>kv_c + k_rope"]
    D --> E["kv_a_norm<br/>kv_c [B, S, Rkv]"]
    E --> F["kv_b<br/>k_nope + value"]

    B --> G["wq_b<br/>q_li [B, T, Ih, Id]"]
    E --> H["wk_weights_proj<br/>k_li [B, S, Id]<br/>weights [B, S, Ih]"]

    G --> I["Lightning Indexer<br/>scores [B, T, S]"]
    H --> I
    I --> J["topk_indices<br/>[B, T, K]"]

    C --> K["Sparse Attention"]
    F --> K
    J --> K
    K --> L["attn output<br/>[B, T, H, Dv]"]
    L --> M["o_proj<br/>[B, T, hidden]"]
```

### 小 Shape 示例

假设：

```text
B = 1
T = S = 8
H = 2
Dq = 4
Dv = 4
index_heads = 2
index_dim = 4
top_k = 3
```

那么：

```text
q:             [1, 8, 2, 4]
k/value:       [1, 8, 2, 4]
q_li:          [1, 8, 2, 4]
k_li:          [1, 8, 4]
index scores:  [1, 8, 8]
topk_indices:  [1, 8, 3]

selected k:    [1, 8, 3, 2, 4]
selected v:    [1, 8, 3, 2, 4]
attn scores:   [1, 8, 2, 3]
attn output:   [1, 8, 2, 4]
final output:  [1, 8, hidden_size]
```

对比 dense causal attention：

```text
dense logits:  [B, H, T, S]
DSA logits:    [B, T, H, K]
```

当 `K << S` 时，主 attention 的计算和访存就从全历史变成 top-k selected history。

---

## 2. GLM-5.2 DSA：IndexShare / idxcache

### 核心结论

GLM-5.2 仍然是 DSA：

```text
Lightning Indexer + Sparse Attention
```

但它在基础 DSA 上加入了 IndexShare / idxcache：

```text
每 4 个 transformer layers 共享一份 top-k indices
```

也就是说，4 层中的第一层运行 lightning indexer，得到 `topk_indices`；后面 3 层不再重新跑 indexer，而是复用这份 indices。

关键点：

```text
复用的是 top-k token 的位置索引。
不是复用 attention output。
不是复用 K/V。
```

每一层仍然会根据自己的 hidden states 计算自己的 Q/K/V，并执行 sparse attention，只是省掉了部分层的 indexer dot product 和 top-k 操作。

### 和基础 DSA 的区别

基础 DSA：

```text
layer i:
  lightning_indexer -> topk_indices_i
  sparse_attention(topk_indices_i)
```

GLM-5.2 IndexShare：

```text
layer 0:
  lightning_indexer -> topk_indices_group0
  sparse_attention(topk_indices_group0)

layer 1:
  reuse topk_indices_group0
  sparse_attention(topk_indices_group0)

layer 2:
  reuse topk_indices_group0
  sparse_attention(topk_indices_group0)

layer 3:
  reuse topk_indices_group0
  sparse_attention(topk_indices_group0)

layer 4:
  lightning_indexer -> topk_indices_group1
  sparse_attention(topk_indices_group1)
```

工程实现中常见名字包括：

```text
skip_topk
topk_indices_buffer
use_index_cache
_get_indexcache_topk_indices
_update_indexcache_topk_indices
```

这些都指向同一个核心思想：部分层计算 top-k，其他层从 buffer/cache 复用 top-k。

### 单层 Shape

单层 sparse attention 的 shape 与基础 DSA 基本一致：

```text
hidden_states: [B, T, hidden]
q:             [B, T, H, D]
k:             [B, T, H, D]
v:             [B, T, H, D]
topk_indices:  [B, T, K]
selected k/v:  [B, T, K, H, D]
scores:        [B, T, H, K]
out:           [B, T, H, D]
output:        [B, T, hidden]
```

区别只在 `topk_indices` 的来源：

```text
Group leader layer: topk_indices = indexer(hidden_states)
Shared layer:       topk_indices = topk_indices_buffer[group_id]
```

### 4 层 Group 数据流

```mermaid
flowchart TD
    X["Input hidden states"] --> L0["Layer 0<br/>Group leader"]
    L0 --> I0["Lightning Indexer<br/>compute top-k"]
    I0 --> T0["topk_indices_buffer<br/>[B, T, K]"]
    L0 --> A0["Layer 0 Q/K/V<br/>Sparse Attention"]
    T0 --> A0

    A0 --> L1["Layer 1<br/>shared layer"]
    L1 --> A1["Layer 1 Q/K/V<br/>Sparse Attention"]
    T0 --> A1

    A1 --> L2["Layer 2<br/>shared layer"]
    L2 --> A2["Layer 2 Q/K/V<br/>Sparse Attention"]
    T0 --> A2

    A2 --> L3["Layer 3<br/>shared layer"]
    L3 --> A3["Layer 3 Q/K/V<br/>Sparse Attention"]
    T0 --> A3

    A3 --> L4["Layer 4<br/>new group leader"]
    L4 --> I4["Lightning Indexer<br/>compute new top-k"]
    I4 --> T4["topk_indices_buffer<br/>new group"]
    L4 --> A4["Layer 4 Q/K/V<br/>Sparse Attention"]
    T4 --> A4
```

### Group 内部的执行模式

```text
Group: layers 4g, 4g+1, 4g+2, 4g+3

layer 4g:
  hidden_states
    -> lightning_indexer
    -> topk_indices_buffer
  hidden_states
    -> Q/K/V projection
    -> sparse_attention(topk_indices_buffer)

layer 4g+1:
  hidden_states
    -> Q/K/V projection
    -> sparse_attention(topk_indices_buffer)

layer 4g+2:
  hidden_states
    -> Q/K/V projection
    -> sparse_attention(topk_indices_buffer)

layer 4g+3:
  hidden_states
    -> Q/K/V projection
    -> sparse_attention(topk_indices_buffer)
```

### 小 Shape 示例

假设：

```text
B = 1
T = 8
num_layers = 8
index_share_group_size = 4
H = 2
D = 4
K = 3
```

执行模式：

```text
layer 0: run indexer  -> topk_indices [1, 8, 3]
layer 1: reuse layer0 -> topk_indices [1, 8, 3]
layer 2: reuse layer0 -> topk_indices [1, 8, 3]
layer 3: reuse layer0 -> topk_indices [1, 8, 3]

layer 4: run indexer  -> topk_indices [1, 8, 3]
layer 5: reuse layer4 -> topk_indices [1, 8, 3]
layer 6: reuse layer4 -> topk_indices [1, 8, 3]
layer 7: reuse layer4 -> topk_indices [1, 8, 3]
```

每一层自己的 attention shape 仍然是：

```text
q/k/v:          [1, 8, 2, 4]
selected k/v:   [1, 8, 3, 2, 4]
attn scores:    [1, 8, 2, 3]
attn output:    [1, 8, 2, 4]
```

### 计算意义

GLM-5.2 的优化重点不是改掉 sparse attention，而是降低 indexer 本身的成本。

```text
基础 DSA:
  每层都跑 indexer + top-k

GLM-5.2 IndexShare:
  4 层中只有 1 层跑 indexer + top-k
  后 3 层复用 topk_indices
```

因此大致可以省掉 3/4 的 indexer dot product 和 top-k 开销，但每层的 Q/K/V projection、selected K/V gather、sparse attention 仍然存在。

---

## 3. DeepSeek V4 Attention

### 核心结论

DeepSeek V4 的 attention 不是普通全量 causal attention，而是：

```text
局部 sliding attention + 压缩 KV attention
```

普通 token KV 和 compressed KV 会拼接到同一个 KV 序列维度上，然后一起进入 attention softmax。

默认配置大致是：

```text
sliding_window = 128
HCA compress_rate = 128
CSA compress_rate = 4
num_heads = 64
num_kv_heads = 1
head_dim = 512
```

KV 是 shared-KV MQA：

```text
q:  [B, H, S, D]
kv: [B, 1, T, D]
```

进入 attention 前，KV 会 repeat 到所有 query heads：

```text
kv -> [B, H, T, D]
```

### Sliding KV

sliding branch 只保留局部窗口：

```text
每个 token 最多看自己 + 前 127 个 token
```

decode 时，每加入一个新 token，最旧的 sliding KV 会被挤出 cache。

prefill 时，本次输入的完整 KV 会参与计算，但 mask 限制每个 query 只能看 sliding window 内的 token；cache 写回时仍只保留最近窗口，供后续 decode 用。

### Compressed KV

HCA 和 CSA 都会额外生成 compressed KV，并拼接到原始 KV 后面：

```python
kv = torch.cat([kv, compressed_kv], dim=2)
```

这里 `dim=2` 是 token/KV slot 维度。

HCA：

```text
每 128 token -> 1 个 compressed KV
未满 128 的尾段有 cache 时暂存在 buffer
```

CSA：

```text
每 4 token -> compressed entry
有效覆盖宽度约 8 token，步长 4 token
再由 Lightning Indexer 为每个 query 选 top-k compressed entries
```

compressed KV 和 sliding KV 不会在 tensor slot 上重叠，但语义上可能覆盖同一批 token。也就是说，最近 128 个 token 既可能以原始 KV 出现在 sliding window 中，也可能以压缩摘要形式出现在 CSA/HCA compressed KV 中。

### Mask 结构

拼接后的 mask 不再是单纯下三角矩阵，而是：

```text
[ sliding causal/local mask | compressed block mask ]
```

例如 HCA：

```text
原始 token mask:       [B, 1, S, S]
HCA block_bias:        [B, 1, S, C]
拼接后 attention mask: [B, 1, S, S + C]
```

前半部分是 sliding causal band，后半部分是 blockwise step mask。

### Shape 流动示例

假设 prefill：

```text
B = 1
S = 566
hidden_size = 4096
H = 64
D = 512
```

```mermaid
flowchart TD
    A["hidden_states<br/>[1, 566, 4096]"]

    A --> B["q_a_proj<br/>[1, 566, 1024]"]
    B --> C["q_b_proj<br/>[1, 566, 32768]"]
    C --> D["view + transpose<br/>q = [1, 64, 566, 512]"]
    D --> E["RoPE<br/>q = [1, 64, 566, 512]"]

    A --> F["kv_proj<br/>[1, 566, 512]"]
    F --> G["view + transpose<br/>kv = [1, 1, 566, 512]"]
    G --> H2["RoPE<br/>sliding kv = [1, 1, 566, 512]"]

    A --> I{"Layer type?"}

    I --> J["sliding_attention<br/>no compressed KV"]
    J --> K["final kv<br/>[1, 1, 566, 512]"]

    I --> L["HCA<br/>566 // 128 = 4"]
    L --> M["compressed_kv<br/>[1, 1, 4, 512]"]
    M --> N["cat dim=2<br/>final kv = [1, 1, 570, 512]"]

    I --> O["CSA<br/>566 // 4 = 141"]
    O --> P["compressed_kv<br/>[1, 1, 141, 512]"]
    P --> Q["cat dim=2<br/>final kv = [1, 1, 707, 512]"]

    K --> R["repeat_kv"]
    N --> R
    Q --> R

    R --> S2["key/value states<br/>[1, 64, KV_LEN, 512]"]
    E --> T["attention logits<br/>q @ k^T<br/>[1, 64, 566, KV_LEN]"]
    S2 --> T
    T --> U["add mask + sink + softmax"]
    U --> V["attn_output<br/>[1, 64, 566, 512]"]
    V --> W["transpose<br/>[1, 566, 64, 512]"]
    W --> X["reverse RoPE + grouped output proj<br/>[1, 566, 4096]"]
```

对应三种层的 `KV_LEN`：

```text
sliding_attention: KV_LEN = 566
HCA:               KV_LEN = 566 + 4   = 570
CSA:               KV_LEN = 566 + 141 = 707
```

### 小 Mask 示例

假设：

```text
seq_len = 8
sliding_window = 4
HCA compressed blocks = 2
c0 覆盖 k0..k3
c1 覆盖 k4..k7
```

拼接后 mask 是 `[8, 10]`：

```text
        k0   k1   k2   k3   k4   k5   k6   k7 |  c0   c1
q0       0 -inf -inf -inf -inf -inf -inf -inf | -inf -inf
q1       0    0 -inf -inf -inf -inf -inf -inf | -inf -inf
q2       0    0    0 -inf -inf -inf -inf -inf | -inf -inf
q3       0    0    0    0 -inf -inf -inf -inf |    0 -inf
q4    -inf    0    0    0    0 -inf -inf -inf |    0 -inf
q5    -inf -inf    0    0    0    0 -inf -inf |    0 -inf
q6    -inf -inf -inf    0    0    0    0 -inf |    0 -inf
q7    -inf -inf -inf -inf    0    0    0    0 |    0    0
```

整体 attention 是：局部 token-level 精细注意力，加上长程 block-level 压缩注意力，两者在同一个 softmax 里竞争权重。

---

## 4. 三类 Attention 对比

| 类型 | 稀疏/压缩对象 | 主 attention 看什么 | 是否拼接 KV | 是否使用 learned indexer | 复用 top-k |
| --- | --- | --- | --- | --- | --- |
| 基础 DSA | 原始历史 token 的 top-k indices | top-k selected original tokens | 否，gather selected K/V | 是，每层运行 | 否 |
| GLM-5.2 DSA | 原始历史 token 的 top-k indices | top-k selected original tokens | 否，gather selected K/V | 是，但 group leader 才运行 | 是，每 4 层共享 |
| DeepSeek V4 HCA/CSA | 压缩后的 block KV | sliding KV + compressed KV | 是，拼到 KV token 维 | CSA 有 indexer 选 compressed entries；HCA 无 | 否 |

### 直观区别

基础 DSA：

```text
先选 token indices
=> 只 gather top-k token 做 sparse attention
```

GLM-5.2 DSA：

```text
先选 token indices
=> 4 层共享同一份 top-k indices
=> 每层仍用自己的 Q/K/V 做 sparse attention
```

DeepSeek V4：

```text
原始局部 token KV + 额外 compressed KV
=> 拼接后一起 softmax
```

### Dense Attention vs 三类 Sparse Attention

```text
Dense causal attention:
  logits: [B, H, T, S]
  每个 query 对所有可见历史 token 打分

基础 DSA:
  index scores: [B, T, S]
  topk_indices: [B, T, K]
  main logits:  [B, T, H, K]

GLM-5.2 DSA:
  group leader 产生 topk_indices: [B, T, K]
  group 内各层 main logits:       [B, T, H, K]

DeepSeek V4:
  logits: [B, H, T, S_local + C]
  S_local 是 sliding window 或 prefill 中受 sliding mask 约束的 token KV
  C 是 HCA/CSA compressed KV 数量
```

### 总结

这三类方法都在减少长上下文 attention 的成本，但切入点不同：

```text
基础 DSA:
  用 learned indexer 从全历史中选 top-k token，主 attention 只算这些 token。

GLM-5.2 DSA:
  沿用 DSA，但把 top-k indices 缓存/共享到多个层，降低 indexer 开销。

DeepSeek V4:
  用 sliding window 保留局部精细信息，用 compressed KV 保留长程摘要。
```



