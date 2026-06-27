

# 1. 背景：DSA indexer 的瓶颈

DeepSeek Sparse Attention / DSA 的流程可以理解为：

```text
Indexer:
    对所有 prefix tokens 打分
    从 L 个 token 中选 top-k

Sparse MLA:
    只对 top-k tokens 做真正 attention
```

DSA 虽然把主 attention 从 dense attention 降成 sparse attention，但它的 **indexer 仍然需要扫描所有 prefix tokens**。如果 indexer 有 (H_I) 个 heads，上下文长度为 (L)，那么每个 query 的 indexer score computation 大致是：

[
O(H_I \cdot L)
]

这里需要注意：

> **indexer 打分计算和 top-k 的 k 关系不大，主要和 (H_I)、(L) 有关；top-k selector 和后续 Sparse MLA 才和 k 强相关。**

所以优化 DSA indexer pipeline 可以从几个轴入手：

| 优化轴              | 优化对象                  |
| ---------------- | --------------------- |
| layer 轴          | 不要每层都独立跑 indexer      |
| token/block 轴    | 不要扫描所有 token          |
| head 轴           | 不要使用所有 indexer heads  |
| time/decode 轴    | 不要每个 decode step 全量刷新 |
| top-k selector 轴 | score 已经算完后，更快选 top-k |

---

# 2. 相关论文总结

## 2.1 IndexCache：跨层复用 top-k indices

**IndexCache** 的核心是利用 DSA indexer 的跨层冗余：相邻层的 top-k selection 往往很相似，所以不必每一层都完整运行 indexer。它把层分成 **Full layers** 和 **Shared layers**：Full layer 正常跑 indexer，Shared layer 直接复用最近 Full layer 的 top-k indices。论文称这种方式可以移除大量 indexer 计算，例如最多去掉 75% 的 indexer computation，并报告最高 1.82× prefill speedup 和 1.48× decode speedup。([arXiv][1])

流程：

```text
Full layer:
    完整 indexer over all L tokens
    输出 top-k

Shared layer:
    不跑 indexer
    直接复用 Full layer 的 top-k
```

优点是工程实现非常简单；缺点是 Shared layer 直接复用 top-k 会有精度风险，尤其是和 Full layer 间隔 2 层、3 层时，当前层真实想选的 token 可能不在被复用的 top-k 里。

---

## 2.2 HISA：block 级粗筛 + token 级精排

**HISA** 的方向是 token/block 轴优化。它先把 prefix tokens 分块，用 block-level representation 做粗筛，选出可能包含重要 token 的 blocks，然后只在候选 blocks 内做 token-level scoring。

流程：

```text
所有 tokens
   ↓
block pooling
   ↓
选择 top blocks
   ↓
在候选 blocks 内做 token-level DSA
   ↓
选 top-k tokens
```

它的优点是减少 token-level indexer 扫描范围；缺点是 block 级粗筛可能漏掉某些 block 内的关键 token。如果某个 block 的整体分数不高，但里面有一个非常重要的 token，HISA 可能会把整个 block 丢掉。

---

## 2.3 MISA：只激活少数 indexer heads

**MISA: Mixture of Indexer Sparse Attention** 是 head 轴优化。它把 DSA 的 indexer heads 当成 MoE experts，每个 query 通过一个轻量 router 只选择少数 active heads，例如从 64 个 indexer heads 里选 8 个，然后只用这些 heads 扫描所有 tokens。MISA 论文称其是 DSA indexer 的 drop-in replacement，并把 heavy token-level scoring 从所有 heads 降到少量 routed heads。([arXiv][2])

原始 DSA：

```text
64 indexer heads × L tokens
```

MISA：

```text
router 选择 8 active heads
8 indexer heads × L tokens
```

MISA 的 router 会先对 prefix 做 block pooling，再估计每个 head 对当前 query 的重要性：

$$
E_{t,j} =

\frac{1}{M}
\sum_b
|w^I_{t,j} \cdot \text{ReLU}(q^I_{t,j} \cdot \tilde{k}^I_b)|
$$

然后选 top-h active heads。

MISA 有两个版本：

| 版本            | 思路                                               | 优点 | 缺点                        |
| ------------- | ------------------------------------------------ | -- | ------------------------- |
| MISA          | active heads 直接选最终 top-k                         | 快  | routed heads 漏 token 后难恢复 |
| MISA† / MISA+ | active heads 先选 top-m，再用完整 DSA indexer 精排到 top-k | 更稳 | 更慢                        |

MISA† 的思想是：

```text
active heads:
    L → top-m candidate

full DSA indexer:
    top-m → top-k
```

这和我们后面提出的 **Top-M IndexCache** 很像。

---

## 2.4 TISA：decode step 时间复用

**TISA: Exploiting Temporal Locality to Accelerate Indexing Sparse Attention** 利用的是相邻 decode step 的时间局部性。它观察到相邻 decode step 的 top-k selection 高度重合，因此不必每一步都重新计算所有 token 的 score。TISA 维护每层的 score cache，每个 decode step 只刷新 (1/K) 的 token scores，其余位置复用旧 score，然后在完整 score cache 上做 top-k。([OpenReview][3])

流程：

```text
step 0:
    完整计算所有 token scores
    初始化 score_cache

step t:
    只刷新第 t % K 个 partition
    其他 score 复用 cache
    在完整 score_cache 上做 top-k
```

复杂度从：

$
O(H_I L)
$

变成：

$
O(H_I L / K)
$

TISA 的优点是工程侵入性较小，可以复用原 indexer kernel，只是每步输入 slice 变小；缺点是它是近似方法，score cache 中部分分数是 stale scores，因此 top-k 可能和完整 DSA 不一致。

---

## 2.5 GVR Top-K：优化 exact top-k selector

**GVR Top-K / Guess-Verify-Refine** 优化的不是 indexer score computation，而是 score 已经算完之后的 **Top-K selection**。它利用相邻 decode step 的 Top-K 相关性，用上一步 Top-K indices 作为 predictor 来猜当前 threshold，然后 verify candidate count，最后在 shared memory 中 refine 得到 exact Top-K。论文报告 GVR 在 DeepSeek-V3.2 + TensorRT-LLM + Blackwell 上，相比 production radix-select kernel 有平均 1.88× 单算子加速，并保持 bit-exact Top-K 输出。([arXiv][4])

流程：

```text
Guess:
    用 previous-step top-k 读取当前 scores 的这些位置
    估计 threshold

Verify:
    扫描当前 scores，确认 candidate count 合法

Refine:
    在候选集里精确选 top-k
```

GVR 的重点是：

> **它不减少 score computation，但可以在不损失精度的情况下加速 top-k selector。**

所以 GVR 推荐做，但不建议第一优先级。只有当 profile 显示 Top-K selector 已经成为明显瓶颈时，GVR 才最值得投入。

---

# 3. 各论文方法对比

| 方法         | 优化轴                     |    是否近似 | 是否直接减少 score computation | 工程难度 | 主要风险                    |
| ---------- | ----------------------- | ------: | -----------------------: | ---: | ----------------------- |
| IndexCache | layer                   |       是 |                        是 |    低 | Shared layer 直接复用导致精度下降 |
| HISA       | token/block             |       是 |                        是 |   中高 | block 粗筛漏掉关键 token      |
| MISA       | head                    |       是 |                        是 |   中高 | active heads 漏掉重要信号     |
| MISA†      | head + candidate rerank |   是，但更稳 |                        是 |    高 | 多一阶段精排，速度下降             |
| TISA       | decode time             |       是 |                        是 |    中 | stale score 影响 top-k    |
| GVR        | top-k selector          | 否，exact |                        否 |   中高 | 只在 Top-K 成为瓶颈时收益明显      |

---

# 4. 进一步优化思路

## 4.1 Top-M IndexCache / Rerank-IndexCache



原始 IndexCache 的问题是：Shared layer 直接复用 Full layer 的 top-k，精度容易下降。我们提出：

> **Full layer 不只输出 top-k，而是输出更大的 top-m；Shared layer 不直接复用 top-k，而是在 top-m 候选中用自己的 indexer 重新打分，再选 top-k。**

流程：

```text
Full layer:
    完整计算 scores over all L tokens
    取 top-m，例如 8192

Shared layer:
    只在 top-m candidates 上计算自己的 indexer scores
    从 top-m 里选 top-k，例如 2048
```

公式上：

$$
T_s^k = \text{TopK}_{i \in T_f^m}(I_s(i), k)
$$

其中：

* (T_f^m)：Full layer 的 top-m 候选；
* (I_s(i))：Shared layer 自己对候选 token (i) 的 indexer score；
* (T_s^k)：Shared layer 最终选出的 top-k。

它的核心指标是：

$$
Recall@M =
\frac{|TopM_{Full} \cap TopK_{Shared}^{true}|}{K}
$$

如果 Full layer 的 top-m 能覆盖 Shared layer 真实 top-k 的大部分 token，那么 Shared layer 在 top-m 内 rerank 后就能接近完整 DSA。

推荐参数：

```text
top-k = 2048
top-m = 4096 / 8192 / 16384 做 ablation
默认先试 top-m = 8192
```

注意工程上建议用 **8192**，不是 8196，因为 8192 是 2 的幂，对 kernel、buffer、alignment 更友好。

---

## 4.2 三类层策略：Full / Rerank / Reuse

我们进一步提出，不要只把层分成 Full 和 Shared，而是分成三类：

| 层类型    | 行为                              | 成本 | 精度 |
| ------ | ------------------------------- | -: | -- |
| Full   | 完整扫描所有 L tokens                 |  高 | 最高 |
| Rerank | 在 Full layer top-m 内重新打分选 top-k |  中 | 较高 |
| Reuse  | 直接复用 Full layer top-k           |  低 | 较低 |

例如：

```text
F R R F R S F R R
```

含义：

```text
F: Full indexer
R: Rerank in top-m candidates
S: Direct reuse top-k
```

这个比原始 IndexCache 更灵活。对于不敏感层，可以直接 reuse；对于中等敏感层，用 top-m rerank；对于高度敏感层，保留 full indexer。

---

## 4.3 距离自适应 top-m

我们还讨论了：Shared layer 离 Full layer 越远，候选集 top-m 应该越大。

例如：

```text
F S1 S2 S3 F
```

可以设置：

| 层  | 距离 Full layer | top-m |
| -- | ------------: | ----: |
| S1 |             1 |  4096 |
| S2 |             2 |  8192 |
| S3 |             3 | 16384 |

直觉是：

```text
相隔 1 层：Full top-4096 可能够
相隔 2 层：需要 top-8192
相隔 3 层：可能需要 top-16384 或直接 Full
```

最稳的方法是离线统计：

$$
Recall@M(d)
=

\frac{|TopM_{Full} \cap TopK_{Shared}^{true}|}{K}
$$

然后选择满足目标召回率的最小 (M)，比如让 Recall@M 达到 98%。

---

## 4.4 IndexCache + TISA 组合

IndexCache 优化 layer 轴，TISA 优化 decode time 轴，二者可以组合：

```text
Full layer:
    不每步完整刷新所有 scores
    使用 TISA 分区刷新 score cache

Rerank layer:
    在 top-m 候选中重新打分
    也可以考虑对候选 scores 做局部 cache

Reuse layer:
    直接复用 top-k
```

组合后的直觉：

```text
IndexCache:
    少跑一些层的 indexer

TISA:
    每次跑 indexer 时少刷新一些 token
```

风险是近似叠加后精度可能下降，所以建议先单独验证，再组合。

---

## 4.5 Cross-layer GVR

GVR 原始用的是：

$
P_{l,t} = T_{l,t-1}
$

也就是当前层上一 decode step 的 top-k，作为 predictor。

我们提出可以尝试：

$
P_{l,t} = T_{l-1,t}
$

也就是上一层当前 step 的 top-k，作为 predictor。

这可以叫 **Cross-layer GVR**。

关键点是：

> **相邻层 top-k 不应该直接替代当前层 top-k；但可以作为 GVR 的 preIdx / predictor，用来猜 threshold。**

只要 GVR 仍然保留 verify/refine/fallback，它仍然可以保持 exact Top-K。

适合场景：

```text
1. first decode step，没有 previous-step top-k 时
2. 某些层 temporal correlation 弱，但 cross-layer correlation 强时
3. 和 IndexCache / Top-M IndexCache 结合
```


---

# 5. 我们提出的新方法总表

| 新方法                   | 核心思想                                       | 优点                       | 风险              | 推荐程度 |
| --------------------- | ------------------------------------------ | ------------------------ | --------------- | ---- |
| Top-M IndexCache      | Full 输出 top-m，Shared 在 top-m 内 rerank      | 精度远好于直接复用，成本远低于 full DSA | top-m 召回不足时仍会掉  | 最高   |
| Full/Rerank/Reuse 三类层 | 按层敏感度选择不同策略                                | 灵活，精度-速度可控               | 需要 calibration  | 很高   |
| 距离自适应 top-m           | 离 Full 越远，top-m 越大                         | 更合理控制成本                  | 参数更多            | 高    |
| IndexCache + TISA     | layer 复用 + time 复用                         | 进一步减少 score computation  | 近似误差叠加          | 中高   |
| Cross-layer GVR       | 用上一层 top-k 作为 GVR predictor                | exact，补充 temporal GVR    | 收益取决于层间相关性      | 中    |

---

# 6. 推荐工程落地路线

我建议按这个顺序落地：

## Step 1：Profile

先分清 DSA indexer pipeline 的耗时：

```text
score computation 占多少？
top-k selection 占多少？
KV gather / Sparse MLA 占多少？
```

如果 score computation 是主瓶颈，优先做 IndexCache / Top-M IndexCache / TISA。
如果 top-k selection 已经明显占比高，再做 GVR-like 优化。

---

## Step 2：实现基础 IndexCache

先做最简单版本：

```text
F S F S
或
F S S F
```

统计每层 top-k overlap 和任务分数变化。

---

## Step 3：实现 Top-M IndexCache

对掉点明显的 Shared layers，不直接复用 top-k，改成：

```text
Full:
    top-m = 8192

Shared:
    在 top-m 内 rerank 到 top-k = 2048
```

这是最值得优先尝试的新方法。

---

## Step 4：引入 Full/Rerank/Reuse 三类层

根据 calibration，把层分成：

```text
敏感层：Full
中等层：Rerank
稳定层：Reuse
```

这样可以更好地平衡速度和精度。

---

## Step 5：加 TISA-lite

对 Full/Rerank 层的 decode 阶段加 score cache：

```text
K = 2 / 4 / 8
默认先试 K = 4
```

每步只刷新部分 scores。

---

## Step 6：如果 Top-K selector 成为瓶颈，再做 GVR

GVR 不建议第一步做，但如果 Top-K 占比高，它是很好的 exact 优化。

推荐位置：

```text
Full layer: L → top-m
TISA score_cache: full cache → top-k
```

不一定要用于：

```text
Rerank layer: 8192 → 2048
```

因为这个规模较小。

---

# 7. 最推荐的最终组合

我认为最实用的工程组合是：

```text
Full layer:
    完整 indexer 或 TISA 分区刷新
    输出 top-m，例如 8192
    当前层自己使用 top-k

Rerank shared layer:
    在 Full layer top-m 内
    用自己的 indexer score 重新选 top-k

Reuse shared layer:
    对非常稳定的层
    直接复用 Full layer top-k

Top-K selector:
    如果 L→top-m 成为瓶颈
    再引入 GVR-like exact top-k
```

最终形态：

```text
F R R F R S F R R
```

其中：

```text
F = Full indexer, optionally with TISA
R = top-m rerank
S = direct reuse
```

---

# 8. 最终结论

上面讨论的论文可以归纳为：

```text
IndexCache:
    layer 轴优化

HISA:
    token/block 轴优化

MISA:
    head 轴优化

TISA:
    decode time 轴优化

GVR:
    top-k selector 轴优化
```

我们在此基础上提出的新工程方案核心是：

> **不要让 IndexCache 的 Shared layer 盲目复用 top-k，而是让 Full layer 提供更大的 top-m 候选集，Shared layer 在候选集内用自己的 indexer 重新选 top-k。**

这就是 **Top-M IndexCache / Rerank-IndexCache**。

它的优势是非常清晰的：

```text
相比原始 IndexCache：
    精度更稳

相比完整 DSA：
    计算量低很多

相比 HISA/MISA：
    工程改动更小

相比 GVR：
    直接减少 score computation，而不是只优化 top-k selector
```


[1]: https://arxiv.org/abs/2603.12201?utm_source=chatgpt.com "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse"
[2]: https://arxiv.org/abs/2605.07363?utm_source=chatgpt.com "MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference"
[3]: https://openreview.net/forum?id=6kOo3YtMdu&utm_source=chatgpt.com "TISA: Exploiting Temporal Locality to Accelerate Indexing ..."
[4]: https://arxiv.org/abs/2604.22312?utm_source=chatgpt.com "Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation"
