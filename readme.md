# LLM Technique Notes

这个仓库用于整理大模型推理、长上下文注意力机制和 DSA 相关工程优化笔记。

## 文档目录

### [Sparse Attention Notes](./sparse_attention.md)

梳理 sparse attention / long-context attention 的三类结构，并重点说明 attention 数据流、张量 shape 变化、mask 或 sparse index 如何约束 token 可见性。

主要内容：

- 基础 DSA：介绍 DeepSeek-V3.2、GLM-5、GLM-5.1 中 MLA、Lightning Indexer 与 Sparse Attention 的主链路。
- GLM-5.2 DSA：说明 IndexShare / idxcache 如何在多层之间复用 top-k indices，降低 indexer 开销。
- DeepSeek V4 Attention：整理 sliding window KV、HCA/CSA compressed KV 和对应 mask 结构。
- 对比总结：对 dense attention、基础 DSA、GLM-5.2 DSA 和 DeepSeek V4 attention 的可见性与工程差异做横向比较。

### [DSA Lightning Indexer Optimization](./dsa_lightninng_indexer_optimization.md)

围绕 DSA indexer 的性能瓶颈，整理现有论文方法、优化轴和可落地的工程路线，重点关注如何减少 indexer score computation、降低 top-k selector 开销以及在速度和精度之间取舍。

主要内容：

- 背景分析：解释为什么 DSA 的主 attention 已经稀疏化后，indexer 仍可能成为长上下文推理瓶颈。
- 论文方法：总结 IndexCache、HISA、MISA、TISA、GVR Top-K 等方法的核心思路、收益和风险。
- 方法对比：按 layer、token/block、head、decode time、top-k selector 等优化轴进行对照。
- 进一步优化：提出 Top-M IndexCache、Full/Rerank/Reuse 三类层策略、距离自适应 top-m、IndexCache + TISA、Cross-layer GVR 等组合方案。
- 工程路线：给出从 profile、基础 IndexCache、Top-M IndexCache 到 TISA-lite 和 GVR 的分阶段落地建议。

## 主题索引

- DSA / DeepSeek Sparse Attention
- MLA / Multi-head Latent Attention
- Lightning Indexer
- Sparse Attention top-k selection
- IndexCache / IndexShare / idxcache
- HISA / MISA / TISA / GVR Top-K
- Sliding window KV / compressed KV
- Long-context inference optimization
