# Common Numeric Data Types

本文记录常见整数、浮点和低精度 AI 数值格式的 bit layout、数值范围、典型用途和主要取舍。

---

## 1. 整数格式

整数格式通常分为 signed integer 和 unsigned integer。

- signed integer 通常使用 two's complement 表示。
- unsigned integer 只表示非负数。
- N bit signed integer 的范围是 `[-2^(N-1), 2^(N-1)-1]`。
- N bit unsigned integer 的范围是 `[0, 2^N-1]`。

| 类型 | bit 数 | signed | 最小值 | 最大值 | 常见用途 |
| --- | ---: | --- | ---: | ---: | --- |
| `int32` | 32 | 是 | `-2,147,483,648` | `2,147,483,647` | 通用整数、索引、shape、CPU/GPU kernel 参数 |
| `uint32` | 32 | 否 | `0` | `4,294,967,295` | bit mask、hash、无符号计数器、地址/offset |
| `int16` | 16 | 是 | `-32,768` | `32,767` | 压缩存储、音频、部分量化中间表示 |
| `uint16` | 16 | 否 | `0` | `65,535` | token id、小范围计数、图像/深度数据 |
| `int8` | 8 | 是 | `-128` | `127` | 模型权重量化、activation 量化、低精度 GEMM |
| `uint8` | 8 | 否 | `0` | `255` | 图像像素、byte buffer、非负量化值 |

### Signed vs Unsigned

同样 bit 数下，signed 和 unsigned 的主要差别是是否把最高位解释为符号信息。

```text
int8:
  1000_0000 -> -128
  1111_1111 -> -1
  0111_1111 -> 127

uint8:
  1000_0000 -> 128
  1111_1111 -> 255
  0111_1111 -> 127
```

在模型量化里：

- `int8` 常用于 symmetric quantization，例如零点为 0。
- `uint8` 常用于 asymmetric quantization，例如用 zero_point 表示真实值 0。

---

## 2. 浮点格式基础

常见二进制浮点数通常写成：

```text
value = (-1)^sign * significand * 2^exponent
```

标准浮点格式一般由三部分组成：

```text
sign bit | exponent bits | fraction / mantissa bits
```

术语说明：

- `sign`：符号位，0 表示正，1 表示负。
- `exponent`：指数位，通常使用 bias 编码。
- `fraction` / `mantissa`：尾数位，决定有效数字精度。
- `implicit leading 1`：规范化二进制浮点数通常隐含最高位 `1`，不显式存储。
- `normal`：指数位既不是全 0 也不是全 1 的普通规范化数。
- `subnormal`：指数位全 0，用于表示比最小 normal 更小的数，精度更低。
- `Inf / NaN`：指数位全 1 时通常表示无穷或非数，但 FP8/FP4 的机器学习变体可能不完全遵循 IEEE 754。

---

## 3. 常规浮点格式

### float32 / FP32 / IEEE binary32

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 32 |
| layout | `1 sign + 8 exponent + 23 fraction` |
| exponent bias | 127 |
| 有效精度 | 24 bits precision，约 7 位十进制有效数字 |
| 最小 positive normal | `2^-126 ~= 1.175e-38` |
| 最大 finite | `~3.403e38` |
| 特殊值 | `+0/-0`, `+Inf/-Inf`, `NaN`, subnormal |

FP32 是最通用的训练和累加格式。即使权重或 activation 使用 FP16/BF16/FP8，很多 kernel 仍会使用 FP32 做 accumulation，以降低舍入误差。

### float16 / FP16 / IEEE binary16

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 16 |
| layout | `1 sign + 5 exponent + 10 fraction` |
| exponent bias | 15 |
| 有效精度 | 11 bits precision，约 3.3 位十进制有效数字 |
| 最小 positive normal | `2^-14 ~= 6.104e-5` |
| 最大 finite | `65504` |
| 特殊值 | `+0/-0`, `+Inf/-Inf`, `NaN`, subnormal |

FP16 的尾数比 BF16 多，局部精度更好；但指数位少，动态范围小，更容易 overflow / underflow。

### bfloat16 / BF16

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 16 |
| layout | `1 sign + 8 exponent + 7 fraction` |
| exponent bias | 127 |
| 有效精度 | 8 bits precision，约 2-3 位十进制有效数字 |
| 最小 positive normal | `2^-126 ~= 1.175e-38` |
| 最大 finite | `~3.39e38` |
| 特殊值 | 通常沿用 IEEE 风格的 `Inf`, `NaN`, subnormal |

BF16 可以看成截短尾数的 FP32：

```text
FP32: 1 sign + 8 exponent + 23 fraction
BF16: 1 sign + 8 exponent +  7 fraction
```

所以 BF16 的动态范围接近 FP32，但精度明显低于 FP16。训练大模型时，BF16 通常比 FP16 更稳定，因为它更不容易 overflow。

### FP16 vs BF16

| 对比项 | FP16 | BF16 |
| --- | --- | --- |
| 总 bit 数 | 16 | 16 |
| exponent bits | 5 | 8 |
| fraction bits | 10 | 7 |
| 动态范围 | 小 | 接近 FP32 |
| 有效精度 | 高于 BF16 | 低于 FP16 |
| 训练稳定性 | 更依赖 loss scaling | 通常更稳定 |
| 常见用途 | 推理、训练、activation/weight | 大模型训练、mixed precision |

---

## 4. FP8 格式

FP8 是 8 bit floating point 的统称。深度学习里常见两种格式：

- `E5M2`：5 bit exponent + 2 bit mantissa。
- `E4M3`：4 bit exponent + 3 bit mantissa。

这里的 `E` 表示 exponent bits，`M` 表示显式 mantissa / fraction bits。

### FP8 E5M2

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 8 |
| layout | `1 sign + 5 exponent + 2 fraction` |
| exponent bias | 15 |
| 最大 finite | `57344` |
| 最小 positive normal | `2^-14` |
| 最小 positive subnormal | `2^-16` |
| 特殊值 | IEEE-like，支持 `Inf` 和 `NaN` |

E5M2 的指数位更多，动态范围更大，但尾数只有 2 bit，精度较低。它常用于梯度等动态范围较大的张量。

### FP8 E4M3

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 8 |
| layout | `1 sign + 4 exponent + 3 fraction` |
| exponent bias | 通常为 7 |
| 最大 finite | 常见 finite-only 变体为 `448` |
| 最小 positive normal | `2^-6` |
| 最小 positive subnormal | `2^-9` |
| 特殊值 | 常见 AI 变体不表示 `Inf`，保留少量 NaN 编码 |

E4M3 的尾数比 E5M2 多，精度更好，但动态范围更小。它常用于 activation 和 weight。

### E5M2 vs E4M3

| 对比项 | FP8 E5M2 | FP8 E4M3 |
| --- | --- | --- |
| exponent bits | 5 | 4 |
| fraction bits | 2 | 3 |
| 动态范围 | 更大 | 更小 |
| 精度 | 更低 | 更高 |
| `Inf` 支持 | 通常支持 | 常见 AI 变体通常不支持 |
| 常见用途 | gradients、动态范围大的张量 | weights、activations |

### Ascend HiFloat8 / HiF8

HiFloat8（HiF8）是 Ascend / HiSilicon 提出的 8 bit tapered floating point 格式。它不是固定 `E5M2` 或 `E4M3`，而是引入 dot field，让 exponent bits 和 mantissa bits 随数值范围变化。

完整 256 个编码、计算公式和实际值见：[HiFloat8 / HiF8 Complete 256-Value Table](./hif8_values.md)。

HiF8 的整体结构：

```text
sign | dot field | exponent field | mantissa field
```

dot field 使用 prefix code，决定当前数值使用多少 exponent bits：

| Dot value | Dot code | exponent bits | mantissa bits | exponent range |
| ---: | --- | ---: | ---: | --- |
| D4 | `11` | 4 | 1 | `±[8, 15]` |
| D3 | `10` | 3 | 2 | `±[4, 7]` |
| D2 | `01` | 2 | 3 | `±[2, 3]` |
| D1 | `001` | 1 | 3 | `±1` |
| D0 | `0001` | 0 | 3 | `0` |
| DML | `0000` | 0 | 3 | denormal-like extended range |

可以把 HiF8 理解成一种 tapered precision：

```text
小/中等 exponent:
  mantissa 更多，精度更高。

大 exponent:
  mantissa 更少，动态范围更大。

DML:
  没有 exponent field，用 mantissa 编码更小的 denormal-like 数值。
```

HiF8 的 normal value：

```text
value = (-1)^sign * 2^E * 1.M
```

DML value：

```text
value = (-1)^sign * 2^(M - 23) * 1.0, M in [1, 7]
```

典型特征：

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 8 |
| 普通 exponent range | `[-15, 15]` |
| 加上 DML 后的 exponent range | `[-22, 15]` |
| 最大正 normal finite | `0b01101110 = 0x6E` |
| 最小正 normal | `0b01111110 = 0x7E = 2^-15` |
| 最大正 DML | `0b00000111 = 0x07 = 2^-16` |
| 最小正 DML | `0b00000001 = 0x01 = 2^-22` |

和固定 FP8 的差异：

- 相比 E4M3，HiF8 动态范围更大。
- 相比 E5M2，HiF8 在较常见的小/中等 exponent 区间保留更多 mantissa bits。
- HiF8 支持 `Inf` 和 `NaN`，但 zero 只有一个 bit pattern，不区分 `+0` 和 `-0`。

---

## 5. FP4 格式

FP4 是 4 bit floating point 的统称。深度学习中最常见的是 `E2M1`：

```text
FP4 E2M1 = 1 sign + 2 exponent + 1 mantissa
```

常见 finite FP4 E2M1 的可表示值可以理解成：

```text
0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
```

FP4 单独使用时动态范围和精度都非常有限，因此通常不会裸用，而是和 per-block scale 搭配，例如 MXFP4 或 NVFP4。

---

## 6. Microscaling / Block-Scaled Formats

Microscaling 和 block-scaled formats 的核心思想是：

```text
多个低精度元素共享一个 scale
真实值 = scale * 低精度元素值
```

这样可以用很少的额外 metadata 扩大动态范围。

### MX 格式的基本结构

OCP MX 格式通常把 32 个元素作为一个 block：

```text
block = shared_scale + 32 low_precision_values
```

常见 scale 是 `E8M0`：

```text
E8M0 = 8 bit exponent-only scale
decoded scale = power-of-two scale
```

E8M0 只有 exponent，没有 mantissa，所以解码后的 scale 是 2 的幂。硬件实现时可以更接近指数调整 / shift，而不是普通乘法。

---

## 7. MXFP8

MXFP8 是 microscaling FP8，常见形式是：

```text
32 个 FP8 values 共享 1 个 E8M0 scale
```

常见变体：

| 类型 | block 元素数 | 元素格式 | block scale | 每 block bit 数 |
| --- | ---: | --- | --- | ---: |
| `MXFP8 E5M2` | 32 | FP8 E5M2 | E8M0 | `8 + 32 * 8 = 264` |
| `MXFP8 E4M3` | 32 | FP8 E4M3 | E8M0 | `8 + 32 * 8 = 264` |

数值解释：

```text
real_value[j] = E8M0_scale * fp8_value[j]
```

特点：

- 相比裸 FP8，MXFP8 用 shared scale 改善 block 内动态范围匹配。
- scale 粒度通常是 32 个元素，所以对 block 内异常值比较敏感。
- 适合矩阵计算中按 tile / vector 分块量化。

---

## 8. MXFP4

MXFP4 是 microscaling FP4，常见形式是：

```text
32 个 FP4 E2M1 values 共享 1 个 E8M0 scale
```

| 类型 | block 元素数 | 元素格式 | block scale | 每 block bit 数 |
| --- | ---: | --- | --- | ---: |
| `MXFP4` | 32 | FP4 E2M1 | E8M0 | `8 + 32 * 4 = 136` |

数值解释：

```text
real_value[j] = E8M0_scale * fp4_e2m1_value[j]
```

特点：

- 存储效率高，平均约 `136 / 32 = 4.25 bits/value`。
- E8M0 scale 是 2 的幂，硬件实现简单。
- 每 32 个元素共享同一个 scale，block 内数值分布差异大时量化误差会变大。
- 常用于极低 bit 权重、activation 或训练/推理研究。

---

## 9. NVFP4

NVFP4 是 NVIDIA Blackwell 相关的 4 bit block-scaled FP 格式。它和 MXFP4 都使用 FP4 E2M1 元素，但 scale 设计不同。

常见结构：

```text
16 个 FP4 E2M1 values 共享 1 个 FP8 E4M3 block scale
多个 block 再共享一个更高层级的 FP32 tensor/global scale
```

| 类型 | block 元素数 | 元素格式 | block scale | 额外 scale | 每 block bit 数 |
| --- | ---: | --- | --- | --- | ---: |
| `NVFP4` | 16 | FP4 E2M1 | FP8 E4M3 | tensor/global FP32 scale | `8 + 16 * 4 = 72` |

数值解释可以抽象为：

```text
real_value[j] = global_scale * e4m3_block_scale * fp4_e2m1_value[j]
```

和 MXFP4 的主要区别：

| 对比项 | MXFP4 | NVFP4 |
| --- | --- | --- |
| block size | 32 values | 16 values |
| 元素格式 | FP4 E2M1 | FP4 E2M1 |
| block scale | E8M0，power-of-two | FP8 E4M3，带 mantissa |
| scale 表达能力 | 粗，只有 2 的幂 | 更细，有尾数 |
| 额外 scale | 通常不强调二级 global scale | 常见为 block scale + global scale |
| 平均存储 | `4.25 bits/value` | `72 / 16 = 4.5 bits/value`，另有 global scale metadata |

NVFP4 的动机是用更小 block 和更精细的 E4M3 scale 降低 FP4 量化误差。代价是 metadata 稍多，scale 也不再只是简单的 power-of-two。

---

## 10. Ascend HiFloat4 / HiF4

HiFloat4（HiF4）是 Ascend / HiSilicon 提出的 4 bit 低精度格式。它不是裸 FP4，而是 block floating point / hierarchical scaling 格式。

HiF4 的基本单元是一个 64 元素 block：

```text
64 * S1P2 values + 32 bit metadata
```

其中 metadata 是三层 scale：

```text
level-1: 1 * E6M2 shared scale
level-2: 8 * E1 micro-exponents
level-3: 16 * E1 micro-exponents
```

总存储开销：

```text
64 * 4 bits + 32 metadata bits = 288 bits
288 / 64 = 4.5 bits/value
```

S1P2 是 4 bit sign-magnitude mantissa：

```text
S1P2 = 1 sign + 1 integer bit + 2 fraction bits
```

典型 S1P2 可表示值：

```text
0, ±0.25, ±0.5, ±0.75, ±1, ±1.25, ±1.5, ±1.75
```

HiF4 的数值可以抽象为：

```text
real_value[j] =
  E6M2_level1_scale
  * 2^(E1_8[ceil(j / 8)] + E1_16[ceil(j / 4)])
  * S1P2_value[j]
```

其中：

- E6M2 是 block 级粗 scale。
- 两层 E1 micro-exponents 提供更细粒度的 power-of-two 局部缩放。
- S1P2 存储每个元素的 4 bit 有符号数值。

和 MXFP4 / NVFP4 的差异：

| 对比项 | MXFP4 | NVFP4 | HiF4 |
| --- | --- | --- | --- |
| 元素格式 | FP4 E2M1 | FP4 E2M1 | S1P2 |
| block size | 32 | 16 | 64 |
| scale 结构 | 1 层 E8M0 | E4M3 block scale + global scale | E6M2 + 两层 E1 micro-exponents |
| 平均存储 | 4.25 bits/value | 4.5 bits/value + global scale | 4.5 bits/value |
| 设计重点 | 简单 power-of-two scale | 小 block + 更细 block scale | 三层 hierarchical scaling |

---

## 11. 浮点特殊值 Bit Pattern

本节集中列出浮点和 block-scaled formats 中常见特殊值的实际 bit pattern。

说明：

- bit pattern 按 `sign | exponent | mantissa/fraction` 写出。
- `0x...` 是对应的十六进制编码。
- IEEE-like 格式中，`exponent = all 1s` 且 `mantissa != 0` 表示 NaN。
- 低精度 AI 格式有多种变体，尤其是 E4M3 / E2M1；这里记录深度学习中常见的 finite-only 变体。

### FP32 / IEEE binary32

Layout：

```text
1 sign + 8 exponent + 23 fraction
```

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 00000000 00000000000000000000000` | `0x00000000` |
| `-0` | `1 00000000 00000000000000000000000` | `0x80000000` |
| `+Inf` | `0 11111111 00000000000000000000000` | `0x7F800000` |
| `-Inf` | `1 11111111 00000000000000000000000` | `0xFF800000` |
| canonical quiet NaN | `0 11111111 10000000000000000000000` | `0x7FC00000` |

NaN 规则：

```text
exponent = 11111111
fraction != 0
```

所以 `0x7FC00000` 只是常见 canonical qNaN，不是唯一 NaN。

### FP16 / IEEE binary16

Layout：

```text
1 sign + 5 exponent + 10 fraction
```

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 00000 0000000000` | `0x0000` |
| `-0` | `1 00000 0000000000` | `0x8000` |
| `+Inf` | `0 11111 0000000000` | `0x7C00` |
| `-Inf` | `1 11111 0000000000` | `0xFC00` |
| canonical quiet NaN | `0 11111 1000000000` | `0x7E00` |

NaN 规则：

```text
exponent = 11111
fraction != 0
```

### BF16

Layout：

```text
1 sign + 8 exponent + 7 fraction
```

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 00000000 0000000` | `0x0000` |
| `-0` | `1 00000000 0000000` | `0x8000` |
| `+Inf` | `0 11111111 0000000` | `0x7F80` |
| `-Inf` | `1 11111111 0000000` | `0xFF80` |
| canonical quiet NaN | `0 11111111 1000000` | `0x7FC0` |

NaN 规则：

```text
exponent = 11111111
fraction != 0
```

### FP8 E5M2

Layout：

```text
1 sign + 5 exponent + 2 fraction
```

E5M2 基本沿用 IEEE-style special value 规则。

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 00000 00` | `0x00` |
| `-0` | `1 00000 00` | `0x80` |
| `+Inf` | `0 11111 00` | `0x7C` |
| `-Inf` | `1 11111 00` | `0xFC` |
| quiet NaN examples | `0 11111 10`, `0 11111 11` | `0x7E`, `0x7F` |
| negative quiet NaN examples | `1 11111 10`, `1 11111 11` | `0xFE`, `0xFF` |

NaN 规则：

```text
exponent = 11111
fraction != 00
```

因此：

```text
positive NaN encodings: 0x7D, 0x7E, 0x7F
negative NaN encodings: 0xFD, 0xFE, 0xFF
```

如果按 quiet/signaling 区分，fraction 最高位为 1 的编码通常视为 quiet NaN。

### FP8 E4M3 finite-only variant

Layout：

```text
1 sign + 4 exponent + 3 fraction
```

深度学习常用 E4M3 变体不表示 infinity，把更多编码留给 finite values；只有 `111` mantissa 的最大 exponent 编码保留给 NaN。

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 0000 000` | `0x00` |
| `-0` | `1 0000 000` | `0x80` |
| `+Inf` | 不支持 | 无 |
| `-Inf` | 不支持 | 无 |
| `+NaN` | `0 1111 111` | `0x7F` |
| `-NaN` | `1 1111 111` | `0xFF` |
| 最大正 finite | `0 1111 110` | `0x7E` |
| 最大负 finite | `1 1111 110` | `0xFE` |

这里 `0x7E` / `0xFE` 仍是 finite values，而不是 infinity。

### Ascend HiFloat8 / HiF8

HiF8 使用 variable-length dot field，所以特殊值不是简单的固定 exponent-all-ones 规则。

| 值 | bit pattern | hex |
| --- | --- | --- |
| zero | `00000000` | `0x00` |
| NaN | `10000000` | `0x80` |
| `+Inf` | `01101111` | `0x6F` |
| `-Inf` | `11101111` | `0xEF` |
| max positive normal finite | `01101110` | `0x6E` |
| min positive normal | `01111110` | `0x7E` |
| max positive DML | `00000111` | `0x07` |
| min positive DML | `00000001` | `0x01` |

HiF8 不区分 `+0` 和 `-0`：

```text
zero = 0x00
```

HiF8 的 DML 编码中，`M = 0` 的两个 bit patterns 被特殊解释为 zero 和 NaN：

```text
0x00 -> zero
0x80 -> NaN
```

HiF8 把最大绝对值 normal 编码保留给 infinity：

```text
0x6F -> +Inf
0xEF -> -Inf
```

### FP4 E2M1 finite-only variant

Layout：

```text
1 sign + 2 exponent + 1 mantissa
```

深度学习中常见 FP4 E2M1 变体通常不表示 Inf / NaN，4 bit 编码基本都用于 zero 或 finite values。

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 00 0` | `0x0` |
| `-0` / second zero encoding | `1 00 0` | `0x8` |
| `+Inf` | 不支持 | 无 |
| `-Inf` | 不支持 | 无 |
| `NaN` | 不支持 | 无 |
| 最大正 finite | `0 11 1` | `0x7` |
| 最大负 finite | `1 11 1` | `0xF` |

常见 E2M1 finite value table：

| hex | bit pattern | value |
| --- | --- | ---: |
| `0x0` | `0 00 0` | `0` |
| `0x1` | `0 00 1` | `0.5` |
| `0x2` | `0 01 0` | `1` |
| `0x3` | `0 01 1` | `1.5` |
| `0x4` | `0 10 0` | `2` |
| `0x5` | `0 10 1` | `3` |
| `0x6` | `0 11 0` | `4` |
| `0x7` | `0 11 1` | `6` |
| `0x8` | `1 00 0` | `-0` 或 zero |
| `0x9` | `1 00 1` | `-0.5` |
| `0xA` | `1 01 0` | `-1` |
| `0xB` | `1 01 1` | `-1.5` |
| `0xC` | `1 10 0` | `-2` |
| `0xD` | `1 10 1` | `-3` |
| `0xE` | `1 11 0` | `-4` |
| `0xF` | `1 11 1` | `-6` |

不同库可能把 `0x8` 显示为 `-0` 或直接规范化为 `0`，但它不是 Inf / NaN。

### E8M0 Scale

E8M0 常用作 MX formats 的 shared scale。

Layout：

```text
8 exponent bits + 0 mantissa bits
```

| 值 | bit pattern | hex |
| --- | --- | --- |
| 最小 power-of-two scale | `00000000` | `0x00` |
| scale = 1 | `01111111` | `0x7F` |
| 最大 finite scale | `11111110` | `0xFE` |
| NaN scale | `11111111` | `0xFF` |
| zero scale | 不支持 | 无 |
| Inf scale | 不支持 | 无 |

常见解码方式：

```text
scale = 2^(encoded_exponent - 127), encoded_exponent in [0, 254]
0xFF is reserved for NaN
```

所以：

```text
0x00 -> 2^-127
0x7F -> 2^0 = 1
0xFE -> 2^127
0xFF -> NaN
```

### MXFP8 Special Values

MXFP8 的一个 block 通常是：

```text
E8M0 scale + 32 FP8 values
```

特殊值来自两层：

| 来源 | 特殊值规则 |
| --- | --- |
| E8M0 scale | `0xFF` 表示 NaN scale；不支持 zero scale / Inf scale |
| FP8 E5M2 element | 使用 E5M2 的 `±0`, `±Inf`, `NaN` 编码 |
| FP8 E4M3 element | 使用 E4M3 finite-only 的 `±0`, `NaN` 编码，不支持 Inf |

数值上：

```text
real_value[j] = E8M0_scale * fp8_value[j]
```

因此：

- 如果 element 是 NaN，结果是 NaN。
- 如果 E8M0 scale 是 `0xFF`，这个 scale 本身是 NaN，通常表示该 block 的缩放值无效。
- `MXFP8 E5M2` 可以通过 element 表示 `±Inf`。
- `MXFP8 E4M3` 的 element 不表示 `±Inf`。

### MXFP4 Special Values

MXFP4 的一个 block 通常是：

```text
E8M0 scale + 32 FP4 E2M1 values
```

| 来源 | 特殊值规则 |
| --- | --- |
| E8M0 scale | `0xFF` 表示 NaN scale；不支持 zero scale / Inf scale |
| FP4 E2M1 element | finite-only，通常没有 Inf / NaN；`0x0` 和 `0x8` 是 zero encodings |

数值上：

```text
real_value[j] = E8M0_scale * fp4_e2m1_value[j]
```

因此，MXFP4 的 element 本身通常不会产生 Inf / NaN；只有 scale 是 NaN 或外部计算产生 NaN 时，结果才会变成 NaN。

### NVFP4 Special Values

NVFP4 常见结构是：

```text
global FP32 scale + FP8 E4M3 block scale + 16 FP4 E2M1 values
```

特殊值来自三层：

| 来源 | 特殊值规则 |
| --- | --- |
| global FP32 scale | 使用 FP32 的 `±0`, `±Inf`, `NaN` 编码，但有效量化张量通常使用 finite scale |
| FP8 E4M3 block scale | 常用作正 scale；E4M3 finite-only 不支持 Inf，`0x7F` 可表示 NaN |
| FP4 E2M1 element | finite-only，通常没有 Inf / NaN |

数值上：

```text
real_value[j] = global_scale * e4m3_block_scale * fp4_e2m1_value[j]
```

因此：

- 正常 NVFP4 数据中，global scale 和 block scale 应该是 finite。
- FP4 element 本身不表示 Inf / NaN。
- 如果 global FP32 scale 是 NaN / Inf，或 E4M3 block scale 是 NaN，解码结果会被这些特殊值污染。

### Ascend HiFloat4 / HiF4 Special Values

HiF4 的特殊值来自 scale metadata 和 S1P2 element 两层。

S1P2 element 是 finite sign-magnitude 数值：

| hex | bit pattern | value |
| --- | --- | ---: |
| `0x0` | `0 000` | `0` |
| `0x1` | `0 001` | `0.25` |
| `0x2` | `0 010` | `0.5` |
| `0x3` | `0 011` | `0.75` |
| `0x4` | `0 100` | `1` |
| `0x5` | `0 101` | `1.25` |
| `0x6` | `0 110` | `1.5` |
| `0x7` | `0 111` | `1.75` |
| `0x8` | `1 000` | `-0` 或 zero |
| `0x9` | `1 001` | `-0.25` |
| `0xA` | `1 010` | `-0.5` |
| `0xB` | `1 011` | `-0.75` |
| `0xC` | `1 100` | `-1` |
| `0xD` | `1 101` | `-1.25` |
| `0xE` | `1 110` | `-1.5` |
| `0xF` | `1 111` | `-1.75` |

S1P2 element 本身不表示 Inf / NaN。

HiF4 的 E6M2 level-1 scale：

| 项目 | bit pattern / 规则 | hex / 值 |
| --- | --- | --- |
| layout | unsigned `6 exponent + 2 mantissa` | no sign bit |
| exponent bias | 48 |  |
| exponent range | `[-48, 15]` |  |
| min finite scale | `000000_00` | `0x00 = 2^-48 * 1.00` |
| max finite scale | `111111_10` | `0xFE = 2^15 * 1.50` |
| NaN scale | `111111_11` | `0xFF` |
| zero scale | 不支持 | 无 |
| infinity scale | 不支持 | 无 |

两层 E1 micro-exponents 是 1 bit power-of-two 局部缩放因子，本身不表示 Inf / NaN。

---

## 12. 总览表

| 类型 | 总 bit / 元素 | layout / 结构 | 动态范围 | 精度 | 典型用途 |
| --- | ---: | --- | --- | --- | --- |
| `int32` | 32 | two's complement | 固定整数范围 | 精确整数 | 索引、shape、通用整数 |
| `uint32` | 32 | unsigned binary | 固定非负范围 | 精确整数 | mask、offset、hash |
| `int16` | 16 | two's complement | 小整数范围 | 精确整数 | 压缩存储、中间表示 |
| `uint16` | 16 | unsigned binary | 小非负整数范围 | 精确整数 | token id、图像/深度数据 |
| `int8` | 8 | two's complement | 很小 | 精确整数 | 对称量化 |
| `uint8` | 8 | unsigned binary | 很小非负 | 精确整数 | 图像、非对称量化 |
| `float32` | 32 | S1 E8 M23 | 很大 | 高 | 训练、累加、通用浮点 |
| `float16` | 16 | S1 E5 M10 | 中等 | 中 | mixed precision、推理 |
| `bf16` | 16 | S1 E8 M7 | 接近 FP32 | 低于 FP16 | 大模型训练 |
| `fp8 E5M2` | 8 | S1 E5 M2 | 大 | 低 | gradients、动态范围大张量 |
| `fp8 E4M3` | 8 | S1 E4 M3 | 中 | 高于 E5M2 | weights、activations |
| `HiF8` | 8 | sign + dot + variable exponent/mantissa | 大于 E4M3 | tapered precision | Ascend 训练低精度 |
| `fp4 E2M1` | 4 | S1 E2 M1 | 很小 | 很低 | 通常配合 scale 使用 |
| `MXFP8` | 8.25 avg | 32 FP8 + E8M0 scale | 取决于 block scale | FP8 级别 | block-scaled FP8 |
| `MXFP4` | 4.25 avg | 32 FP4 + E8M0 scale | 取决于 block scale | FP4 级别 | 极低 bit 量化 |
| `NVFP4` | 4.5 avg + global scale | 16 FP4 + E4M3 scale + global scale | 取决于两级 scale | FP4 级别，但 scale 更细 | Blackwell FP4 训练/推理 |
| `HiF4` | 4.5 avg | 64 S1P2 + E6M2 + E1 micro-exponents | 取决于三层 scale | FP4 级别，但局部 scale 更细 | Ascend FP4 训练/推理 |

---

## 13. 选择建议

```text
需要通用数值稳定性:
  float32

需要 16 bit 训练且更稳定:
  bf16

需要 16 bit 推理/训练且更高局部精度:
  float16

需要 8 bit 浮点:
  activation / weight 优先看 E4M3
  gradient / 动态范围大张量优先看 E5M2
  Ascend 训练场景可以关注 HiF8

需要更低存储和更高吞吐:
  fp4 + block scale
  即 MXFP4 / NVFP4 / HiF4 这类格式
```

简单经验：

- `exponent bits` 越多，动态范围越大。
- `mantissa bits` 越多，相对精度越高。
- `block scale` 可以扩大低 bit 格式的可用范围，但会引入 block 内共享 scale 的量化误差。
- block 越小，scale 越贴合局部分布，误差通常更低，但 metadata 占比更高。

---

## 参考资料

- [IEEE 754](https://standards.ieee.org/ieee/754/6210/)：binary32 / binary16 浮点格式。
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)：E4M3 / E5M2。
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)：MXFP8 / MXFP4 / E8M0 scale。
- [NVIDIA NVFP4 相关资料](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：FP4 E2M1 + E4M3 block scale + global scale。
- [Ascend HiFloat8 Format for Deep Learning](https://arxiv.org/html/2409.16626v1)：HiF8 tapered precision / dot field。
- [HiFloat4: A 4-bit Floating Point Format for Deep Learning Inference](https://arxiv.org/html/2602.11287)：HiF4 S1P2 + E6M2 + E1 micro-exponents。
- [Scaling FP4 Training to Trillion-Token LLMs](https://arxiv.org/pdf/2604.08826)：Ascend FP4/HiF4 large-scale training context。
