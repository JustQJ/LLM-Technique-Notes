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

#### 1. HiF8 exponent field 的计算方式

HiF8 的 exponent field 不是 IEEE 754 使用的 unsigned exponent + bias。对 normal 区域 `D1` 到 `D4`，exponent field 的最高位表示 exponent 的正负，剩余位表示当前 dot 区间内的 offset。

对 `Dk`，其中 `k in [1, 4]`：

```text
exponent field = exponent_sign | exponent_offset
                 1 bit          k - 1 bits

exponent_magnitude = 2^(k - 1) + unsigned(exponent_offset)

exponent_sign = 0: E = +exponent_magnitude
exponent_sign = 1: E = -exponent_magnitude
```

写成公式：

```text
E = (-1)^exponent_sign * (
      2^(k - 1) + unsigned(exponent_offset)
    )
```

各 dot 区域的指数解释如下：

| Dot | bit layout（不含 value sign） | exponent 计算 | exponent range |
| --- | --- | --- | --- |
| D4 | `11 | e_sign e_offset[2:0] | M` | `E = (-1)^e_sign * (8 + offset)` | `±[8, 15]` |
| D3 | `10 | e_sign e_offset[1:0] | MM` | `E = (-1)^e_sign * (4 + offset)` | `±[4, 7]` |
| D2 | `01 | e_sign e_offset[0] | MMM` | `E = (-1)^e_sign * (2 + offset)` | `±[2, 3]` |
| D1 | `001 | e_sign | MMM` | `E = (-1)^e_sign` | `±1` |
| D0 | `0001 | MMM` | `E = 0` | `0` |

这里有两个独立的 sign：

```text
value sign:
  决定整个数值是正数还是负数。

exponent sign:
  决定 E 是正指数还是负指数；E < 0 表示绝对值小于 1，
  不代表整个数值为负数。
```

例如 D4 的 4 bit exponent field：

```text
0111:
  exponent_sign = 0
  exponent_offset = 111b = 7
  E = +(8 + 7) = +15

1111:
  exponent_sign = 1
  exponent_offset = 111b = 7
  E = -(8 + 7) = -15
```

所以 `0111` 表示 `E = +15`，而不是 `1111`。`1111` 表示 `E = -15`。

#### 2. D0 和 D1

D0 没有 exponent bits，指数固定为 `E = 0`：

```text
layout = sign | 0001 | mantissa[2:0]

value = (-1)^sign * 2^0 * (1 + M / 8)
```

D0 的正数取值是：

```text
1.000, 1.125, 1.250, 1.375,
1.500, 1.625, 1.750, 1.875
```

例如：

```text
0 | 0001 | 100 = 0x0C
value = 2^0 * (1 + 4/8) = 1.5
```

D1 有 1 个 exponent bit。因为指数 magnitude 固定为 1，这一位只需要选择 `E = +1` 或 `E = -1`：

```text
layout = sign | 001 | exponent_sign | mantissa[2:0]

exponent_sign = 0: E = +1
exponent_sign = 1: E = -1
```

D1 正指数侧的正数取值：

```text
2.0, 2.25, 2.5, 2.75,
3.0, 3.25, 3.5, 3.75
```

D1 负指数侧的正数取值：

```text
0.5, 0.5625, 0.625, 0.6875,
0.75, 0.8125, 0.875, 0.9375
```

例如：

```text
0 | 001 | 0 | 010 = 0x12
value = 2^1 * (1 + 2/8) = 2.5

0 | 001 | 1 | 100 = 0x1C
value = 2^-1 * (1 + 4/8) = 0.75
```

#### 3. HiF8 最大有限值和溢出

HiF8 在 `E = +15` 时只有 1 个 mantissa bit：

```text
0 | 11 | 0111 | 0 = 0x6E
value = 2^15 * 1.0 = 32768

0 | 11 | 0111 | 1 = 0x6F
nominal value = 2^15 * 1.5 = 49152
HiF8 interpretation = +Inf
```

因为 `0x6F` 被保留给 `+Inf`，所以 HiF8 的最大正有限值是 `32768`，而不是 `49152`。负数侧对应：

```text
0xEE = -32768
0xEF = -Inf
```

`40960` 是 `32768` 与 nominal `49152` 的舍入中点：

```text
(32768 + 49152) / 2 = 40960
```

HiFloat8 仓库的 CUDA / NPU kernel 因此采用：

```text
32768 <= abs(x) < 40960: quantize to sign(x) * 32768
abs(x) >= 40960:         quantize to sign(x) * Inf
abs(x) <= 2^-23:         quantize to zero
```

#### 4. HiF8 的伪量化公式

HiFloat8 仓库里的 HiF8 是逐元素格式，没有 per-tensor、per-channel 或 per-block shared scale。每个元素通过自身的 exponent 表示动态范围。

对处于 finite normal quantization path 的输入：

```text
a = abs(x)
E = floor(log2(a))

if E == -23:
    E = -22
```

根据指数范围选择 mantissa bits `p`：

```text
p = 3, if abs(E) <= 3
p = 2, if abs(E) <= 7
p = 1, if abs(E) <= 15
p = 0, otherwise
```

量化步长和 round-half-away-from-zero 伪量化公式是：

```text
step = 2^(E - p)

Q(x) = sign(x) * floor(abs(x) / step + 0.5) * step
```

这里的伪量化不会输出 packed 8-bit code，而是把输入映射到最近的 HiF8 可表示值，再使用输入的 `float32`、`float16` 或 `bfloat16` dtype 保存结果。

例如 `x = 5.3`：

```text
E = floor(log2(5.3)) = 2
p = 3
step = 2^(2 - 3) = 0.5

Q(5.3) = floor(5.3 / 0.5 + 0.5) * 0.5
       = 5.5
```

#### 5. 与 HiFloat8 kernel 边界一致的 PyTorch 实现

下面的实现包含 HiFloat8 CUDA kernel 使用的 zero、Inf 和最小指数修正规则：

```python
import torch


@torch.no_grad()
def quant_dequant_hif8(x: torch.Tensor) -> torch.Tensor:
    """Map floating-point values to the HiF8 representable-value set."""
    if not x.is_floating_point():
        raise TypeError("quant_dequant_hif8 expects a floating-point tensor")

    dtype_ori = x.dtype
    x_fp32 = x.to(torch.float32)
    abs_x = x_fp32.abs()

    underflow = abs_x <= 2.0**-23
    pos_overflow = x_fp32 >= 40960.0
    neg_overflow = x_fp32 <= -40960.0
    is_nan = torch.isnan(x_fp32)
    finite_path = ~(underflow | pos_overflow | neg_overflow | is_nan)

    # Non-finite-path lanes use 1 only to keep log2/division well-defined;
    # their results are overwritten by the boundary handling below.
    safe_abs_x = torch.where(finite_path, abs_x, torch.ones_like(abs_x))
    exponent = torch.floor(torch.log2(safe_abs_x))

    # Values immediately above 2^-23 use the DML minimum exponent E=-22.
    exponent = torch.where(
        exponent == -23,
        torch.full_like(exponent, -22),
        exponent,
    )

    abs_exponent = exponent.abs()
    mantissa_bits = torch.where(
        abs_exponent <= 3,
        torch.full_like(exponent, 3),
        torch.where(
            abs_exponent <= 7,
            torch.full_like(exponent, 2),
            torch.where(
                abs_exponent <= 15,
                torch.ones_like(exponent),
                torch.zeros_like(exponent),
            ),
        ),
    )

    step = torch.exp2(exponent - mantissa_bits)
    quantized_abs = torch.floor(safe_abs_x / step + 0.5) * step
    quantized = torch.sign(x_fp32) * quantized_abs

    # Start from the original values so NaN payloads remain NaN.
    out = x_fp32.clone()
    out = torch.where(finite_path, quantized, out)
    out = torch.where(underflow, torch.zeros_like(out), out)
    out = torch.where(pos_overflow, torch.full_like(out, torch.inf), out)
    out = torch.where(neg_overflow, torch.full_like(out, -torch.inf), out)
    return out.to(dtype_ori)
```

使用示例：

```python
x = torch.tensor([
    5.3,
    32768.0,
    35000.0,
    40959.0,
    40960.0,
])

y = quant_dequant_hif8(x)

# tensor([5.5000, 32768.0000, 32768.0000, 32768.0000, inf])
```

HiFloat8 仓库的调用入口是：

```python
from quant_cy import QType, quant_dequant_float

qtype = QType("hif8").dim(0)
y = quant_dequant_float(x, qtype)
```

默认情况下，CUDA / NPU tensor 会选择对应的 HiF8 kernel；`force_py=True` 会选择仓库中的 PyTorch fallback。当前 fallback 只实现了指数分区和舍入公式，没有显式实现 `2^-23` 下溢、`40960` 上溢等完整边界，因此需要严格模拟 kernel 时应使用上面的完整实现或仓库的 CUDA / NPU kernel。

训练时，仓库的 `QuantFunc.backward` 使用 straight-through estimator（STE）：

```text
forward:  y = HiF8_fake_quant(x)
backward: grad_x = grad_y
```

如果模型张量可能超过 HiF8 的固定范围，又不希望产生 `Inf`，需要在 HiF8 外部显式增加 scale：

```text
q = HiF8(x / scale)
x_dequant = q * scale
```

这属于 scaled HiF8 量化策略，不是当前 HiFloat8 仓库中裸 `hif8_quant` 的格式组成部分。

---

## 5. FP6 格式

FP6 是 6 bit floating point 的统称。OCP MX 规格和 Microsoft `microxcaling` 实现中常见两种裸 element：

```text
FP6 E2M3 = 1 sign + 2 exponent + 3 mantissa
FP6 E3M2 = 1 sign + 3 exponent + 2 mantissa
```

这里的 FP6 是 finite-only AI format：支持 zero、subnormal、normal finite values，但不保留 `Inf / NaN` 编码。转换溢出时通常 saturate 到最大 finite 值，过小值会 round / flush 到 subnormal 或 zero。

### FP6 E2M3

`E2M3` 的特点是 mantissa bits 更多，精度高于 E3M2，但动态范围更小。

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 6 |
| layout | `1 sign + 2 exponent + 3 mantissa` |
| exponent bias | 1 |
| normal exponent range | `[0, 2]` |
| 最小 positive subnormal | `2^-3 = 0.125` |
| 最大 positive subnormal | `7 * 2^-3 = 0.875` |
| 最小 positive normal | `1.0` |
| 最大 finite | `1.875 * 2^2 = 7.5` |
| 特殊值 | 不支持 Inf / NaN |

数值解释：

```text
E = 00:
  subnormal / zero
  value = (-1)^S * (M / 8) * 2^0

E != 00:
  normal
  value = (-1)^S * (1 + M / 8) * 2^(E_bits - bias)
```

这里 `E=00` 的 subnormal exponent 使用最小 normal exponent，也就是 `0`，所以最小正 subnormal 是 `1/8 = 0.125`。

### FP6 E3M2

`E3M2` 的特点是 exponent bits 更多，动态范围大于 E2M3，但 mantissa 精度更低。

| 项目 | 值 |
| --- | --- |
| 总 bit 数 | 6 |
| layout | `1 sign + 3 exponent + 2 mantissa` |
| exponent bias | 3 |
| normal exponent range | `[-2, 4]` |
| 最小 positive subnormal | `2^-4 = 0.0625` |
| 最大 positive subnormal | `3 * 2^-4 = 0.1875` |
| 最小 positive normal | `2^-2 = 0.25` |
| 最大 finite | `1.75 * 2^4 = 28` |
| 特殊值 | 不支持 Inf / NaN |

数值解释：

```text
E = 000:
  subnormal / zero
  value = (-1)^S * (M / 4) * 2^-2

E != 000:
  normal
  value = (-1)^S * (1 + M / 4) * 2^(E_bits - bias)
```

### FP6 E2M3 vs E3M2

| 对比项 | FP6 E2M3 | FP6 E3M2 |
| --- | --- | --- |
| layout | S1 E2 M3 | S1 E3 M2 |
| 最大 finite | `7.5` | `28` |
| 最小 positive normal | `1.0` | `0.25` |
| 最小 positive subnormal | `0.125` | `0.0625` |
| 精度 | 更高 | 更低 |
| 动态范围 | 更小 | 更大 |
| 适合场景 | 数值分布较集中、希望保留更多尾数 | 数值分布更宽、需要更大范围 |

FP6 单独使用时仍然偏低精度，实际训练/推理中更常与 block scale 搭配，例如 OCP MXFP6。

---

## 6. FP4 格式

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

## 7. Microscaling / Block-Scaled Formats

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

## 8. MXFP8

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

## 9. MXFP6

MXFP6 是 OCP MX 标准里的 block-scaled FP6。常见形式是：

```text
32 个 FP6 values 共享 1 个 E8M0 scale
```

常见变体：

| 类型 | block 元素数 | 元素格式 | block scale | 每 block bit 数 |
| --- | ---: | --- | --- | ---: |
| `MXFP6 E2M3` | 32 | FP6 E2M3 | E8M0 | `8 + 32 * 6 = 200` |
| `MXFP6 E3M2` | 32 | FP6 E3M2 | E8M0 | `8 + 32 * 6 = 200` |

平均存储开销：

```text
200 / 32 = 6.25 bits/value
```

数值解释：

```text
real_value[j] = E8M0_scale * fp6_value[j]
```

特点：

- 相比 MXFP8，MXFP6 存储和带宽更低，但 element 精度/范围更弱。
- 相比 MXFP4，MXFP6 element 自身有更高精度和更大范围，通常量化误差更小。
- `E2M3` 偏精度，`E3M2` 偏动态范围。
- E8M0 scale 是 power-of-two，因此每个 block 的缩放更像指数平移。
- block size 通常是 32，和 MXFP8 / MXFP4 一致。

OCP MX 的 scale 选择通常基于 block 内最大绝对值：

```text
shared_scale ~= power_of_two_scale(max(abs(block_values)) / max_fp6_value)
```

其中：

```text
max_fp6_value(E2M3) = 7.5
max_fp6_value(E3M2) = 28
```

规格重点约束的是编码语义和转换行为，具体 scale selection / rounding policy 可能由硬件或软件库实现决定。

---

## 10. MXFP4

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

### MXFP4 元素格式：FP4 E2M1 二进制与数值对应

MXFP4 的元素使用 FP4 E2M1 格式：

```text
FP4 E2M1 = 1 sign + 2 exponent + 1 mantissa
```

前面这个公式：

```text
value = (-1)^S × (1 + M / 2) × 2^(E - bias)
```

只适用于 **normal number**，也就是：

```text
E != 00
```

对于 E2M1 来说，`E=00` 是特殊区间，用来表示：

```text
0 和 subnormal 数
```

#### 1. normal 区间：E 不是 00

E2M1 中：

```text
S: 1 bit
E: 2 bit
M: 1 bit
bias = 1
```

当 `E != 00` 时，用 normal 公式：

```text
value = (-1)^S × (1 + M / 2) × 2^(E - 1)
```

例如正数：

| E    |  M | 计算              |     值 |
| ---- | -: | --------------- | ----: |
| `01` |  0 | `1.0 × 2^(1-1)` | `1.0` |
| `01` |  1 | `1.5 × 2^(1-1)` | `1.5` |
| `10` |  0 | `1.0 × 2^(2-1)` | `2.0` |
| `10` |  1 | `1.5 × 2^(2-1)` | `3.0` |
| `11` |  0 | `1.0 × 2^(3-1)` | `4.0` |
| `11` |  1 | `1.5 × 2^(3-1)` | `6.0` |

所以 normal 区间得到：

```text
1.0, 1.5, 2.0, 3.0, 4.0, 6.0
```

#### 2. special/subnormal 区间：E = 00

当：

```text
E = 00
```

不能再用：

```text
1 + M / 2
```

而是用 subnormal 逻辑：

```text
value = (-1)^S × (M / 2) × 2^(1 - bias)
```

E2M1 里 `bias = 1`，所以：

```text
2^(1 - bias) = 2^(1 - 1) = 1
```

因此：

```text
value = (-1)^S × (M / 2)
```

所以：

| E    |  M | 计算      |     值 |
| ---- | -: | ------- | ----: |
| `00` |  0 | `0 / 2` | `0.0` |
| `00` |  1 | `1 / 2` | `0.5` |

也就是：

```text
E=00, M=0 → 0
E=00, M=1 → 0.5
```

#### 3. 如果错误地用 normal 公式会怎样？

如果对 `E=00` 也硬套 normal 公式：

```text
value = (1 + M / 2) × 2^(E - 1)
```

那么：

##### `E=00, M=0`

```text
(1 + 0/2) × 2^(0-1)
= 1 × 0.5
= 0.5
```

但真实应该是：

```text
0.0
```

##### `E=00, M=1`

```text
(1 + 1/2) × 2^(0-1)
= 1.5 × 0.5
= 0.75
```

但真实应该是：

```text
0.5
```

所以 `E=00` 必须特殊处理。

#### 4. E2M1 完整正数表

| bits   | E    |  M | 类型        |     值 |
| ------ | ---- | -: | --------- | ----: |
| `0000` | `00` |  0 | zero      | `0.0` |
| `0001` | `00` |  1 | subnormal | `0.5` |
| `0010` | `01` |  0 | normal    | `1.0` |
| `0011` | `01` |  1 | normal    | `1.5` |
| `0100` | `10` |  0 | normal    | `2.0` |
| `0101` | `10` |  1 | normal    | `3.0` |
| `0110` | `11` |  0 | normal    | `4.0` |
| `0111` | `11` |  1 | normal    | `6.0` |

加上符号位后，E2M1 的完整可表示值集合为：

```text
0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
```

#### 5. 对应到量化代码

典型 MXFP4 量化代码没有真的编码 `E` 和 `M`，而是在数值上模拟这个集合：

```python
private_exp = torch.floor(torch.log2(x.abs().clamp(min=_MXFP4_EPSILON))).clamp(min=0.0)
```

把小于 `1` 的数也强制放到 `private_exp = 0` 这个区间。

然后：

```python
x = x * 2
x = round(x)
x = x * 0.5
```

就可以产生：

```text
0 和 0.5
```

例如：

```text
x = 0.2
x * 2 = 0.4
round(0.4) = 0
0 * 0.5 = 0.0
```

```text
x = 0.3
x * 2 = 0.6
round(0.6) = 1
1 * 0.5 = 0.5
```

所以是的：**`0` 和 `0.5` 是 E2M1 的 zero/subnormal 特殊值，不走 normal 浮点公式。**

#### 6. MXFP4 伪量化参考实现（PyTorch）

下面给出完整的 MXFP4 pseudo-quantization 参考实现，对每行代码添加详细注释，帮助理解量化 → 反量化的完整流程：

```python
# ============================================================
# MXFP4 常量定义
# ============================================================

# E2M1 格式的 exponent bits 数（2 bit exponent）
_MXFP4_EBITS = 2

# E2M1 格式的 mantissa bits 数（1 bit mantissa）+ 1 bit implicit leading
# 这里 _MXFP4_MBITS = 3 是量化算法内部使用的"有效精度"参数，
# 实际 E2M1 只有 1 bit 显式 mantissa，但量化时用 3 bit 精度做中间计算
_MXFP4_MBITS = 3

# E2M1 的最大 exponent 值（2 bit exponent 能表示的最大值是 3，
# 但 E=11 是最大 normal 区间，_EMAX=2 用于 scale 计算中的归一化因子）
_MXFP4_EMAX = 2

# E2M1 能表示的最大 finite 值：
# S=0, E=11, M=1 → (1 + 1/2) × 2^(3-1) = 1.5 × 4 = 6.0
_MXFP4_MAX_NORM = 6.0

# MXFP4 block 大小：32 个元素共享一个 E8M0 scale
_MXFP4_BLOCK_SIZE = 32

# private_exp 的下限，确保小于 1 的数被强制归入 E=00 区间
# 这样它们会走 subnormal/zero 逻辑，产生 0 或 0.5
_MXFP4_MIN_EXP = 0.0

# 量化时的缩放因子：将数值放大 2 倍后取整
# 目的是让 [1, 2) 区间的数映射到 [2, 4)，round 后得到 2 或 3
# 再乘以 INV_SCALE_FACTOR 就得到 1.0 或 1.5
_MXFP4_SCALE_FACTOR = 2.0

# 反量化时的缩放因子：取整后缩小 2 倍，恢复为 E2M1 可表示值
_MXFP4_INV_SCALE_FACTOR = 0.5

# 防止 log2(0) 出现 -inf 的最小值，接近 FP32 最小 positive subnormal
_MXFP4_EPSILON = 1.17e-38

# E8M0 scale 的最大 exponent 值（8 bit unsigned，bias=127，最大 127）
_E8M0_SCALE_EMAX = 127


def _mxfp4_quant_tf(x, qdim, stochastic_rounding=False):
    """MXFP4-C7 quantization → dequantized (recovered) tensor.

    实现参考 triton to_mxfp4c7(p_cx=7)：
    shared_exp = ceil(log2(max_val / 7))。
    其中 p_cx=7 是 MXFP4 的归一化常数，用于确定 shared exponent。

    参数：
      x: 输入张量（待量化）
      qdim: 量化维度，block 沿此维度划分
      stochastic_rounding: 是否使用随机舍入（训练时减少偏差）

    返回：
      recovered: 反量化后的张量（shape 与输入相同）
    """
    # 保存输入张量的维度数，用于后续处理负数维度索引
    ndim = x.ndim

    # 保存原始 shape，最后需要 reshape 回去
    orig_shape = x.shape

    # 将负数维度索引转换为正数索引，便于后续计算
    # 例如 qdim=-1 在 3D 张量中变为 2
    normalized_qdim = qdim if qdim >= 0 else ndim + qdim

    # reduction_dim 是用于计算 block 内 max 的维度
    # unflatten 后新增的 block 内维度，需要在此维度上求 amax
    reduction_dim = normalized_qdim + 1

    # 将 qdim 维度拆分为 (-1, BLOCK_SIZE)，形成 block 结构
    # 例如 shape (M, K) 且 qdim=-1 → (M, K//32, 32)
    x = x.unflatten(qdim, (-1, _MXFP4_BLOCK_SIZE))

    # ============================================================
    # 第一步：计算 shared exponent（E8M0 scale）
    # ============================================================

    # 求 block 内每个元素的绝对值的最大值
    # shape: (..., num_blocks, 1)，keepdim=True 便于广播
    max_val = torch.amax(x.abs(), reduction_dim, keepdim=True)

    # MXFP4-C7 的归一化常数：将 max_val 除以 7 后再取 log2
    # 除以 7 是为了留出 headroom，确保量化后的值不会溢出 E2M1 的表示范围
    inv_constant = 1 / 7

    # 计算 shared exponent：
    # shared_exp = ceil(log2(max_val / 7))
    # 这样 scale = 2^shared_exp 能保证 block 内最大值的量化结果 ≤ E2M1_MAX
    shared_exp = torch.ceil(torch.log2(max_val.clamp(min=_MXFP4_EPSILON) * inv_constant))

    # 将 shared exponent 限制在 E8M0 的表示范围内 [-127, 127]
    # E8M0 使用 8 bit unsigned，bias=127，有效范围 [0, 254] → [-127, 127]
    shared_exp = shared_exp.clamp(-127, 127)

    # ============================================================
    # 第二步：用 shared exponent 归一化，将数值缩放到 E2M1 可表示范围
    # ============================================================

    # 除以 scale（即乘以 2^(-shared_exp)），使 block 内数值落入 [-6, 6]
    # 这是 E2M1 能表示的 finite 范围
    x = x * torch.exp2(-shared_exp)

    # ============================================================
    # 第三步：计算 private exponent（模拟 E2M1 的 exponent 编码）
    # ============================================================

    # 对每个元素计算其"私有" exponent：
    # floor(log2(|x|)) 得到该元素在 E2M1 中应有的 exponent E
    # clamp(min=0) 确保小于 1 的数被强制归入 E=00 区间
    # 这些数会走 subnormal/zero 逻辑，最终量化为 0 或 0.5
    private_exp = torch.floor(torch.log2(x.abs().clamp(min=_MXFP4_EPSILON))).clamp(min=_MXFP4_MIN_EXP)

    # ============================================================
    # 第四步：去除 exponent，只保留 mantissa 部分，然后取整
    # ============================================================

    # 除以 2^private_exp，将数值归一化到 [1, 2) 区间（normal 数）
    # 或 [0, 1) 区间（subnormal/zero）
    # 然后乘以 SCALE_FACTOR=2，将 [1, 2) 映射到 [2, 4)
    # 乘以 SCALE_FACTOR 是为了将尾数第一位提到整数，避免round的时候消除掉
    x = x * torch.exp2(-private_exp) * _MXFP4_SCALE_FACTOR

    if stochastic_rounding:
        # 随机舍入：加上 [0, 1) 均匀分布随机数后向下取整
        # 期望上无偏，训练时减少量化偏差累积
        x.add_(torch.rand_like(x)).floor_()
    else:
        # 确定性舍入（round-to-nearest）：
        # 1. 取符号位，后续对绝对值操作
        x_sign = torch.sign(x)
        # 2. 对绝对值加 0.5 后向下取整，实现四舍五入
        #    例如 2.3 + 0.5 = 2.8 → floor = 2
        #    例如 2.6 + 0.5 = 3.1 → floor = 3
        x = x_sign * torch.floor_(x.abs() + 0.5)

    # ============================================================
    # 第五步：反量化，恢复为 E2M1 可表示的数值
    # ============================================================

    # 1. 乘以 INV_SCALE_FACTOR=0.5，将取整后的值映射回 E2M1 的 mantissa
    #    例如 round 后为 2 → 2 * 0.5 = 1.0
    #    例如 round 后为 3 → 3 * 0.5 = 1.5
    # 2. 乘以 2^private_exp，恢复 exponent
    # 3. clamp 确保结果在 E2M1 的 finite 范围 [-6, 6] 内
    x = (x * _MXFP4_INV_SCALE_FACTOR * torch.exp2(private_exp)).clamp(-_MXFP4_MAX_NORM, _MXFP4_MAX_NORM)

    # ============================================================
    # 第六步：乘以 shared scale，恢复原始数值量级
    # ============================================================

    # 乘以 2^shared_exp，将量化后的值恢复到原始数值范围
    recovered = x * torch.exp2(shared_exp)

    # 将 shape 恢复为原始输入 shape
    return recovered.reshape(orig_shape)
```

**量化流程总结：**

```text
原始值 x
  │
  ├─ 1. 按 block 求 max_val
  ├─ 2. 计算 shared_exp = ceil(log2(max_val / 7))
  ├─ 3. x / 2^shared_exp → 归一化到 [-6, 6]
  │
  ├─ 4. 计算 private_exp = floor(log2(|x|)), min=0
  ├─ 5. x / 2^private_exp × 2 → 归一化到 [2, 4) 或 [0, 2)
  ├─ 6. round → 整数
  ├─ 7. × 0.5 × 2^private_exp → E2M1 可表示值
  │
  └─ 8. × 2^shared_exp → 恢复原始量级
```

**关键理解：**

- `private_exp = 0` 对应 E2M1 的 `E=00` 区间，产生 `0` 或 `0.5`
- `private_exp = 1` 对应 `E=01`，产生 `1.0` 或 `1.5`
- `private_exp = 2` 对应 `E=10`，产生 `2.0` 或 `3.0`
- `private_exp = 3` 对应 `E=11`，产生 `4.0` 或 `6.0`

---

## 11. NVFP4

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

## 12. Ascend HiFloat4 / HiF4

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

### HiF4 S1P2 的解释

`S1P2` 不是 `E1M2` 浮点格式，而是 sign-magnitude 定点 mantissa：

```text
S1P2 = sign | integer_bit | fraction_bit_1 fraction_bit_0
```

后 3 bit 表示 `I.F1F0_2`，所以可以把它看成：

```text
magnitude_code = unsigned(I F1 F0)
S1P2_value = (-1)^sign * magnitude_code / 4
```

可表示范围是：

```text
S1P2_value in [-1.75, 1.75]
step = 0.25
```

也就是说：

```text
0b0_000 -> +0
0b0_001 -> +0.25
0b0_100 -> +1.00
0b0_111 -> +1.75
0b1_111 -> -1.75
```

这里除以 `4` 是因为有 2 个 fraction bits，步长是 `2^-2 = 0.25`。如果按 `E1M2` 来解释，中间那一位会变成 exponent，语义就变成 `sign | exponent | mantissa`，这不是 HiF4 的 S1P2 element。

### HiF4 可表示范围

对固定的 block scale `S = E6M2_level1_scale`：

```text
E1_8 in {0, 1} 二阶scale
E1_16 in {0, 1} 三阶scale
2^(E1_8 + E1_16) in {1, 2, 4}
```

因此：

```text
value range for fixed S = [-1.75 * 4 * S, 1.75 * 4 * S]
                        = [-7S, 7S]

min positive for fixed S = 0.25 * S
```

如果按 HiF4 的 E6M2 scale 规则估算：

```text
E6M2 exponent bits = 6
E6M2 mantissa bits = 2
bias = 48

min finite scale = 1.00 * 2^-48
max finite scale = 1.50 * 2^15
```

则理论 finite 范围约为：

```text
max |value| = 7 * 1.50 * 2^15 = 344064
min positive = 0.25 * 2^-48 = 2^-50
```

### HiF4 量化 / 反量化参考流程

下面的代码是 HiF4 pseudo quant-dequant：它不真正 pack 成 4 bit，而是把输入张量按 HiF4 的 scale / mantissa 规则量化，再反量化回浮点值，用于模拟量化误差。

```python
def quant_dequant_hif4(x: torch.Tensor, quant_type: str = "hifx4", axe: int = -1):
    dtype_ori = x.dtype
    device = x.device
    C = x.shape[axe]
    blk_size_total = 64
    padC = (blk_size_total - C % blk_size_total) % blk_size_total

    pad_shape = list(x.shape)
    pad_shape[axe] = padC

    x_pad = torch.zeros(pad_shape, dtype=dtype_ori, device=device)
    x_padded = torch.cat([x, x_pad], dim=axe)

    total_C = C + padC
    arange_vec = torch.arange(total_C, device=device)
    mask_vector = (arange_vec < C).to(dtype_ori)

    unsqueezes = [None] * len(x_padded.shape)
    unsqueezes[axe] = slice(None)
    attention_mask = mask_vector[tuple(unsqueezes)]

    # 如果 C 不是 64 的倍数，应对 x_padded 做 kernel；
    # 如果直接传 x，则要求被量化维度本来就是 64 的倍数。
    qdq_out = _quantize_hif4_kernel(x_padded, axe)
    qdq_out *= attention_mask

    return qdq_out.narrow(axe, 0, C).to(dtype_ori)


def _quantize_hif4_kernel(x: torch.Tensor, qdim: int):
    # 64 channels -> [8, 2, 4]
    # level-1: 64 元素共享 scale_factor
    # level-2: 8 个 E1，每 8 元素一组
    # level-3: 16 个 E1，每 4 元素一组
    x = x.unflatten(qdim, (-1, 8, 2, 4))

    man_bits = 3
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    # Three-level max: innermost 4 / middle 8 / outer 64 channels.
    max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    max_lv2 = torch.max(max_lv3, dim=qdim - 1, keepdim=True)[0]
    max_lv1 = torch.max(max_lv2, dim=qdim - 2, keepdim=True)[0]

    # HiF4 最大局部倍率是 2 * 2，S1P2 最大 mantissa 是 1.75；
    # 1.75 * 2 * 2 = 7，因此 base scale 取 max / 7，就是把数据放缩到7以内
    div7 = torch.ones_like(max_lv1) / 7.0
    div7 = div7.to(torch.bfloat16).to(x.dtype)
    scale_factor = max_lv1 * div7

    # Round scale_factor to BF16 mantissa precision.
    # 这里是模拟bf16的7个尾数位，所以去掉指数后乘以2^7再round
    e_sf = torch.floor(torch.log2(scale_factor))
    mant_sf = scale_factor / 2 ** e_sf * 2 ** 7
    scale_factor = torch.round(mant_sf) / 2 ** 7 * 2 ** e_sf

    # Round scale_factor to E6M2-like grid: 2 mantissa bits.
    # 同样，模拟E6M2的2个尾数位
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)

    # 这里其实也是根据当前block的最大值，如果在E6M2的scale后，还能达到4，就取大的scale进行放缩，保证放缩到4以内，才能被下一步表示
    # L2 sub-block: scale_lv2 = 2 if max_lv2 >= 4 * scale_factor else 1.
    scale_lv2 = max_lv2 * rec_sf
    scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())

    # 这里就是第三次放缩，确定4个元素的最大值是否达到2
    # L3 sub-block: scale_lv3 = 2 if max_lv3 >= 2 * scale_factor * scale_lv2 else 1.
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())

    # S1P2 mantissa quantization: step = 0.25, max = 1.75.
    # 处理尾数，以0.25的步长投射到对应的数据
    mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
    mant = torch.floor(mant * 2 ** (man_bits - 1) + 0.5) / 2 ** (man_bits - 1)
    upper_bound = 2 - 2 ** (-man_bits + 1)
    mant = torch.clamp_max(mant, max=upper_bound)

    # Dequantize.
    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor
    out = out.flatten(qdim - 3, qdim)
    return out
```

上面 wrapper 中有一个容易踩坑的点：如果先 padding 到 64 的倍数，那么 kernel 应该处理 `x_padded`，反量化后再 narrow 回原始长度；如果直接调用 `_quantize_hif4_kernel(x, -1)`，则要求最后一维本来就是 64 的倍数。

### `scale_factor` 如何近似到 E6M2

代码中先计算：

```python
scale_factor = max_lv1 / 7.0
```

这个 `scale_factor` 是每 64 个元素共享的粗 scale。随后有两次 rounding。

第一次模拟 BF16 mantissa rounding：

```python
e_sf = torch.floor(torch.log2(scale_factor))
mant_sf = scale_factor / 2 ** e_sf * 2 ** 7
scale_factor = torch.round(mant_sf) / 2 ** 7 * 2 ** e_sf
```

设 `x = scale_factor`：

```text
e = floor(log2(x))
x = m * 2^e, m in [1, 2)
m_q = round(m * 128) / 128
x_q = m_q * 2^e
```

`2^7 = 128` 对应 BF16 的 7 个 fraction bits。这个步骤只模拟 mantissa 精度，没有完整模拟 BF16 的 subnormal / Inf / NaN。

第二次把 scale 近似到 E6M2 的 2-bit mantissa 网格：

```python
e_sf = torch.floor(torch.log2(scale_factor))
scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)
```

等价于：

```text
e = floor(log2(x))
m = x / 2^e
m_q = round(m * 4) / 4
x_q = m_q * 2^e
```

E6M2 的 significand 网格是：

```text
1.00, 1.25, 1.50, 1.75
```

如果真正编码成 E6M2，常见规则是：

```text
exponent_bits = e + 48
mantissa_bits = round((m_q - 1) * 4)
```

实际实现还需要 clamp 到 E6M2 的 exponent 范围，并处理 `0xFF` NaN 等特殊编码；这段 pseudo quant-dequant 只返回量化后的浮点 scale 数值，不返回 bit pattern。

### 两层 E1 micro-exponent 怎么算

计算两层局部 scale 的代码是：

```python
rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)

scale_lv2 = max_lv2 * rec_sf
scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())

scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())
```

`scale_lv2` 只有 `{1, 2}` 两种值。令：

```text
r2 = max_lv2 / scale_factor
```

则：

```text
scale_lv2 = 2^floor(clip(r2, 0, 4) / 4)

r2 < 4  -> scale_lv2 = 1
r2 >= 4 -> scale_lv2 = 2
```

`scale_lv3` 也只有 `{1, 2}` 两种值。令：

```text
r3 = max_lv3 / (scale_factor * scale_lv2)
```

则：

```text
scale_lv3 = 2^floor(clip(r3, 0, 2) / 2)

r3 < 2  -> scale_lv3 = 1
r3 >= 2 -> scale_lv3 = 2
```

最终：

```text
out = sign * mant * scale_factor * scale_lv2 * scale_lv3
```

其中：

```text
mant in {0, 0.25, 0.5, ..., 1.75}
scale_lv2 * scale_lv3 in {1, 2, 4}
```

### 为什么阈值是 `4` 和 `2`，不是 `3.5` 和 `1.75`

`1.75` 和 `3.5` 是 S1P2 在某个 scale 下的饱和边界：

```text
total_scale = 1 -> max representable = 1.75 * scale_factor
total_scale = 2 -> max representable = 3.5  * scale_factor
total_scale = 4 -> max representable = 7.0  * scale_factor
```

但代码里的 E1 micro-exponent 不是按 mantissa 饱和边界切档，而是按 power-of-two exponent binade 切档：

```text
max / scale_factor < 2      -> total_scale = 1
2 <= max / scale_factor < 4 -> total_scale = 2
4 <= max / scale_factor     -> total_scale = 4
```

因此阈值自然是 `2` 和 `4`。它的取舍是：

```text
用 1.75 / 3.5 做阈值:
  更早打开更大的 scale
  clipping 更少
  但量化步长更早变大，小值精度更差

用 2 / 4 做阈值:
  按二进制指数档位切换
  保留更细量化步长更久
  允许 [1.75, 2) 或 [3.5, 4) 区间出现少量饱和
```

例如 `scale_factor = 1` 且局部最大值为 `1.8`：

```text
如果阈值是 1.75:
  total_scale = 2
  mantissa step = 0.25 * 2 = 0.5

如果阈值是 2:
  total_scale = 1
  mantissa step = 0.25
  最大值 1.8 会被 clamp 到 1.75，但其他小值精度更好
```

所以这段实现选择的是：按 E1 exponent 的二进制档位边界切换，而不是按 S1P2 mantissa 最大值边界切换。

和 MXFP4 / NVFP4 的差异：

| 对比项 | MXFP4 | NVFP4 | HiF4 |
| --- | --- | --- | --- |
| 元素格式 | FP4 E2M1 | FP4 E2M1 | S1P2 |
| block size | 32 | 16 | 64 |
| scale 结构 | 1 层 E8M0 | E4M3 block scale + global scale | E6M2 + 两层 E1 micro-exponents |
| 平均存储 | 4.25 bits/value | 4.5 bits/value + global scale | 4.5 bits/value |
| 设计重点 | 简单 power-of-two scale | 小 block + 更细 block scale | 三层 hierarchical scaling |

---

## 13. 浮点特殊值 Bit Pattern

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

### FP6 finite-only variants

FP6 E2M3 / E3M2 在 OCP MX 语义中是 finite-only element，不表示 Inf / NaN。

#### FP6 E2M3

Layout：

```text
1 sign + 2 exponent + 3 mantissa
```

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 00 000` | `0x00` |
| `-0` / second zero encoding | `1 00 000` | `0x20` |
| `+Inf` | 不支持 | 无 |
| `-Inf` | 不支持 | 无 |
| `NaN` | 不支持 | 无 |
| 最小正 subnormal | `0 00 001` | `0x01 = 0.125` |
| 最大正 subnormal | `0 00 111` | `0x07 = 0.875` |
| 最小正 normal | `0 01 000` | `0x08 = 1.0` |
| 最大正 finite | `0 11 111` | `0x1F = 7.5` |
| 最大负 finite | `1 11 111` | `0x3F = -7.5` |

#### FP6 E3M2

Layout：

```text
1 sign + 3 exponent + 2 mantissa
```

| 值 | bit pattern | hex |
| --- | --- | --- |
| `+0` | `0 000 00` | `0x00` |
| `-0` / second zero encoding | `1 000 00` | `0x20` |
| `+Inf` | 不支持 | 无 |
| `-Inf` | 不支持 | 无 |
| `NaN` | 不支持 | 无 |
| 最小正 subnormal | `0 000 01` | `0x01 = 0.0625` |
| 最大正 subnormal | `0 000 11` | `0x03 = 0.1875` |
| 最小正 normal | `0 001 00` | `0x04 = 0.25` |
| 最大正 finite | `0 111 11` | `0x1F = 28` |
| 最大负 finite | `1 111 11` | `0x3F = -28` |

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

### MXFP6 Special Values

MXFP6 的一个 block 通常是：

```text
E8M0 scale + 32 FP6 values
```

特殊值来自两层：

| 来源 | 特殊值规则 |
| --- | --- |
| E8M0 scale | `0xFF` 表示 NaN scale；不支持 zero scale / Inf scale |
| FP6 E2M3 element | finite-only，不支持 Inf / NaN；`0x00` 和 `0x20` 是 zero encodings |
| FP6 E3M2 element | finite-only，不支持 Inf / NaN；`0x00` 和 `0x20` 是 zero encodings |

数值上：

```text
real_value[j] = E8M0_scale * fp6_value[j]
```

因此，MXFP6 的 element 本身不会产生 Inf / NaN；只有 E8M0 scale 是 NaN 或外部计算产生 NaN 时，结果才会变成 NaN。

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

## 14. 总览表

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
| `fp6 E2M3` | 6 | S1 E2 M3 | 小于 E3M2 | 高于 E3M2 | FP6 element，偏精度 |
| `fp6 E3M2` | 6 | S1 E3 M2 | 大于 E2M3 | 低于 E2M3 | FP6 element，偏范围 |
| `fp4 E2M1` | 4 | S1 E2 M1 | 很小 | 很低 | 通常配合 scale 使用 |
| `MXFP8` | 8.25 avg | 32 FP8 + E8M0 scale | 取决于 block scale | FP8 级别 | block-scaled FP8 |
| `MXFP6` | 6.25 avg | 32 FP6 + E8M0 scale | 取决于 block scale | FP6 级别 | OCP block-scaled FP6 |
| `MXFP4` | 4.25 avg | 32 FP4 + E8M0 scale | 取决于 block scale | FP4 级别 | 极低 bit 量化 |
| `NVFP4` | 4.5 avg + global scale | 16 FP4 + E4M3 scale + global scale | 取决于两级 scale | FP4 级别，但 scale 更细 | Blackwell FP4 训练/推理 |
| `HiF4` | 4.5 avg | 64 S1P2 + E6M2 + E1 micro-exponents | 取决于三层 scale | FP4 级别，但局部 scale 更细 | Ascend FP4 训练/推理 |

---

## 15. 选择建议

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

需要 6 bit 浮点:
  数值分布较集中时看 FP6 E2M3 / MXFP6 E2M3
  数值范围更宽时看 FP6 E3M2 / MXFP6 E3M2

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
- [OCP Microscaling Formats MX v1.0 Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：MXFP8 / MXFP6 / MXFP4 / E8M0 scale。
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)：MX formats 背景和 block scaling 思路。
- [Microsoft microxcaling](https://github.com/microsoft/microxcaling)：MXFP6 E2M3 / E3M2 等格式 emulation 参考实现。
- [NVIDIA NVFP4 相关资料](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：FP4 E2M1 + E4M3 block scale + global scale。
- [Ascend HiFloat8 Format for Deep Learning](https://arxiv.org/html/2409.16626v1)：HiF8 tapered precision / dot field。
- [Global Computing Consortium HiFloat8](https://github.com/global-computing-consortium/HiFloat8)：HiF8 CUDA / NPU kernel、PyTorch fallback 和伪量化测试代码。
- [HiFloat4: A 4-bit Floating Point Format for Deep Learning Inference](https://arxiv.org/html/2602.11287)：HiF4 S1P2 + E6M2 + E1 micro-exponents。
- [Scaling FP4 Training to Trillion-Token LLMs](https://arxiv.org/pdf/2604.08826)：Ascend FP4/HiF4 large-scale training context。
