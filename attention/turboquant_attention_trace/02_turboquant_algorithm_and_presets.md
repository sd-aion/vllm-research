# TurboQuant Algorithm And Presets

TurboQuant compresses the KV cache after attention has produced keys and values. Prefill attention itself operates on the original, uncompressed Q/K/V tensors; TurboQuant only changes what is retained for later decode or continuation prefill.

The implementation has two key modes. The FP8 mode casts keys directly to FP8. The MSE mode applies the full TurboQuant procedure: separate key magnitude from direction, rotate the direction with a Hadamard matrix, quantize each rotated coordinate to a Lloyd-Max centroid, and save the original magnitude. Values use affine 3-bit or 4-bit quantization in both modes.

The cache-write implementation is in `vllm/v1/attention/ops/triton_turboquant_store.py`. The decode and bulk-dequantization implementation is in `vllm/v1/attention/ops/triton_turboquant_decode.py`. Presets are defined in `vllm/model_executor/layers/quantization/turboquant/config.py`.

## 1. Cache Storage And Quantization

Consider one token and one local KV head. The attention layer supplies a key vector $\mathbf{k}\in\mathbb{R}^{D}$ and a value vector $\mathbf{v}\in\mathbb{R}^{D}$, where $D$ is the head dimension. The same procedure is independently applied to every token-head pair in the input tensors.

At the launcher level, `triton_turboquant_store(...)` receives keys and values with shape $N\times H_{\mathrm{KV}}\times D$. It flattens the token and head dimensions, producing $NH_{\mathrm{KV}}$ independent rows of length $D$.

### 1.1 MSE Key Quantization

The goal is to represent each full-precision key using:

1. one small integer index per coordinate, and
2. one fp16 scalar containing the key's original magnitude.

The following table is the complete store algorithm for one key.

| Step | Operation | Result | Why it is done |
| ---: | --- | --- | --- |
| 1 | Compute the key norm. | $r=\lVert\mathbf{k}\rVert_2$ | $r$ captures the magnitude of the original key in one scalar. |
| 2 | Normalize the key. | $\mathbf{x}=\mathbf{k}/(r+10^{-8})$ | $\mathbf{x}$ contains only the direction of the key and has approximately unit norm. |
| 3 | Apply the normalized Hadamard transform. | $\mathbf{y}=\mathbf{x}\mathbf{H}$ | The rotation spreads energy across coordinates while preserving norm and dot products. |
| 4 | Quantize each rotated coordinate. | $i_d=Q_{\mathrm{index}}(y_d)$ | Each coordinate becomes a 3-bit or 4-bit index into a shared centroid table. |
| 5 | Pack the indices. | $(i_0,\ldots,i_{D-1})$ becomes a compact byte sequence. | The integer indices are the compressed representation of the key direction. |
| 6 | Append the norm. | Store $r$ as fp16 after the packed indices. | Inference uses $r$ to restore the key's original magnitude. |

#### Step 1: Separate Magnitude From Direction

The Euclidean norm of the key is:

$$
r
=
\lVert\mathbf{k}\rVert_2
=
\sqrt{\sum_{d=0}^{D-1}k_d^2}
$$

The launcher then normalizes the key:

$$
x_d
=
\frac{k_d}{r+10^{-8}}
$$

This creates a unit-direction vector $\mathbf{x}$. The small $10^{-8}$ term prevents division by zero when the original key is zero or extremely small.

This step does not discard the original magnitude. The scalar $r$ is saved separately in the cache. Quantization is applied only to the normalized direction because a quantizer designed around unit vectors behaves more consistently than one that must also absorb arbitrary per-key magnitudes.

#### Step 2: Rotate The Direction

TurboQuant multiplies the normalized direction by a normalized Hadamard matrix $\mathbf{H}\in\mathbb{R}^{D\times D}$:

$$
\mathbf{y}
=
\mathbf{x}\mathbf{H}
$$

The matrix is orthonormal:

$$
\mathbf{H}\mathbf{H}^{\mathsf T}
=
\mathbf{I}
$$

Therefore:

$$
\lVert\mathbf{y}\rVert_2
=
\lVert\mathbf{x}\rVert_2
\approx
1
$$

The transform does not reduce precision by itself. Its purpose is to redistribute the key's energy. If a few coordinates of $\mathbf{x}$ are unusually large, the rotation spreads their contribution across many coordinates of $\mathbf{y}$. This makes scalar quantization less sensitive to outliers.

The implementation builds a Sylvester Hadamard matrix in `_build_hadamard(...)` from `vllm/v1/attention/backends/turboquant_attn.py`. This matrix is symmetric:

$$
\mathbf{H}^{\mathsf T}
=
\mathbf{H}
$$

Consequently, the same matrix performs the forward rotation and inverse rotation. The code stores it under both `Pi` and `PiT`, but they refer to the same mathematical matrix in this implementation.

#### Step 3: Quantize Rotated Coordinates With Lloyd-Max Centroids

Let $b_k$ be the key bit width. TurboQuant constructs a shared table containing:

$$
L
=
2^{b_k}
$$

centroids:

$$
\mathbf{C}
=
\begin{bmatrix}
C_0 & C_1 & \cdots & C_{L-1}
\end{bmatrix}
$$

A 3-bit key has $L=8$ centroids, while a 4-bit key has $L=16$ centroids. The centroid table is generated once for the head dimension and bit width; it is not stored separately for every key.

The centroids approximate coordinates drawn from:

$$
\mathcal{N}\left(0,\frac{1}{D}\right)
$$

This distribution is used because the rotated vector has approximately unit norm, so its average squared coordinate magnitude is approximately $1/D$.

Adjacent centroids define decision boundaries:

$$
M_j
=
\frac{C_j+C_{j+1}}{2},
\qquad
j\in\{0,\ldots,L-2\}
$$

For each rotated coordinate $y_d$, the store kernel binary-searches these boundaries and chooses an index $i_d$:

$$
i_d
=
Q_{\mathrm{index}}(y_d)
$$

The floating-point approximation represented by that index is:

$$
c_d
=
C_{i_d}
$$

The store path does not need to materialize $c_d$; it only stores $i_d$. The value $c_d$ is reconstructed later during inference by indexing the shared centroid table.

The key point is that one centroid table $\mathbf{C}$ is shared, while every cached key stores its own index vector:

$$
\mathbf{i}
=
\begin{bmatrix}
i_0 & i_1 & \cdots & i_{D-1}
\end{bmatrix}
$$

#### Step 4: Pack The Key Representation

For a 4-bit key, two centroid indices fit in one byte. For a 3-bit key, eight centroid indices occupy 24 bits, or three bytes.

The number of index bytes is:

$$
B_{\mathrm{indices}}
=
\left\lceil\frac{Db_k}{8}\right\rceil
$$

TurboQuant appends the original norm $r$ as fp16, adding two bytes:

$$
B_{\mathrm{MSE\ key}}
=
\left\lceil\frac{Db_k}{8}\right\rceil+2
$$

For $D=128$:

$$
B_{\mathrm{4\text{-}bit\ key}}
=
\frac{128\cdot4}{8}+2
=
66\text{ bytes}
$$

$$
B_{\mathrm{3\text{-}bit\ key}}
=
\frac{128\cdot3}{8}+2
=
50\text{ bytes}
$$

The stored key is therefore conceptually:

$$
\underbrace{\operatorname{pack}(i_0,\ldots,i_{D-1})}_{\text{quantized direction}}
\quad\Vert\quad
\underbrace{\operatorname{fp16}(r)}_{\text{original magnitude}}
$$

Here, $\Vert$ means byte concatenation, not a vector norm.

### 1.2 FP8 Key Storage

The `turboquant_k8v4` preset does not use normalization, Hadamard rotation, Lloyd-Max centroids, or a separately stored key norm. It directly casts each coordinate of $\mathbf{k}$ to FP8:

$$
\widehat{k}_d
=
\operatorname{FP8}(k_d)
$$

Since FP8 occupies one byte per coordinate:

$$
B_{\mathrm{FP8\ key}}
=
D
$$

The store and decode paths both call `_use_fp8_e4b15(...)` so that they agree on the FP8 byte interpretation. The implementation uses `float8e4b15` on CUDA-like devices below capability 8.9 and `float8e4nv` on Hopper-or-newer CUDA and non-CUDA platforms.

### 1.3 Value Quantization

Values do not use the Hadamard transform or Lloyd-Max centroids. Each value vector $\mathbf{v}$ receives its own affine quantization range.

The following table is the complete value-store algorithm for one value vector.

| Step | Operation | Result |
| ---: | --- | --- |
| 1 | Find the smallest and largest coordinate. | $v_{\min}$ and $v_{\max}$ |
| 2 | Compute the number of usable integer intervals. | $L_v=2^{b_v}-1$ |
| 3 | Compute the real-value step between adjacent integer levels. | $\Delta_v=\max((v_{\max}-v_{\min})/L_v,10^{-8})$ |
| 4 | Map each coordinate to an integer. | $j_d=\operatorname{clamp}(\operatorname{round}((v_d-v_{\min})/\Delta_v),0,L_v)$ |
| 5 | Pack the integer indices. | Compact 3-bit or 4-bit value data |
| 6 | Append quantization parameters. | fp16 $\Delta_v$ followed by fp16 $v_{\min}$ |

First:

$$
v_{\min}
=
\min_{0\leq d<D}v_d,
\qquad
v_{\max}
=
\max_{0\leq d<D}v_d
$$

For value bit width $b_v$, define:

$$
L_v
=
2^{b_v}-1
$$

The scale is:

$$
\Delta_v
=
\max\left(
\frac{v_{\max}-v_{\min}}{L_v},
10^{-8}
\right)
$$

Each coordinate becomes an integer index:

$$
j_d
=
\operatorname{clamp}
\left(
\operatorname{round}
\left(
\frac{v_d-v_{\min}}{\Delta_v}
\right),
0,
L_v
\right)
$$

The cache stores:

$$
\underbrace{\operatorname{pack}(j_0,\ldots,j_{D-1})}_{\text{quantized value coordinates}}
\quad\Vert\quad
\underbrace{\operatorname{fp16}(\Delta_v)}_{\text{scale}}
\quad\Vert\quad
\underbrace{\operatorname{fp16}(v_{\min})}_{\text{real-space offset}}
$$

The implementation calls $v_{\min}$ the value zero because integer index zero reconstructs to $v_{\min}$. It is an affine offset and is not necessarily numerically zero.

The value storage size is:

$$
B_{\mathrm{value}}
=
\left\lceil\frac{Db_v}{8}\right\rceil+4
$$

The extra four bytes contain the fp16 scale and fp16 minimum.

### 1.4 Complete Cache Entry

For an MSE-key preset, one token and one KV head occupy:

$$
\operatorname{pack}(\mathbf{i})
\;\Vert\;
\operatorname{fp16}(r)
\;\Vert\;
\operatorname{pack}(\mathbf{j})
\;\Vert\;
\operatorname{fp16}(\Delta_v)
\;\Vert\;
\operatorname{fp16}(v_{\min})
$$

For the FP8-key preset, the packed key indices and norm are replaced by $D$ FP8 key bytes.

The complete slot size is:

$$
B_{\mathrm{slot}}
=
B_{\mathrm{key}}+B_{\mathrm{value}}
$$

The available presets are:

| Preset | Key storage | Value storage | Norm correction at inference |
| --- | --- | --- | --- |
| `turboquant_k8v4` | FP8 coordinates | 4-bit affine | Not applicable |
| `turboquant_4bit_nc` | 4-bit centroid indices and fp16 norm | 4-bit affine | Enabled |
| `turboquant_k3v4_nc` | 3-bit centroid indices and fp16 norm | 4-bit affine | Enabled |
| `turboquant_3bit_nc` | 3-bit centroid indices and fp16 norm | 3-bit affine | Enabled |

For $D=128$, the per-token, per-KV-head sizes are:

| Preset | Key bytes | Value bytes | Total slot bytes |
| --- | ---: | ---: | ---: |
| `turboquant_k8v4` | 128 | 68 | 196 |
| `turboquant_4bit_nc` | 66 | 68 | 134 |
| `turboquant_k3v4_nc` | 50 | 68 | 118 |
| `turboquant_3bit_nc` | 50 | 52 | 102 |

## 2. Inference And Dequantization

At inference time, TurboQuant has two different read paths:

1. Decode computes attention directly from the compressed cache without reconstructing every key in the original coordinate system.
2. Large continuation prefill fully dequantizes cached K/V because FlashAttention or SDPA requires explicit key and value tensors.

These paths share the same interpretation of the stored bytes, but they use the reconstructed information differently.

### 2.1 Direct Decode From The Compressed Cache

Consider a query vector $\mathbf{q}\in\mathbb{R}^{D}$ and one cached MSE key. The cache contains the key's packed centroid indices $\mathbf{i}$, the original key norm $r$, the packed value indices $\mathbf{j}$, and the value parameters $\Delta_v$ and $v_{\min}$.

The following table is the direct-decode algorithm for one query/key pair.

| Step | Operation | Result | Implementation role |
| ---: | --- | --- | --- |
| 1 | Rotate the query. | $\mathbf{q}_{\mathrm{rot}}=\mathbf{q}\mathbf{H}$ | Places the query in the same coordinate system as the compressed key direction. |
| 2 | Unpack key indices. | $(i_0,\ldots,i_{D-1})$ | Recovers the centroid-table selections for this cached key. |
| 3 | Gather centroids. | $c_d=C_{i_d}$ | Reconstructs an approximate rotated key direction $\mathbf{c}$. |
| 4 | Optionally correct its norm. | $\widetilde{\mathbf{c}}=\mathbf{c}/\lVert\mathbf{c}\rVert_2$ | Removes radial distortion introduced by independent scalar quantization. |
| 5 | Compute the rotated-space dot product. | $t=\mathbf{q}_{\mathrm{rot}}\mathbf{d}^{\mathsf T}$ | Avoids inverse-rotating every cached key. |
| 6 | Restore original key magnitude and apply attention scale. | $s=\alpha rt$ | Produces the attention logit for this cached token. |
| 7 | Unpack and dequantize the value. | $\widehat{v}_d=j_d\Delta_v+v_{\min}$ | Produces the value used by online softmax accumulation. |

Here, $\alpha$ is the model's attention scale, normally $1/\sqrt{D}$, and $\mathbf{d}$ denotes the decoded direction actually used for scoring.

#### Reconstructing The Quantized Direction

The kernel unpacks each key index and gathers its centroid:

$$
c_d
=
C_{i_d},
\qquad
d\in\{0,\ldots,D-1\}
$$

This produces:

$$
\mathbf{c}
=
\begin{bmatrix}
c_0 & c_1 & \cdots & c_{D-1}
\end{bmatrix}
$$

The original rotated direction $\mathbf{y}$ had approximately unit norm. The reconstructed centroid vector $\mathbf{c}$ generally does not:

$$
\mathbf{c}
\neq
\mathbf{y}
$$

$$
\lVert\mathbf{c}\rVert_2
\neq
1
$$

These are two distinct errors. The first is direction error: the selected centroids do not reproduce every coordinate exactly. The second is radial error: independently quantizing the coordinates changes the vector's total length.

#### Why The Stored Norm Is Not Enough

The original key norm $r$ was saved during quantization. Without norm correction, inference would reconstruct the rotated key as:

$$
\widehat{\mathbf{k}}_{\mathrm{rot}}
=
r\mathbf{c}
$$

Its norm would be:

$$
\left\lVert
\widehat{\mathbf{k}}_{\mathrm{rot}}
\right\rVert_2
=
r\lVert\mathbf{c}\rVert_2
$$

The desired norm is $r$. Multiplication by $r$ restores the original norm only if $\lVert\mathbf{c}\rVert_2=1$. Because scalar centroid quantization does not enforce that constraint, simply multiplying by the stored norm leaves an additional multiplicative error of $\lVert\mathbf{c}\rVert_2$.

#### Norm Correction

Norm correction first measures the length of the reconstructed centroid vector:

$$
s_c
=
\sqrt{
\sum_{d=0}^{D-1}c_d^2
+10^{-16}
}
$$

It then projects that vector back onto the unit sphere:

$$
\widetilde{\mathbf{c}}
=
\frac{\mathbf{c}}{s_c}
$$

The rotated key approximation becomes:

$$
\widehat{\mathbf{k}}_{\mathrm{rot}}
=
r\widetilde{\mathbf{c}}
$$

Its norm is now approximately the saved original norm:

$$
\left\lVert
\widehat{\mathbf{k}}_{\mathrm{rot}}
\right\rVert_2
\approx
r
$$

Norm correction removes radial error, but it does not remove direction error. The direction $\widetilde{\mathbf{c}}$ is still only an approximation to $\mathbf{y}$.

The `_nc` suffix in a preset enables this operation during direct decode and bulk dequantization. It does not alter the bytes written by the store kernel.

#### Numerical Norm-Correction Example

Suppose the original key norm saved in the cache is $r=5$, and unpacking the centroid indices reconstructs:

$$
\mathbf{c}
=
\begin{bmatrix}
0.6 & 0.6 & 0.2 & 0.2
\end{bmatrix}
$$

The reconstructed direction has norm:

$$
\begin{aligned}
\lVert\mathbf{c}\rVert_2
&=
\sqrt{
0.6^2+0.6^2+0.2^2+0.2^2
} \\
&=
\sqrt{0.8} \\
&\approx
0.8944
\end{aligned}
$$

Without norm correction:

$$
r\mathbf{c}
=
\begin{bmatrix}
3 & 3 & 1 & 1
\end{bmatrix}
$$

and:

$$
\lVert r\mathbf{c}\rVert_2
\approx
4.472
$$

The saved norm was $5$, but the reconstructed key has norm $4.472$ because $\mathbf{c}$ was too short.

With norm correction:

$$
\widetilde{\mathbf{c}}
=
\frac{\mathbf{c}}{0.8944}
\approx
\begin{bmatrix}
0.6708 & 0.6708 & 0.2236 & 0.2236
\end{bmatrix}
$$

Therefore:

$$
r\widetilde{\mathbf{c}}
\approx
\begin{bmatrix}
3.354 & 3.354 & 1.118 & 1.118
\end{bmatrix}
$$

and:

$$
\left\lVert
r\widetilde{\mathbf{c}}
\right\rVert_2
\approx
5
$$

#### Computing The Attention Score In Rotated Space

Ordinary attention requires:

$$
s
=
\alpha
\mathbf{q}
\widehat{\mathbf{k}}^{\mathsf T}
$$

Direct decode does not inverse-rotate each cached key. Instead, it rotates the query once:

$$
\mathbf{q}_{\mathrm{rot}}
=
\mathbf{q}\mathbf{H}
$$

An orthonormal transform preserves dot products:

$$
\mathbf{q}
\widehat{\mathbf{k}}^{\mathsf T}
=
(\mathbf{q}\mathbf{H})
(\widehat{\mathbf{k}}\mathbf{H})^{\mathsf T}
$$

The centroid vector is already in the transformed coordinate system. Define:

$$
\mathbf{d}
=
\begin{cases}
\mathbf{c},
& \text{without norm correction}, \\
\widetilde{\mathbf{c}},
& \text{with norm correction}.
\end{cases}
$$

The decode kernel computes:

$$
t
=
\mathbf{q}_{\mathrm{rot}}
\mathbf{d}^{\mathsf T}
$$

and then:

$$
s
=
\alpha rt
$$

This is mathematically equivalent to scoring against the reconstructed key, but it avoids an inverse Hadamard transform for every cached token.

In `_tq_decode_stage1(...)`, the mathematical quantities map to implementation variables as follows:

| Algorithm quantity | Kernel variable |
| --- | --- |
| $\mathbf{q}_{\mathrm{rot}}$ | `q_rot` |
| gathered $\mathbf{c}$ for a KV tile | `c_vals` |
| $\lVert\mathbf{c}\rVert_2^2$ | `c_norm_sq` |
| $1/\lVert\mathbf{c}\rVert_2$ | `c_inv_norm` |
| transformed-space dot product $t$ | `term1` |
| stored key norm $r$ | `vec_norms` |
| attention scale $\alpha$ | `ATTN_SCALE` |
| attention logits $s$ | `scores` |

The kernel processes a tile of cached tokens simultaneously. For a tile of $\mathrm{BLOCK\_KV}$ tokens, `c_vals` has logical shape $\mathrm{BLOCK\_KV}\times D$. The kernel computes one reconstructed direction norm, one saved key norm, and one attention score for each token in that tile.

#### Dequantizing Values During Decode

The value side is direct:

$$
\widehat{v}_d
=
j_d\Delta_v+v_{\min}
$$

The kernel reconstructs the value coordinates for the current KV tile, applies the softmax probabilities, and accumulates the weighted values using online softmax. It does not need to materialize a full dequantized value cache.

### 2.2 FP8-Key Decode

For `turboquant_k8v4`, the query is not Hadamard-rotated because the keys were not rotated at store time. The kernel interprets each cached key byte as the selected FP8 format, converts it to float32 for accumulation, and computes:

$$
s
=
\alpha
\sum_{d=0}^{D-1}
q_d\widehat{k}_d
$$

There is no separately stored key norm and no norm-correction step. Values are still reconstructed from their 4-bit affine representation.

### 2.3 Full Dequantization For Continuation Prefill

Large continuation prefill must combine previously cached K/V with a new chunk of uncompressed K/V and invoke FlashAttention or SDPA. Those attention implementations expect explicit key and value tensors, so TurboQuant uses `_tq_full_dequant_kv(...)`.

The full MSE-key reconstruction algorithm is:

| Step | Operation | Result |
| ---: | --- | --- |
| 1 | Unpack centroid indices. | $\mathbf{i}$ |
| 2 | Gather centroid values. | $\mathbf{c}$ |
| 3 | Apply norm correction when enabled. | $\mathbf{d}=\mathbf{c}/\lVert\mathbf{c}\rVert_2$ or $\mathbf{d}=\mathbf{c}$ |
| 4 | Read and apply the saved key norm. | $\widehat{\mathbf{k}}_{\mathrm{rot}}=r\mathbf{d}$ |
| 5 | Apply the inverse Hadamard transform. | $\widehat{\mathbf{k}}=\widehat{\mathbf{k}}_{\mathrm{rot}}\mathbf{H}$ |
| 6 | Dequantize values. | $\widehat{v}_d=j_d\Delta_v+v_{\min}$ |

The full-dequantization Triton kernel performs steps 1 through 4 and writes the rotated key approximation:

$$
\widehat{\mathbf{k}}_{\mathrm{rot}}
=
r\mathbf{d}
$$

The continuation-prefill code in `TurboQuantAttentionImpl._continuation_prefill_dequant(...)` then applies:

$$
\widehat{\mathbf{k}}
=
\widehat{\mathbf{k}}_{\mathrm{rot}}\mathbf{H}
$$

Because the selected Hadamard matrix is symmetric and orthonormal:

$$
\mathbf{H}^{-1}
=
\mathbf{H}^{\mathsf T}
=
\mathbf{H}
$$

Thus, multiplying by $\mathbf{H}$ again returns the key approximation to the original coordinate system.

The backend concatenates the reconstructed cached K/V with the current chunk's uncompressed K/V. It then runs causal attention over the complete sequence using FlashAttention when available or PyTorch SDPA as a fallback.

### 2.4 Algorithm Initialization

`TurboQuantAttentionImpl._ensure_on_device(...)` prepares the shared algorithm data the first time a layer executes:

| Tensor | Purpose |
| --- | --- |
| `_tq_Pi` and `_tq_PiT` | Float32 normalized Hadamard matrix used for key and query rotation. |
| `_tq_Pi_half` | Float16 Hadamard matrix used for inverse rotation during continuation prefill. |
| `_tq_centroids` | Float32 Lloyd-Max reconstruction table $\mathbf{C}$. |
| `_tq_midpoints` | Float32 decision boundaries $\mathbf{M}$ used by the store kernel. |

`_build_hadamard_cached(D, device)` reuses the same matrix for equal head dimensions and devices. `get_centroids(D, bits)` similarly caches centroid generation results.
