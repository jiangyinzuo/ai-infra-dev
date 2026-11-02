## 任意维张量，给定索引（slice）求内存地址偏移量的计算公式

设张量为 \(N\) 维，形状为

\[
\mathrm{shape}=(d_0,d_1,\dots,d_{N-1})
\]

步长为

\[
\mathrm{stride}=(s_0,s_1,\dots,s_{N-1})
\]

元素索引为

\[
(i_0,i_1,\dots,i_{N-1}),\quad 0\le i_k<d_k
\]

则从张量起始地址到该元素的内存偏移量为

\[
\mathrm{offset}=\sum_{k=0}^{N-1} i_k s_k
\]

即

\[
\mathrm{offset}=i_0s_0+i_1s_1+\cdots+i_{N-1}s_{N-1}
\]

若 `stride` 以元素个数为单位，则元素地址为

\[
\mathrm{address}=\mathrm{base}+\mathrm{offset}
\]

若 `stride` 以字节为单位，则元素地址为

\[
\mathrm{address}=\mathrm{base}+\mathrm{offset}
\]

若 `stride` 以元素个数为单位但地址按字节计，则

\[
\mathrm{address}=\mathrm{base}+\mathrm{offset}\times \mathrm{element\_size}
\]

对于行主序连续张量，步长满足

\[
s_{N-1}=1
\]

\[
s_k=s_{k+1}d_{k+1},\quad k=N-2,N-3,\dots,0
\]

因此任意维张量元素偏移量的通用公式为

\[
\boxed{\mathrm{offset}=\sum_{k=0}^{N-1} i_k s_k}
\]

## 行主序和列主序下给定shape求stride

还需要明确**stride 的单位**：元素个数还是字节。

设 \(N\) 维张量形状为

\[
\mathrm{shape}=(d_0,d_1,\dots,d_{N-1})
\]

### 行主序连续存储

最后一维变化最快：

\[
s_{N-1}=1
\]

\[
s_k=s_{k+1}d_{k+1},\quad k=N-2,N-3,\dots,0
\]

等价地：

\[
s_k=\prod_{j=k+1}^{N-1}d_j
\]

### 列主序连续存储

第一维变化最快：

\[
s_0=1
\]

\[
s_k=s_{k-1}d_{k-1},\quad k=1,2,\dots,N-1
\]

等价地：

\[
s_k=\prod_{j=0}^{k-1}d_j
\]

### 若 stride 以字节为单位

若元素大小为 `element_size` 字节，则：

\[
s_k^{\text{bytes}}=s_k\times \mathrm{element\_size}
\]

### 结论

给定 `shape`、连续存储布局和元素大小后，`stride` 可以完全计算出来：

\[
\boxed{\text{shape} + \text{布局约定} \Rightarrow \text{stride}}
\]

但**只给 `shape` 不能唯一确定 `stride`**，因为行主序和列主序会得到不同的 stride。
