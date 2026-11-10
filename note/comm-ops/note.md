博客教程
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devguide/hccl/hcclug/hcclug_000004.html
- https://github.com/devinjeon/gpu-comm-visualizer
- https://roboflow.com/blog/collective-communication-distributed-systems-pytorch#/#/#/#/#/search#1
- https://jax-ml.github.io/scaling-book/sharding/#a-deeper-dive-into-tpu-communication-primitives
- https://mlsysbook.ai/vol2/05_collective_communication/05_collective_communication.html

## AllReduce = ReduceScatter + AllGather

\[
\mathrm{AllReduce} \;=\; \mathrm{AllGather}\circ \mathrm{ReduceScatter}
\]

即先做 ReduceScatter，再做 AllGather，得到的结果与直接 AllReduce 等价。

---

### 数学上为什么成立

设有 \(p\) 个进程，进程 \(i\) 的输入向量为 \(x_i\)，规约算子为 \(\oplus\)（如求和、取最大值）。把向量按进程数分成 \(p\) 块：

\[
x_i=[x_i^0,x_i^1,\dots,x_i^{p-1}]
\]

**AllReduce 的定义**：每个进程都得到所有输入逐元素规约后的完整向量：

\[
y_i=\bigoplus_{j=0}^{p-1}x_j
=\left[\bigoplus_j x_j^0,\bigoplus_j x_j^1,\dots,\bigoplus_j x_j^{p-1}\right]
\]

对所有 \(i\) 相同。

**ReduceScatter 的输出**：进程 \(i\) 只保留第 \(i\) 块的规约结果：

\[
r_i=\bigoplus_{j=0}^{p-1}x_j^i
\]

**AllGather 的输出**：每个进程收集所有进程的 \(r_i\)：

\[
z_i=[r_0,r_1,\dots,r_{p-1}]
=\left[\bigoplus_j x_j^0,\bigoplus_j x_j^1,\dots,\bigoplus_j x_j^{p-1}\right]
\]

于是：

\[
z_i=\bigoplus_j x_j=y_i
\]

所以每个进程最终都得到完整规约向量，和 AllReduce 的输出一致。

---

### 成立需要满足的条件

1. **分块方式一致**：ReduceScatter 的第 \(i\) 块对应 AllGather 中进程 \(i\) 持有的块。
2. **规约算子满足结合律和交换律**：如 sum、max、min；否则顺序会影响结果。
3. **所有进程参与**，且结果布局约定一致。
4. **“+”应理解为顺序复合**，不是数学加法。写成：
   \[
   \mathrm{AllReduce}(x)=\mathrm{AllGather}(\mathrm{ReduceScatter}(x))
   \]

---

### 一个需要小心的点：浮点数

在浮点数求和时，由于加法不满足严格结合律，不同规约顺序可能带来微小数值差异。因此：

- **语义上/理想数学上**：严格成立。
- **浮点实现上**：结果可能因规约顺序不同而有舍入误差，不一定逐位相同。

---

### 通信代价上也“成立”

在环形算法中，ReduceScatter 和 AllGather 的通信量大致各占 AllReduce 的一半，所以常写：

\[
T_{\mathrm{AllReduce}}\approx T_{\mathrm{ReduceScatter}}+T_{\mathrm{AllGather}}
\]

这也是工程上常说“AllReduce = ReduceScatter + AllGather”的原因。

**结论：作为集合通信算子的复合恒等式，它在语义和数学定义上成立；但不能理解为普通数值加法，且浮点实现可能有舍入差异。**

## AllReduce = ReduceScatter + AllGather举例

用 4 个进程、每个进程持有一个长度为 4 的向量来举例。

设规约算子为 **sum**，4 个进程的输入为：

| 进程 | 输入向量 |
|---|---|
| P0 | \([1, 2, 3, 4]\) |
| P1 | \([10, 20, 30, 40]\) |
| P2 | \([100, 200, 300, 400]\) |
| P3 | \([1000, 2000, 3000, 4000]\) |

---

### 1. AllReduce(sum) 的结果

AllReduce 是**逐元素规约**，输出长度和输入长度相同：

\[
\begin{aligned}
&[1,2,3,4]\\
+\;&[10,20,30,40]\\
+\;&[100,200,300,400]\\
+\;&[1000,2000,3000,4000]\\
\hline
=&[1111,2222,3333,4444]
\end{aligned}
\]

所以 **4 个进程最终都得到**：

\[
\boxed{[1111,\;2222,\;3333,\;4444]}
\]

注意：不是把所有元素加成一个标量：

\[
1+2+3+4+10+\cdots+4000=11110
\]

那个 \(11110\) 是“所有元素总和”，不是对向量直接做 AllReduce 的默认结果。

---

### 2. 用 ReduceScatter + AllGather 分解

把每个进程的 4 元素向量分成 4 块，每块 1 个元素：

\[
P0:[1]\;[2]\;[3]\;[4]
\]

\[
P1:[10]\;[20]\;[30]\;[40]
\]

\[
P2:[100]\;[200]\;[300]\;[400]
\]

\[
P3:[1000]\;[2000]\;[3000]\;[4000]
\]

#### 第一步：ReduceScatter(sum)

每个进程只得到自己负责的那一块的规约结果：

| 进程 | ReduceScatter 后得到 |
|---|---|
| P0 | \(1+10+100+1000=1111\) |
| P1 | \(2+20+200+2000=2222\) |
| P2 | \(3+30+300+3000=3333\) |
| P3 | \(4+40+400+4000=4444\) |

即：

\[
P0\to 1111,\quad
P1\to 2222,\quad
P2\to 3333,\quad
P3\to 4444
\]

#### 第二步：AllGather

AllGather **不做规约**，只把上面 4 个结果收集并拼接起来。于是每个进程都得到：

\[
[1111,\;2222,\;3333,\;4444]
\]

正好等于直接做 AllReduce(sum) 的结果。

---

### 3. 关键对比

| 操作 | P0 得到 | P1 得到 | P2 得到 | P3 得到 |
|---|---|---|---|---|
| ReduceScatter | 1111 | 2222 | 3333 | 4444 |
| AllGather | \([1111,2222,3333,4444]\) | 同左 | 同左 | 同左 |
| AllReduce | \([1111,2222,3333,4444]\) | 同左 | 同左 | 同左 |

所以：

\[
\boxed{
\mathrm{AllReduce}
=
\mathrm{AllGather}\circ \mathrm{ReduceScatter}
}
\]

在这个例子里：

- **ReduceScatter 是“分块规约”**：每个进程只拿到一块结果。
- **AllGather 只负责收集拼接**，不带规约。
- **AllReduce 的语义是完整向量逐元素规约**，每个进程都拿到完整的 \([1111,2222,3333,4444]\)。

