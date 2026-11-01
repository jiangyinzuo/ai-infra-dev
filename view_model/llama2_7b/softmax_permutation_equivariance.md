# 逐行 softmax 的置换等变性证明
关键不是整个矩阵做 softmax，而是**逐行 softmax**。记

\[
S=\frac{QK^\top}{\sqrt{d_k}},\qquad A=\operatorname{softmax}(S),
\]

其中 softmax 对每一行归一化，即

\[
A_{i,j}=\frac{\exp(S_{i,j})}{\sum_{k}\exp(S_{i,k})}.
\]

要证的是

\[
\operatorname{softmax}(PSP^\top)=P\,\operatorname{softmax}(S)\,P^\top.
\]

---

## 1. 逐元素看

设 \(P\) 是置换矩阵，并约定

\[
(PX)_i=X_{\sigma(i)}.
\]

那么

\[
(PSP^\top)_{i,j}=S_{\sigma(i),\sigma(j)}.
\]

对 \(PSP^\top\) 的第 \(i\) 行做 softmax：

\[
\bigl[\operatorname{softmax}(PSP^\top)\bigr]_{i,j}
=
\frac{\exp(S_{\sigma(i),\sigma(j)})}
{\sum_k \exp(S_{\sigma(i),\sigma(k)})}.
\]

令 \(l=\sigma(k)\)。因为 \(\sigma\) 是置换，\(k\) 遍历所有列等价于 \(l\) 遍历所有列，所以分母变成

\[
\sum_l \exp(S_{\sigma(i),l}).
\]

因此

\[
\bigl[\operatorname{softmax}(PSP^\top)\bigr]_{i,j}
=
\frac{\exp(S_{\sigma(i),\sigma(j)})}
{\sum_l \exp(S_{\sigma(i),l})}
=
A_{\sigma(i),\sigma(j)}.
\]

而

\[
(PAP^\top)_{i,j}=A_{\sigma(i),\sigma(j)}.
\]

所以

\[
\operatorname{softmax}(PSP^\top)=PAP^\top
=P\operatorname{softmax}(S)P^\top.
\]

---

## 2. 直观理解

- 左乘 \(P\)：重排矩阵的**行**；
- 右乘 \(P^\top\)：重排每一行内部的**列**；
- softmax 是逐行独立做的，只依赖每行元素；
- 对一行向量 \(v\) 来说，把元素重新排列后再 softmax，等价于先 softmax 再同样重排：

\[
\operatorname{softmax}(vP^\top)=\operatorname{softmax}(v)P^\top.
\]

因此

\[
\operatorname{softmax}(PSP^\top)
=
P\operatorname{softmax}(S P^\top)
=
P\operatorname{softmax}(S)P^\top.
\]

---

## 3. 代回 attention

令

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
\]

若 token 顺序被 \(P\) 重排，则

\[
Q'=PQ,\quad K'=PK,\quad V'=PV.
\]

于是

\[
Q'K'^\top
=
PQK^\top P^\top.
\]

所以

\[
\operatorname{softmax}\left(\frac{Q'K'^\top}{\sqrt{d_k}}\right)
=
PAP^\top.
\]

输出为

\[
O'=
\operatorname{softmax}\left(\frac{Q'K'^\top}{\sqrt{d_k}}\right)V'
=
PAP^\top PV.
\]

因为 \(P\) 是置换矩阵，满足

\[
P^\top P=I,
\]

所以

\[
O'=PAV
=
P\,O.
\]

也就是

\[
\boxed{
\operatorname{softmax}\left(\frac{PQK^\top P^\top}{\sqrt{d_k}}\right)PV
=
P\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
}
\]

注意左边最后乘的是 \(PV\)，中间会多出一个 \(P^\top P=I\) 被消掉。

