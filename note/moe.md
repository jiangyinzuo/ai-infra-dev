MoE的FFN中，虽然是降维，但代码中仍然叫“上投影”。这是历史遗留问题。稠密模型的中间维度都大于hidden state，但MoE中，单个FFN的中间维度小于hidden state，只有乘上选中的专家数量，维度才大于hidden state。
