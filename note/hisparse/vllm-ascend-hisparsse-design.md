主仓layout和ascend layout不一样：BLHNC LBHNC

hot和resident共享同一个block table。显存够用就全放resident。
hot -> resident -> host
- hot: LRU

- device放: indexer resident hot
核心算子
- resolver

算子可能可以优化。MTE算子？


