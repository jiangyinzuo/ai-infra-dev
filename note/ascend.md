查看npu型号

只能看出是910，看不出具体型号
```
[root@node-97-30 ~]# npu-smi info
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.1.0                              Version: 26.1.0                                                      |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip  Phy-ID              | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     Ascend910           | OK            | 169.9                51                      0    / 0                |
| 0     0                   | 0000:9D:00.0  | 0                    0    / 0                3127 / 65536            |
+------------------------------------------------------------------------------------------------------------------+
```

0xD803 对应 Ascend 910C / Atlas A3，而 0xD802 对应 Ascend 910B
```
[root@node-97-30 ~]# npu-smi info -t board -i 0
        NPU ID                         : 0
        Product Name                   : NA
        Model                          : NA
        Manufacturer                   : Huawei
        Serial Number                  : NA
        Software Version               : 26.1.0
        Firmware Version               : 9.0.0.6.200
        Compatibility                  : OK
        Board ID                       : 0x70
        Main Board ID                  : 0x18
        PCB ID                         : B
        BOM ID                         : 2
        PCIe Bus Info                  : 0000:9D:00.0
        Slot ID                        : 0
        Class ID                       : NA
        PCI Vendor ID                  : 0x19E5
        PCI Device ID                  : 0xD803
        Subsystem Vendor ID            : 0x19E5
        Subsystem Device ID            : 0x3000
        Chip Count                     : 2
```
