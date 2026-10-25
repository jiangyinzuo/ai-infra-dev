"""插桩与运行时之间的轻量协议；探针热路径继续使用 tuple。"""

# 文件、限定函数名、起止行列、操作细节；不为每次探针分配位置对象。
Site = tuple[str, str, int, int, int, int, str]
OriginSpec = tuple[str, str, int, str]
