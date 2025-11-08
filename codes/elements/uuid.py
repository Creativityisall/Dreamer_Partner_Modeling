import string
import uuid as uuidlib

import numpy as np


class UUID:
    """
  自定义 UUID 类。UUID 存储为 16 字节的 bytes 字符串，并支持与 int,
  Base62 字符串, 和 NumPy 数组之间的相互转换。
  """

    # 优化内存使用，仅允许实例拥有这些属性
    __slots__ = ('value', '_hash')

    # 用于调试模式的计数器 ID
    DEBUG_ID = None

    # Base62 字符集: 0-9, a-z, A-Z (共 62 个字符)
    BASE62 = string.digits + string.ascii_letters
    # Base62 字符到其数值索引的映射字典 (用于解码 Base62 字符串)
    BASE62REV = {x: i for i, x in enumerate(BASE62)}

    @classmethod
    def reset(cls, *, debug):
        """
    重置或启用/禁用调试模式。
    如果 debug 为 True，则 DEBUG_ID 从 0 开始计数。
    """
        cls.DEBUG_ID = 0 if debug else None

    def __init__(self, value=None):
        # --- 1. 初始化 UUID 值 (self.value: 16字节 bytes) ---
        if value is None:
            # 如果未提供值
            if self.DEBUG_ID is None:
                # 正常模式：生成标准的随机 UUID (uuid4)
                self.value = uuidlib.uuid4().bytes
            else:
                # 调试模式：使用递增的整数作为 ID
                type(self).DEBUG_ID += 1
                # 将整数转换为 16 字节的大端序 bytes
                self.value = self.DEBUG_ID.to_bytes(16, 'big')

        elif isinstance(value, UUID):
            # 从另一个 UUID 实例复制
            self.value = value.value

        elif isinstance(value, int):
            # 从整数转换：转换为 16 字节大端序 bytes
            self.value = value.to_bytes(16, 'big')

        elif isinstance(value, bytes):
            # 从 bytes 转换：必须是 16 字节
            assert len(value) == 16, value
            self.value = value

        elif isinstance(value, str):
            # 从字符串转换
            if self.DEBUG_ID is None:
                # 正常模式：字符串被视为 Base62 编码的整数
                integer = 0
                # 从字符串末尾开始，进行 Base62 解码
                for index, char in enumerate(value[::-1]):
                    integer += (62 ** index) * self.BASE62REV[char]
                # 将解码后的整数转换为 16 字节 bytes
                self.value = integer.to_bytes(16, 'big')
            else:
                # 调试模式：字符串被视为十进制整数
                self.value = int(value).to_bytes(16, 'big')

        elif isinstance(value, np.ndarray):
            # 从 NumPy 数组转换：转换为 bytes
            self.value = value.tobytes()

        else:
            raise ValueError(value)

        # 最终检查，确保 self.value 是 16 字节的 bytes
        assert type(self.value) == bytes, type(self.value)  # noqa
        assert len(self.value) == 16, len(self.value)

        # 缓存哈希值，提高性能
        self._hash = hash(self.value)

    def __int__(self):
        """
    将 UUID 转换为大整数。
    """
        # 从 16 字节 bytes 转换为大端序整数
        return int.from_bytes(self.value, 'big')

    def __str__(self):
        """
    将 UUID 转换为字符串。调试模式下为十进制字符串，否则为 Base62 字符串。
    """
        if self.DEBUG_ID is not None:
            # 调试模式：直接返回整数的十进制字符串形式
            return str(int(self))

        # 正常模式：Base62 编码
        chars = []
        integer = int(self)  # 获取大整数值

        # 将整数转换为 Base62 编码 (类似于十进制转其他进制)
        while integer != 0:
            chars.append(self.BASE62[integer % 62])
            integer //= 62

        # 填充前导零：确保 Base62 字符串长度至少为 22 位，以保留所有信息
        while len(chars) < 22:
            chars.append('0')

        # Base62 编码结果是逆序的，需要反转
        return ''.join(chars[::-1])

    def __array__(self):
        """
    将 UUID 转换为 NumPy 数组 (16个 np.uint8 元素)。
    """
        return np.frombuffer(self.value, np.uint8)

    def __getitem__(self, index):
        """
    允许像访问数组一样访问 UUID 的字节。
    """
        return self.__array__()[index]

    def __repr__(self):
        """
    返回 UUID 的字符串表示（调用 __str__）。
    """
        return str(self)

    def __eq__(self, other):
        """
    比较两个 UUID 实例是否相等。
    """
        return self.value == other.value

    def __hash__(self):
        """
    返回缓存的哈希值。
    """
        return self._hash