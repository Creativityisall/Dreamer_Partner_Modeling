import io
import json
import re

from . import path


class Config(dict):
    """
  配置类，继承自 Python 字典，用于管理嵌套配置参数。
  支持点分隔符访问 (e.g., config.a.b) 和只读特性 (通过 update() 方法创建新配置)。
  支持 JSON 和 YAML 格式的保存与加载。
  """

    # 键分隔符，用于扁平化和嵌套操作
    SEP = '.'
    # 匹配任何包含非字母数字、下划线、点或连字符的键名。
    # 用于判断一个键是否被当作正则表达式模式处理。
    IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')

    def __init__(self, *args, **kwargs):
        # 将输入转换为一个标准字典
        mapping = dict(*args, **kwargs)
        # 1. 扁平化：将嵌套字典转换为扁平结构，键使用 SEP 分隔 (e.g., {'a.b': 1})
        mapping = self._flatten(mapping)
        # 2. 键校验：确保扁平键中没有使用正则表达式模式（模式只能用于 update 时）
        mapping = self._ensure_keys(mapping)
        # 3. 值校验：确保值类型安全和列表/元组内元素类型一致
        mapping = self._ensure_values(mapping)

        # 存储扁平化后的配置
        self._flat = mapping
        # 4. 嵌套：将扁平结构恢复为嵌套字典结构 (供 __getitem__ 和 dict() 使用)
        self._nested = self._nest(mapping)

        # 将嵌套字典赋值给基类 dict，确保 Config 实例作为字典行为正常
        super().__init__(self._nested)

    @property
    def flat(self):
        """返回扁平化配置的副本。"""
        return self._flat.copy()

    def save(self, filename):
        """将配置保存到文件，支持 .json 和 .yml/.yaml 格式。"""
        filename = path.Path(filename)
        if filename.suffix == '.json':
            # JSON 格式保存
            filename.write(json.dumps(dict(self)))
        elif filename.suffix in ('.yml', '.yaml'):
            # YAML 格式保存 (需要 ruamel.yaml 库)
            from ruamel.yaml import YAML
            yaml = YAML(typ='safe')
            # 使用 io.StringIO 捕获 YAML 输出，然后写入文件
            with io.StringIO() as stream:
                yaml.dump(dict(self), stream)
                filename.write(stream.getvalue())
        else:
            raise NotImplementedError(filename.suffix)

    @classmethod
    def load(cls, filename):
        """从文件加载配置，支持 .json 和 .yml/.yaml 格式，并返回 Config 实例。"""
        filename = path.Path(filename)
        if filename.suffix == '.json':
            return cls(json.loads(filename.read()))
        elif filename.suffix in ('.yml', '.yaml'):
            # YAML 格式加载
            from ruamel.yaml import YAML
            yaml = YAML(typ='safe')
            return cls(yaml.load(filename.read()))
        else:
            raise NotImplementedError(filename.suffix)

    def __contains__(self, name):
        """检查配置中是否包含某个键（支持嵌套键访问）。"""
        try:
            self[name]
            return True
        except KeyError:
            return False

    def __getattr__(self, name):
        """
    通过点号访问配置项 (e.g., config.key)。
    如果键不存在，则抛出 AttributeError。
    """
        if name.startswith('_'):
            return super().__getattr__(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name):
        """
    通过方括号访问配置项 (e.g., config['key'] 或 config['a.b'])。
    支持使用 SEP 分隔的嵌套键路径。
    """
        result = self._nested
        # 遍历键路径的各个部分
        for part in name.split(self.SEP):
            try:
                result = result[part]
            except TypeError:
                # 如果中间部分不是字典，则路径无效
                raise KeyError

        # 如果获取的结果仍然是字典，则将其包装成新的 Config 实例，以便继续点访问
        if isinstance(result, dict):
            result = type(self)(result)
        return result

    def __setattr__(self, key, value):
        """
    禁止通过点号直接设置属性，以保证配置的不可变性。
    必须使用 update() 方法创建新的 Config 实例。
    """
        if key.startswith('_'):
            return super().__setattr__(key, value)
        message = f"Tried to set key '{key}' on immutable config. Use update()."
        raise AttributeError(message)

    def __setitem__(self, key, value):
        """
    禁止通过方括号直接设置配置项，以保证配置的不可变性。
    必须使用 update() 方法创建新的 Config 实例。
    """
        if key.startswith('_'):
            return super().__setitem__(key, value)
        message = f"Tried to set key '{key}' on immutable config. Use update()."
        raise AttributeError(message)

    def __reduce__(self):
        """
    实现 pickle 协议，确保 Config 对象可以被序列化和反序列化。
    反序列化时，只需使用其字典表示进行初始化。
    """
        return (type(self), (dict(self),))

    def __str__(self):
        """
    格式化 Config 对象的字符串表示，显示扁平化键、值和类型。
    """
        lines = ['\nConfig:']
        keys, vals, typs = [], [], []
        # 收集扁平键、格式化值和格式化类型
        for key, val in self.flat.items():
            keys.append(key + ':')
            vals.append(self._format_value(val))
            typs.append(self._format_type(val))

        # 计算最大宽度，用于对齐
        max_key = max(len(k) for k in keys) if keys else 0
        max_val = max(len(v) for v in vals) if vals else 0

        # 格式化输出每一行
        for key, val, typ in zip(keys, vals, typs):
            key = key.ljust(max_key)
            val = val.ljust(max_val)
            lines.append(f'{key}  {val}  ({typ})')

        return '\n'.join(lines)

    def update(self, *args, **kwargs):
        """
    更新配置，返回一个新的 Config 实例，原实例不变。
    支持使用正则表达式模式匹配键进行批量更新。
    """
        result = self._flat.copy()
        inputs = self._flatten(dict(*args, **kwargs))

        for key, new in inputs.items():
            # 检查键是否是正则表达式模式
            if self.IS_PATTERN.match(key):
                pattern = re.compile(key)
                # 找到所有匹配当前模式的现有键
                keys = {k for k in result if pattern.match(k)}
            else:
                # 如果不是模式，则只更新该单个键
                keys = [key]

            if not keys:
                raise KeyError(f'Unknown key or pattern {key}.')

            for key in keys:
                if key in result:
                    old = result[key]
                    try:
                        # 尝试类型转换：将新值转换为旧值的类型 (保证类型一致性)
                        if isinstance(old, int) and isinstance(new, float):
                            # 特殊处理：不允许将带有小数部分的浮点数转换为整数
                            if float(int(new)) != new:
                                message = f"Cannot convert fractional float {new} to int."
                                raise ValueError(message)
                        result[key] = type(old)(new)
                    except (ValueError, TypeError):
                        # 捕获类型转换失败的错误
                        raise TypeError(
                            f"Cannot convert '{new}' to type '{type(old).__name__}' " +
                            f"for key '{key}' with previous value '{old}'.")
                else:
                    # 如果键是新的，直接添加
                    result[key] = new

        # 用更新后的扁平字典创建并返回新的 Config 实例
        return type(self)(result)

    def _flatten(self, mapping):
        """
    将嵌套字典递归地转换为扁平字典，键使用 SEP 分隔符连接。
    同时处理键中的正则表达式模式（如果存在）。
    """
        result = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                for k, v in self._flatten(value).items():
                    # 如果任一键包含模式字符，则用特殊的双反斜杠进行连接 (避免在模式匹配时产生歧义)
                    if self.IS_PATTERN.match(key) or self.IS_PATTERN.match(k):
                        combined = f'{key}\\{self.SEP}{k}'
                    else:
                        # 否则使用标准 SEP 连接
                        combined = f'{key}{self.SEP}{k}'
                    result[combined] = v
            else:
                result[key] = value
        return result

    def _nest(self, mapping):
        """
    将扁平字典恢复为嵌套字典结构。
    """
        result = {}
        for key, value in mapping.items():
            parts = key.split(self.SEP)
            node = result
            # 遍历键路径，创建必要的中间字典
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            # 在最深层设置值
            node[parts[-1]] = value
        return result

    def _ensure_keys(self, mapping):
        """
    初始化时，确保所有键名都不包含正则表达式模式。
    模式只允许在 update() 方法的输入键中使用。
    """
        for key in mapping:
            assert not self.IS_PATTERN.match(key), key
        return mapping

    def _ensure_values(self, mapping):
        """
    校验值类型：
    1. 确保值是 JSON 可序列化的基本类型。
    2. 将列表 (list) 转换为元组 (tuple)。
    3. 确保列表/元组非空且所有元素类型一致。
    """
        # 强制进行 JSON 序列化和反序列化，确保所有值都是基本类型
        result = json.loads(json.dumps(mapping))
        for key, value in result.items():
            if isinstance(value, list):
                # 内部存储使用元组
                value = tuple(value)
            if isinstance(value, tuple):
                if len(value) == 0:
                    message = 'Empty lists are disallowed because their type is unclear.'
                    raise TypeError(message)
                # 列表/元组只能包含基本类型
                if not isinstance(value[0], (str, float, int, bool)):
                    message = 'Lists can only contain strings, floats, ints, bools'
                    message += f' but not {type(value[0])}'
                    raise TypeError(message)
                # 确保列表中所有元素类型一致
                if not all(isinstance(x, type(value[0])) for x in value[1:]):
                    message = 'Elements of a list must all be of the same type.'
                    raise TypeError(message)
            result[key] = value
        return result

    def _format_value(self, value):
        """格式化值，特别是列表/元组的字符串表示。"""
        if isinstance(value, (list, tuple)):
            return '[' + ', '.join(self._format_value(x) for x in value) + ']'
        return str(value)

    def _format_type(self, value):
        """格式化类型名称，将列表/元组的类型表示为 "Type s" (e.g., "ints")。"""
        if isinstance(value, (list, tuple)):
            assert len(value) > 0, value
            return self._format_type(value[0]) + 's'
        return str(type(value).__name__)