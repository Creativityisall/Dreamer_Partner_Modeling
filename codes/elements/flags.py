import re
import sys

from . import config


class Flags:
    """
  命令行参数解析器，将命令行标志 (--key=value) 转换为类型安全的配置对象。
  它使用 Config 实例作为配置基础，并利用其类型信息和 update 方法。
  """

    def __init__(self, *args, **kwargs):
        # 使用 config.Config 实例作为基础配置，包含了默认值和类型信息
        self._config = config.Config(*args, **kwargs)

    def parse(self, argv=None, help_exits=True):
        """
    解析命令行参数，只返回已知的配置项。
    如果遇到未知或无法解析的参数，则抛出错误。
    """
        # 调用 parse_known 进行解析
        parsed, remaining = self.parse_known(argv)

        # 检查所有剩余的参数是否都是未知的 flag
        for flag in remaining:
            if flag.startswith('--') and flag[2:] not in self._config.flat:
                raise KeyError(f"Flag '{flag}' did not match any config keys.")

        # 如果还有剩余参数（既不是已知 flag 也不是未知 flag），则抛出错误
        if remaining:
            raise ValueError(
                f'Could not parse all arguments. Remaining: {remaining}')

        return parsed

    def parse_known(self, argv=None, help_exits=False):
        """
    解析命令行参数，返回已知的配置项和一个包含所有剩余/未知参数的列表。
    """
        if argv is None:
            # 默认使用 sys.argv[1:] (排除脚本名)
            argv = sys.argv[1:]

        # 处理 --help 标志
        if '--help' in argv:
            print('\nHelp:')
            # 格式化并打印基础 Config 的内容作为帮助信息
            lines = str(self._config).split('\n')[2:]
            print('\n'.join('--' + re.sub(r'[:,\[\]]', '', x) for x in lines))
            help_exits and sys.exit()

        parsed = {}  # 存储解析出的键值对
        remaining = []  # 存储无法解析的参数
        key = None  # 当前正在解析的 flag 键
        vals = None  # 当前 flag 对应的值列表

        # 遍历命令行参数
        for arg in argv:
            if arg.startswith('--'):
                # 遇到新的 flag
                if key:
                    # 如果上一个 flag 尚未提交，则先提交它
                    self._submit_entry(key, vals, parsed, remaining)

                if '=' in arg:
                    # 处理 --key=value 格式
                    key, val = arg.split('=', 1)
                    vals = [val]
                else:
                    # 处理 --key value1 value2 格式
                    key, vals = arg, []
            else:
                # 遇到非 flag 的参数
                if key:
                    # 如果前面有 flag 键，则认为这是它的值
                    vals.append(arg)
                else:
                    # 否则将其视为剩余参数
                    remaining.append(arg)

        # 提交最后一个解析到的 flag
        self._submit_entry(key, vals, parsed, remaining)

        # 使用 Config.update() 方法将解析出的参数应用到基础配置上，确保类型安全
        parsed = self._config.update(parsed)

        return parsed, remaining

    def _submit_entry(self, key, vals, parsed, remaining):
        """
    将解析出的一个 flag 及其值 (key, vals) 提交到 parsed 字典或 remaining 列表。
    """
        if not key and not vals:
            return

        if not key:
            # 如果有值但没有 flag (不应该发生)，则添加到剩余参数
            vals = ', '.join(f"'{x}'" for x in vals)
            remaining.extend(vals)
            return

        name = key[len('--'):]  # 移除前缀 '--'

        if '=' in name:
            # 如果 flag 中包含 '='，但前面没有解析到（如 --key==value），则视为无法解析
            remaining.extend([key] + vals)
            return

        if not vals:
            # 如果 flag 后面没有值，则将其添加到剩余参数 (视为无法解析)
            remaining.extend([key])
            return

        # ------------------ 处理特殊语法和匹配 ------------------

        if name.endswith('+') and name[:-1] in self._config:
            # 1. 列表追加模式：--key+ value1 value2
            key = name[:-1]
            default = self._config[key]

            if not isinstance(default, tuple):
                raise TypeError(
                    f"Cannot append to key '{key}' which is of type "
                    f"'{type(default).__name__}' instead of tuple.")

            # 如果该键尚未被解析，先用默认值初始化
            if key not in parsed:
                parsed[key] = default

            # 解析新值并追加到现有值之后
            parsed[key] += self._parse_flag_value(default, vals, key)

        elif self._config.IS_PATTERN.fullmatch(name):
            # 2. 正则表达式模式匹配：--.*name.* value
            pattern = re.compile(name)
            # 找到所有匹配该模式的扁平配置键
            keys = [k for k in self._config.flat if pattern.fullmatch(k)]

            if keys:
                # 对所有匹配的键应用该值
                for key in keys:
                    parsed[key] = self._parse_flag_value(self._config[key], vals, key)
            else:
                # 如果没有匹配的键，则视为无法解析
                remaining.extend([key] + vals)

        elif name in self._config:
            # 3. 标准键匹配：--key value
            key = name
            # 解析值并设置
            parsed[key] = self._parse_flag_value(self._config[key], vals, key)

        else:
            # 4. 未知键：添加到剩余参数
            remaining.extend([key] + vals)

    def _parse_flag_value(self, default, value, key):
        """
    将命令行值 (字符串列表) 转换为目标类型 (default 的类型)。
    执行类型转换和验证。
    """
        # 将输入值确保为元组或列表
        value = value if isinstance(value, (tuple, list)) else (value,)

        # 递归处理列表/元组类型的值
        if isinstance(default, (tuple, list)):
            # 允许使用逗号分隔的单个参数 (如: --list_flag="a,b,c")
            if len(value) == 1 and ',' in value[0]:
                value = value[0].split(',')
            # 递归地对列表中的每个元素进行类型转换（使用 default[0] 的类型）
            return tuple(self._parse_flag_value(default[0], [x], key) for x in value)

        # 对于非列表/元组类型，必须只收到一个值
        if len(value) != 1:
            raise TypeError(
                f"Expected a single value for key '{key}' but got: {value}")

        value = str(value[0])  # 提取唯一的字符串值

        if default is None:
            # 如果默认值是 None，则不进行类型转换，直接返回字符串
            return value

        # ------------------ 特殊类型转换 ------------------

        if isinstance(default, bool):
            # 布尔值：只接受 'False' 和 'True' 字符串 (大小写敏感)
            try:
                return bool(['False', 'True'].index(value))
            except ValueError:
                message = f"Expected bool but got '{value}' for key '{key}'."
                raise TypeError(message)

        if isinstance(default, int):
            # 整数：允许使用科学记数法（如 1e6）并确保没有小数部分
            try:
                value = float(value)
                assert float(int(value)) == value  # 检查是否是整数值
            except (ValueError, TypeError, AssertionError):
                message = f"Expected int but got '{value}' for key '{key}'."
                raise TypeError(message)
            return int(value)

        if isinstance(default, dict):
            # 如果键指向一个完整的字典，则要求用户指定子键（Config 不允许直接覆盖整个字典）
            raise KeyError(
                f"Key '{key}' refers to a whole dict. Please speicfy a subkey.")

        # ------------------ 一般类型转换 ------------------

        try:
            # 尝试将字符串转换为默认值 (default) 的类型
            return type(default)(value)
        except ValueError:
            raise TypeError(
                f"Cannot convert '{value}' to type '{type(default).__name__}' for "
                f"key '{key}'.")