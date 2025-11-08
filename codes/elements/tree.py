from . import printing  # 假设这是一个用于格式化输出的模块


def map(fn, *trees, isleaf=None):
    """
  递归地遍历一个或多个嵌套 Python 结构（列表、元组、字典），
  并在所有“叶子节点”上应用函数 `fn`。

  Args:
    fn (callable): 要应用于叶子节点的函数。
    *trees (tuple): 一个或多个结构相同的嵌套 Python 结构。
    isleaf (callable, optional): 可选函数，用于自定义判断一个节点是否为叶子节点。
                                 如果返回 True，则停止递归并应用 fn。
  Returns:
    新的嵌套结构，其叶子节点是 fn 的结果。
  Raises:
    TypeError: 如果输入结构不匹配或不兼容。
  """
    assert trees, 'Provide one or more nested Python structures'
    kw = dict(isleaf=isleaf)  # 传递 isleaf 参数到递归调用
    first = trees[0]

    try:
        # 1. 类型和结构检查 (确保所有输入的结构形状相同)
        assert all(isinstance(x, type(first)) for x in trees)

        # 2. 自定义叶子节点检查
        if isleaf and isleaf(trees[0]):
            return fn(*trees)

        # 3. 递归处理列表 (list)
        if isinstance(first, list):
            assert all(len(x) == len(first) for x in trees)
            return [map(
                fn, *[t[i] for t in trees], **kw) for i in range(len(first))]

        # 4. 递归处理元组 (tuple)
        if isinstance(first, tuple):
            assert all(len(x) == len(first) for x in trees)
            return tuple([map(
                fn, *[t[i] for t in trees], **kw) for i in range(len(first))])

        # 5. 递归处理标准字典 (dict)
        if isinstance(first, dict):
            assert all(set(x.keys()) == set(first.keys()) for x in trees)
            return {k: map(fn, *[t[k] for t in trees], **kw) for k in first}

        # 6. 递归处理其他类似字典的对象 (如 defaultdict, OrderedDict)
        if hasattr(first, 'keys') and hasattr(first, 'get'):
            assert all(set(x.keys()) == set(first.keys()) for x in trees)
            # 创建与第一个输入相同类型的对象
            return type(first)(
                {k: map(fn, *[t[k] for t in trees], **kw) for k in first})

    except AssertionError:
        # 捕获结构不匹配错误，并抛出更具描述性的 TypeError
        raise TypeError(printing.format_(trees))

    # 7. 叶子节点：如果不是上述容器类型，则认为是叶子节点，应用函数 fn
    return fn(*trees)


def flatten(tree, isleaf=None):
    """
  将嵌套结构扁平化为 (叶子节点元组, 结构模板)。

  Args:
    tree: 要扁平化的嵌套结构。
    isleaf (callable, optional): 用于判断叶子节点的函数。
  Returns:
    (tuple, structure): 包含所有叶子节点的元组，以及一个保持原结构、
                        但所有叶子节点都被 None 替换的结构模板。
  """
    leaves = []
    # 使用 map 遍历结构，将所有叶子节点收集到 leaves 列表中
    map(lambda x: leaves.append(x), tree, isleaf=isleaf)
    # 再次使用 map 创建结构模板，所有叶子节点被替换为 None
    structure = map(lambda x: None, tree, isleaf=isleaf)
    return tuple(leaves), structure


def unflatten(leaves, structure):
    """
  根据结构模板，将扁平的叶子节点列表恢复为原始的嵌套结构。

  Args:
    leaves (iterable): 包含所有叶子节点的列表或元组。
    structure: 结构模板（由 flatten 函数返回）。
  Returns:
    object: 恢复后的嵌套结构。
  """
    leaves = iter(tuple(leaves))  # 创建叶子节点的迭代器
    # 使用 map 遍历结构模板，并通过 next(leaves) 依次填充叶子节点
    return map(lambda x: next(leaves), structure)


def flatdict(tree, sep='/'):
    """
  将嵌套的字典/元组结构扁平化为单层字典，
  使用分隔符 (sep) 连接嵌套键。

  Args:
    tree (dict or tuple): 要扁平化的嵌套结构。
    sep (str): 用于连接键的分隔符。
  Returns:
    dict: 扁平化后的字典 (key 是路径，value 是叶子节点)。
  """
    assert isinstance(tree, (dict, tuple)), type(tree)
    mapping = {}

    # 遍历字典的键值对
    if isinstance(tree, dict):
        iterator = tree.items()
    # 将元组视为字典，键为索引 '[i]'
    else:
        iterator = {f'[{i}]': x for i, x in enumerate(tree)}.items()

    for key, value in iterator:
        if isinstance(value, dict):
            # 递归处理嵌套字典
            inner = flatdict(value, sep=sep)
            mapping.update({f'{key}{sep}{k}': v for k, v in inner.items()})
        elif isinstance(value, tuple):
            # 递归处理嵌套元组 (将其视为索引化的临时字典)
            inner = flatdict({f'[{i}]': x for i, x in enumerate(value)}, sep=sep)
            mapping.update({f'{key}{sep}{k}': v for k, v in inner.items()})
        else:
            # 叶子节点
            mapping[key] = value

    return mapping


def nestdict(mapping, sep='/'):
    """
  将扁平字典恢复为嵌套的字典/元组结构。
  能够识别并重建扁平化时被转换的元组（键为 '[i]' 的字典）。

  Args:
    mapping (dict): 扁平字典 (key 是路径)。
    sep (str): 用于分割路径的分隔符。
  Returns:
    object: 恢复后的嵌套字典或元组。
  """
    assert isinstance(mapping, dict)
    tree = {}

    # 1. 恢复嵌套字典结构
    for path, value in mapping.items():
        node = tree
        parts = path.split(sep)
        for part in parts[:-1]:
            # 使用 setdefault 确保中间节点是字典
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # 2. 递归后处理：将键为 '[i]' 的字典转换回元组
    def post(tree):
        if isinstance(tree, dict):
            # 递归处理所有子节点
            tree = {k: post(v) for k, v in tree.items()}

            # 检查当前字典是否应该是一个元组
            if all(k.startswith('[') and k.endswith(']') for k in tree):
                # 确保键是连续的 '[0]', '[1]', ...
                available = set(int(x[1:-1]) for x in tree.keys())
                assert available == set(range(len(tree))), available
                # 转换为元组
                tree = tuple(tree[f'[{i}]'] for i in range(len(tree)))

        return tree

    tree = post(tree)
    return tree