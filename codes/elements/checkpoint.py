import inspect
import pickle

from . import path as pathlib
from . import printing
from . import timer
from . import utils


class Saveable:
    """
  辅助类：用于创建 save() -> data 和 load(data) 方法，使对象变得可保存。
  对象可以通过两种方式实现可保存：
  1. 通过 attrs 列表指定要保存的实例属性。
  2. 通过提供自定义的 save 和 load 函数。
  """

    def __init__(self, attrs=None, save=None, load=None):
        # 确保 save 和 load 要么都提供，要么都不提供
        assert bool(save) == bool(load)
        # 确保 save/load 函数和 attrs 列表是互斥的，只能选择一种实现方式
        assert bool(save) != bool(attrs)
        self._save = save
        self._load = load
        self._attrs = attrs

    def save(self):
        """返回需要保存的对象状态数据。"""
        if self._save:
            return self._save()
        if self._attrs:
            # 如果使用 attrs 列表，则返回一个包含指定属性及其值的字典
            return {k: getattr(self, k) for k in self._attrs}

    def load(self, data):
        """从数据中加载对象状态。"""
        if self._load:
            return self._load(data)
        if self._attrs:
            # 如果使用 attrs 列表，则遍历字典并设置实例属性
            for key in self._attrs:
                setattr(self, key, data[key])


class Checkpoint:
    """
  检查点管理类。负责组织、保存、加载和清理检查点文件。

  检查点文件结构如下：
  directory/
    latest               # 文本文件，包含最新完整保存的子文件夹名。
    <timestamp>-<step>/  # 检查点实际数据目录
      foo.pkl            # 单个对象数据文件
      bar.pkl            # 另一个对象数据文件
      baz-0.pkl          # 分片数据的第一个文件
      ...
      baz-N.pkl          # 分片数据的最后一个文件
      done               # 空文件，标记保存操作已完成。
    ...
  """

    def __init__(self, directory=None, keep=1, step=None, write=True):
        assert keep is None or keep >= 1
        # 检查点根目录
        self._directory = directory and pathlib.Path(directory)
        # 要保留的检查点数量
        self._keep = keep
        # 当前训练步数，用于命名文件夹
        self._step = step
        # 是否真正写入文件 (用于测试等场景)
        self._write = write
        # 存储所有可保存的对象 {name: object}
        self._saveables = {}

    def __setattr__(self, name, value):
        """
    属性设置器，用于捕获被检查点跟踪的对象。
    任何非下划线开头的属性都会被视为可保存对象。
    """
        if name.startswith('_'):
            return super().__setattr__(name, value)

        # 检查对象是否实现了 save() 和 load() 方法
        has_load = hasattr(value, 'load') and callable(value.load)
        has_save = hasattr(value, 'save') and callable(value.save)
        if not (has_load and has_save):
            raise ValueError(
                f"Checkpointed object '{name}' must implement save() and load().")
        self._saveables[name] = value

    def __getattr__(self, name):
        """属性获取器，用于从 _saveables 字典中获取被跟踪的对象。"""
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._saveables[name]
        except AttributeError:
            raise ValueError(name)

    def exists(self, path=None):
        """
    检查是否存在一个完整的检查点。
    Args:
      path (Path): 可选，检查特定路径下的检查点。
    """
        assert self._directory or path
        if path:
            result = exists(path)
        else:
            # 默认检查最新的检查点是否存在且完整
            result = bool(self.latest())
        if result:
            print('Found existing checkpoint.')
        else:
            print('Did not find any checkpoint.')
        return result

    @timer.section('checkpoint_save')
    def save(self, path=None, keys=None):
        """
    保存所有或指定的检查点对象。
    """
        assert self._directory or path

        # 确定要保存哪些对象的 save 函数
        if keys is None:
            savefns = {k: v.save for k, v in self._saveables.items()}
        else:
            assert all([not k.startswith('_') for k in keys]), keys
            savefns = {k: self._saveables[k].save for k in keys}

        # 确定保存路径
        if path:
            folder = None
        else:
            # 生成时间戳和步数作为文件夹名
            folder = utils.timestamp(millis=True)
            if self._step is not None:
                folder += f'-{int(self._step):012d}'
            path = self._directory / folder

        printing.print_(f'Saving checkpoint: {path}')

        # 执行保存操作 (调用下面的 save 辅助函数)
        save(path, savefns, self._write)

        # 完成后更新 'latest' 文件并进行清理
        if folder and self._write:
            # 写入最新完成的检查点文件夹名
            (self._directory / 'latest').write_text(folder)
            self._cleanup()

        print('Saved checkpoint.')

    @timer.section('checkpoint_load')
    def load(self, path=None, keys=None):
        """
    加载所有或指定的检查点对象。
    """
        assert self._directory or path

        # 确定要加载哪些对象的 load 函数
        if keys is None:
            loadfns = {k: v.load for k, v in self._saveables.items()}
        else:
            assert all([not k.startswith('_') for k in keys]), keys
            loadfns = {k: self._saveables[k].load for k in keys}

        # 确定加载路径 (如果未指定，则加载最新的)
        if not path:
            path = self.latest()
            assert path

        printing.print_(f'Loading checkpoint: {path}')

        # 执行加载操作 (调用下面的 load 辅助函数)
        load(path, loadfns)
        print('Loaded checkpoint.')

    def load_or_save(self):
        """如果存在检查点，则加载它；否则保存一个新的检查点。"""
        if self.exists():
            self.load()
        else:
            self.save()

    def latest(self):
        """
    读取 'latest' 文件，返回最新完成的检查点路径。
    """
        filename = (self._directory / 'latest')
        if not filename.exists():
            return None
        # 构造完整路径
        return self._directory / filename.read_text()

    def _cleanup(self):
        """
    根据 self._keep 数量删除旧的检查点文件夹。
    """
        if not self._keep:
            return
        folders = self._directory.glob('*')
        # 排除 'latest' 文件
        folders = [x for x in folders if x.name != 'latest']
        # 按名称（时间戳）排序，并保留最新的 self._keep 个
        old = sorted(folders)[:-self._keep]
        for folder in old:
            # 递归删除旧文件夹
            folder.remove(recursive=True)


def exists(path):
    """
  检查给定路径下的检查点是否完整（即是否存在 'done' 文件）。
  """
    path = pathlib.Path(path)
    return (path / 'done').exists()


def save(path, savefns, write=True):
    """
  辅助函数：将数据保存到指定的路径。
  支持单个对象和分片生成器（用于大型对象）。
  """
    path = pathlib.Path(path)
    # 确保目标路径不是一个已完成的检查点
    assert not exists(path), path

    # 创建目录
    write and path.mkdir(parents=True)

    for name, savefn in savefns.items():
        try:
            data = savefn()
            if inspect.isgenerator(data):
                # 处理生成器（分片保存）
                for i, shard in enumerate(data):
                    assert i < 1e5, i
                    if write:
                        buffer = pickle.dumps(shard)
                        # 以 name-<索引>.pkl 的格式保存
                        (path / f'{name}-{i:04d}.pkl').write_bytes(buffer)
            else:
                # 处理单个对象保存
                if write:
                    buffer = pickle.dumps(data)
                    # 以 name.pkl 的格式保存
                    (path / f'{name}.pkl').write_bytes(buffer)
        except Exception:
            print(f"Error save '{name}' to checkpoint.")
            raise

    # 标记保存完成
    write and (path / 'done').write_bytes(b'')


def load(path, loadfns):
    """
  辅助函数：从指定的路径加载数据。
  支持单个文件和分片文件加载。
  """
    path = pathlib.Path(path)
    # 确保检查点是完整的
    assert exists(path), path

    filenames = list(path.glob('*'))
    for name, loadfn in loadfns.items():
        try:
            # 尝试加载单个文件 (name.pkl)
            if (path / f'{name}.pkl') in filenames:
                buffer = (path / f'{name}.pkl').read_bytes()
                data = pickle.loads(buffer)
                loadfn(data)

            # 尝试加载分片文件 (name-0000.pkl)
            elif (path / f'{name}-0000.pkl') in filenames:
                # 找到所有分片并排序
                shards = [x for x in filenames if x.name.startswith(f'{name}-')]
                shards = sorted(shards)

                # 定义一个生成器来按顺序加载分片
                def generator():
                    for filename in shards:
                        buffer = filename.read_bytes()
                        data = pickle.loads(buffer)
                        yield data

                # 将生成器传递给 load 函数
                loadfn(generator())
            else:
                # 既没有找到单个文件，也没有找到分片文件
                raise KeyError(name)

        except Exception:
            print(f"Error loading '{name}' from checkpoint.")
            raise