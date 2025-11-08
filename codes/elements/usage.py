import gc
import inspect
import os
import re
import threading
import time
import tracemalloc
from collections import defaultdict

from . import agg  # 假设这是一个聚合统计 (Aggregation) 模块
from . import timer  # 假设这是一个计时器模块


class Usage:
    """
  主入口类，用于聚合和调度各种资源统计工具。
  """

    def __init__(self, **kwargs):
        # 可用的统计工具及其对应的类
        available = {
            'psutil': PsutilStats,  # 进程级和全局 CPU/RAM 使用率
            'nvsmi': NvsmiStats,  # 全局 NVIDIA GPU 使用率 (通过命令行)
            'gputil': GputilStats,  # 进程级/全局 GPU 使用率 (通过库，异步)
            'malloc': MallocStats,  # 进程内存分配跟踪 (通过 tracemalloc)
            'gc': GcStats,  # 进程垃圾回收统计 (通过 gc 模块)
        }
        self.tools = {}

        # 根据传入的 kwargs 启用相应的工具
        for name, enabled in kwargs.items():
            assert isinstance(enabled, bool), (name, type(enabled))
            if enabled:
                self.tools[name] = available[name]()

    def stats(self):
        """
    收集所有已启用工具的统计数据。
    """
        stats = {}
        for name, tool in self.tools.items():
            # 调用每个工具的 __call__ 方法并添加前缀
            stats.update({f'{name}/{k}': v for k, v in tool().items()})
        return stats


# --- GPU 使用率 (通过 nvidia-smi 命令行) ---

class NvsmiStats:
    """
  通过执行 'nvidia-smi' 命令行工具来收集全局 GPU 使用率统计。
  """

    PATTERNS = {
        # 正则表达式用于从 nvidia-smi 输出中提取 Min/Avg/Max 百分比
        'compute_min': (r'GPU Utilization Samples(.|\n)+?Min.*?: (\d+) %', 2),  # 注意这里的 group 索引应该是 2
        'compute_avg': (r'GPU Utilization Samples(.|\n)+?Avg.*?: (\d+) %', 2),
        'compute_max': (r'GPU Utilization Samples(.|\n)+?Max.*?: (\d+) %', 2),
        'memory_min': (r'Memory Utilization Samples(.|\n)+?Min.*?: (\d+) %', 2),
        'memory_avg': (r'Memory Utilization Samples(.|\n)+?Avg.*?: (\d+) %', 2),
        'memory_max': (r'Memory Utilization Samples(.|\n)+?Max.*?: (\d+) %', 2),
    }

    def __init__(self):
        pass

    @timer.section('nvsmi_stats')
    def __call__(self):
        # 执行 nvidia-smi 命令
        output = os.popen('nvidia-smi --query -d UTILIZATION 2>&1').read()
        if not output:
            print('To log GPU stats, make sure nvidia-smi is working.')
            return {}

        metrics = {'output': output}

        # 解析输出，提取指标
        for name, (pattern, group) in self.PATTERNS.items():
            # re.findall 返回匹配的组的列表，这里 group 应该是 2 (匹配的数字)
            numbers = [x[group - 1] for x in re.findall(pattern, output)]
            for i, number in enumerate(numbers):
                # 结果转换为分数 (除以 100)
                metrics[f'{name}/gpu{i}'] = float(numbers[i]) / 100
        return metrics


# --- CPU/RAM 使用率 (通过 psutil 库) ---

class PsutilStats:
    """
  使用 psutil 库收集进程级和全局 CPU/RAM 使用率。
  """

    def __init__(self):
        import psutil
        self.proc = psutil.Process()  # 当前进程对象

    @timer.section('psutil_stats')
    def __call__(self):
        import psutil
        gb = 1024 ** 3
        cpus = psutil.cpu_count()
        memory = psutil.virtual_memory()
        stats = {
            'proc_cpu_usage': self.proc.cpu_percent() / 100,  # 进程 CPU 占比
            'proc_ram_frac': self.proc.memory_info().rss / memory.total,  # 进程 RSS 内存占比
            'proc_ram_gb': self.proc.memory_info().rss / gb,  # 进程 RSS 内存 (GB)
            'total_cpu_count': cpus,
            'total_cpu_frac': psutil.cpu_percent() / 100,  # 总 CPU 占比
            'total_ram_frac': memory.percent / 100,  # 总 RAM 占比
            'total_ram_total_gb': memory.total / gb,
            'total_ram_used_gb': memory.used / gb,
            'total_ram_avail_gb': memory.available / gb,
        }
        return stats


# --- GPU 使用率 (通过 GPUtil 库，异步) ---

class GputilStats:
    """
  使用 GPUtil 库收集 GPU 状态，并在单独的线程中持续聚合数据。
  """

    def __init__(self):
        import GPUtil
        self.gpus = GPUtil.getGPUs()
        print(f'GPUtil found {len(self.gpus)} GPUs')
        self.error = None
        # 为每个 GPU 维护一个聚合器
        self.aggs = defaultdict(agg.Agg)
        self.once = True
        # 启动后台线程进行持续收集
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    @timer.section('gputil_stats')
    def __call__(self):
        if self.error:
            # 如果工作线程出错，则在主线程中重新抛出
            raise self.error
        stats = {}
        # 获取每个 GPU 聚合器中的结果
        for i, agg_ in self.aggs.items():
            stats.update(agg_.result(prefix=f'gpu{i}'))

        # 第一次运行时添加 GPU 摘要信息
        if self.once:
            self.once = False
            lines = [f'GPU {i}: {gpu.name}' for i, gpu in enumerate(self.gpus)]
            stats['summary'] = '\n'.join(lines)

        return stats

    def _worker(self):
        """后台工作线程：每 0.5 秒收集一次 GPU 状态并添加到聚合器。"""
        try:
            while True:
                for i, gpu in enumerate(self.gpus):
                    agg = self.aggs[i]
                    agg.add('load', gpu.load, 'avg')
                    # 注意：这里 memoryFree / 1024 / 1024 应该是 / 1024 ** 2 (MB to GB)，但代码中使用 / 1024
                    agg.add('mem_free_gb', gpu.memoryFree / 1024, 'min')
                    agg.add('mem_used_gb', gpu.memoryUsed / 1024, 'max')  # 应该是 memoryUsed
                    agg.add('mem_total_gb', gpu.memoryTotal / 1024)
                    agg.add('memory_util', gpu.memoryUtil, ('min', 'avg', 'max'))
                    agg.add('temperature', gpu.temperature, 'max')
                time.sleep(0.5)
        except Exception as e:
            print(f'Exception in Gputil worker: {e}')
            self.error = e


# --- 垃圾回收统计 (通过 gc 模块) ---

class GcStats:
    """
  使用 Python 的 gc 模块收集和报告垃圾回收 (GC) 统计信息。
  """

    def __init__(self):
        # 注册回调函数，以便在 GC 开始和停止时收集数据
        gc.callbacks.append(self._callback)
        self.stats = agg.Agg()  # 聚合器用于记录 GC 发生频率、耗时等
        self.keys = set()
        self.counts = [{}, {}, {}]  # 记录每个代 (Generation) 的对象类型计数
        self.start = None

    @timer.section('gc_stats')
    def __call__(self, log=True):
        stats = {k: 0 for k in self.keys}
        stats.update(self.stats.result())
        stats['objcounts'] = self._summary()  # 获取对象类型变化的摘要
        log and print(stats['objcounts'])
        self.keys |= set(stats.keys())
        return stats

    def _summary(self):
        """
    生成一个摘要，显示每个 GC 代中对象类型数量的变化。
    """
        lines = ['GC Most Common Types']
        for gen in range(3):  # 遍历三个 GC 代

            # 收集当前代的所有对象，并排除帧对象
            objs = {
                id(obj): obj for obj in gc.get_objects(gen)
                if not inspect.isframe(obj)}
            # 额外包含被这些对象引用的、但未被跟踪的对象
            for obj in list(objs.values()):
                for obj in gc.get_referents(obj):
                    if not gc.is_tracked(obj):
                        objs[id(obj)] = obj

            counts = defaultdict(int)
            for obj in objs.values():
                counts[type(obj).__name__] += 1  # 按类型名计数

            # 计算与上次记录的差值 (delta)
            deltas = {k: v - self.counts[gen].get(k, 0) for k, v in counts.items()}
            self.counts[gen] = counts  # 更新当前计数

            # 仅显示变化最大的 Top 10 类型
            deltas = dict(sorted(deltas.items(), key=lambda x: -abs(x[1]))[:10])

            lines.append(f'\nGeneration {gen}\n')
            for name, delta in deltas.items():
                lines.append(f'- {name}: {delta:+d} ({counts[name]})')  # 格式化输出: 类型: 增量 (当前总数)

        return '\n'.join(lines)

    def _callback(self, phase, info):
        """
    GC 回调函数，在 GC 运行时记录时间、收集和不可收集对象数量。
    注意：此函数不能被 @timer.section 包装。
    """
        now = time.perf_counter_ns()
        if phase == 'start':
            self.start = now  # 记录 GC 开始时间
        if phase == 'stop' and self.start:
            gen = info['generation']
            agg_methods = ('avg', 'max', 'sum')
            self.stats.add(f'gen{gen}/calls', 1, agg='sum')
            self.stats.add(f'gen{gen}/collected', info['collected'], agg_methods)
            # 【注意】这里 info['collected'] 被错误地重复用于 uncollectable，应该使用 info['uncollectable']
            self.stats.add(f'gen{gen}/uncollectable', info['collected'], agg_methods)
            self.stats.add(f'gen{gen}/duration', (now - self.start) / 1e9, agg_methods)
            self.start = None  # 重置开始时间


# --- 内存分配统计 (通过 tracemalloc 库) ---

class MallocStats:
    """
  使用 Python 的 tracemalloc 模块跟踪内存分配情况。
  """

    def __init__(self, root_module):
        tracemalloc.start()  # 启动内存跟踪
        self.root_module = root_module  # 用于过滤/分组的根模块名
        self.previous = None  # 用于计算两次快照之间差异的上一快照

    @timer.section('malloc_stats')
    def __call__(self, log=True):
        stats = {}
        snapshot = tracemalloc.take_snapshot()  # 拍摄当前内存快照

        stats['full'] = self._summary(snapshot)  # 完整分配报告
        stats['diff'] = self._summary(snapshot, self.previous)  # 与上一快照的差异报告
        self.previous = snapshot

        log and print(stats['full'])
        return stats

    def _summary(self, snapshot, relative=None, top=50):
        """
    生成内存分配统计摘要。
    """
        # 根据是否有相对快照，决定是使用 statistics 还是 compare_to
        if relative:
            statistics = snapshot.compare_to(relative, 'traceback')
        else:
            statistics = snapshot.statistics('traceback')

        # 聚合：(文件名, 行号) -> [大小, 计数]
        agg = defaultdict(lambda: [0, 0])

        for stat in statistics:
            filename = stat.traceback[-1].filename
            lineno = stat.traceback[-1].lineno
            root = self.root_module

            # 尝试找到与 root_module 相关的帧，用于更精确地归因
            for frame in reversed(stat.traceback):
                if f'/{root}/' in frame.filename:
                    filename = f'{root}/' + frame.filename.split(f'/{root}/')[-1]
                    lineno = frame.lineno
                    break

            size = stat.size_diff if relative else stat.size
            count = stat.count_diff if relative else stat.count

            agg[(filename, lineno)][0] += size
            agg[(filename, lineno)][1] += count

        lines = []
        lines.append('\nMemory Allocation' + (' Changes' if relative else ''))

        # 报告 Top N 按大小排序
        lines.append(f'\nTop {top} by size:\n')
        entries = sorted(agg.items(), key=lambda x: -abs(x[1][0]))
        for (filename, lineno), (size, count) in entries[:top]:
            size = size / (1024 ** 2)  # 转换为 MB
            lines.append(f'- {size:.2f}Mb ({count}) {filename}:{lineno}')

        # 报告 Top N 按计数排序
        lines.append(f'\nTop {top} by count:\n')
        entries = sorted(agg.items(), key=lambda x: -abs(x[1][1]))
        for (filename, lineno), (size, count) in entries[:top]:
            size = size / (1024 ** 2)  # 转换为 MB
            lines.append(f'- {size:.2f}Mb ({count}) {filename}:{lineno}')

        return '\n'.join(lines)