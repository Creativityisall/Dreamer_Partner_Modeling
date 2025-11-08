import collections
import concurrent.futures
import functools
import json
import os
import re
import time

import numpy as np

from . import path
from . import printing
from . import timer


class Logger:
    """
  主要的日志记录器类。负责收集不同类型的指标和数据，并调度它们写入各种输出后端。
  """

    def __init__(self, step, outputs, multiplier=1):
        assert outputs, 'Provide a list of logger outputs.'
        # 当前步数（通常是训练步数）
        self.step = step
        # 接收一个或多个输出后端 (如 TerminalOutput, JSONLOutput 等)
        self.outputs = outputs
        # 步数乘数，用于调整写入日志的步数 (e.g., 实际步数 * multiplier)
        self.multiplier = multiplier
        self._last_step = None
        self._last_time = None
        # 存储待写入的指标列表：[(step, name, value), ...]
        self._metrics = []

    @timer.section('logger_add')
    def add(self, mapping, prefix=None):
        """
    添加一个或多个指标到缓冲区。
    Args:
      mapping (dict): {metric_name: value} 形式的字典。
      prefix (str): 可选的前缀。
    """
        mapping = dict(mapping)
        # 限制指标数量和键名长度，防止内存爆炸
        assert len(mapping) <= 1000, list(mapping.keys())
        for key in mapping.keys():
            assert len(key) <= 200, (len(key), key[:200] + '...')

        # 计算实际用于记录的步数
        step = int(self.step) * self.multiplier

        for name, value in mapping.items():
            name = f'{prefix}/{name}' if prefix else name

            # 处理 numpy 数组中的字符串类型
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, str):
                value = str(value)

            if not isinstance(value, str):
                value = np.asarray(value)
                # 验证数组维度：只接受 0, 1, 2, 3, 4 维 (标量, 向量, 图像, 视频)
                if len(value.shape) not in (0, 1, 2, 3, 4):
                    raise ValueError(
                        f"Shape {value.shape} for name '{name}' cannot be "
                        "interpreted as scalar, vector, image, or video.")

            # 将 (step, name, value) 元组添加到缓冲区
            self._metrics.append((step, name, value))

    # ----------------- 方便方法 -----------------
    # 提供了针对特定数据类型的封装方法，同时进行维度检查

    def scalar(self, name, value):
        value = np.asarray(value)
        assert len(value.shape) == 0, value.shape
        self.add({name: value})

    def vector(self, name, value):
        value = np.asarray(value)
        assert len(value.shape) == 1, value.shape
        self.add({name: value})

    def image(self, name, value):
        value = np.asarray(value)
        # 允许 2D (H, W) 或 3D (H, W, C)
        assert len(value.shape) in (2, 3), value.shape
        self.add({name: value})

    def video(self, name, value):
        value = np.asarray(value)
        # 必须是 4D (T, H, W, C)
        assert len(value.shape) == 4, value.shape
        self.add({name: value})

    def text(self, name, value):
        assert isinstance(value, str), (type(value), str(value)[:100])
        self.add({name: value})

    # --------------------------------------------

    @timer.section('logger_write')
    def write(self):
        """
    将缓冲区中的所有指标写入所有配置的输出后端。
    """
        if not self._metrics:
            return

        # 将缓冲区数据转换为不可变的元组 (防止在异步写入过程中被修改)
        metrics_to_write = tuple(self._metrics)

        for output in self.outputs:
            with timer.section(type(output).__name__):
                # 调用每个输出后端的可调用对象 (__call__)
                output(metrics_to_write)

        self._metrics.clear()  # 清空缓冲区

    def close(self):
        """
    关闭日志记录器，确保所有挂起的写入任务完成。
    """
        self.write()
        for output in self.outputs:
            if hasattr(output, 'wait'):
                try:
                    output.wait()  # 等待异步任务完成
                except Exception as e:
                    print(f'Error waiting on output: {e}')


class AsyncOutput:
    """
  抽象基类/包装器，为任何输出回调提供异步写入功能。
  使用 ThreadPoolExecutor 在后台线程中执行实际的写入操作。
  """

    def __init__(self, callback, parallel=True):
        self._callback = callback  # 实际执行写入操作的函数 (_write)
        self._parallel = parallel
        if parallel:
            name = type(self).__name__
            # 创建一个单线程执行器，确保写入操作是顺序的
            self._worker = concurrent.futures.ThreadPoolExecutor(
                1, f'logger_{name}_async')
            self._future = None  # 存储上一个提交的任务的 Future 对象

    def wait(self):
        """等待当前正在执行的异步写入任务完成。"""
        if self._parallel and self._future:
            concurrent.futures.wait([self._future])

    def __call__(self, summaries):
        """
    接收指标，并提交给工作线程进行处理。
    """
        if self._parallel:
            # 确保上一个任务已完成 (或抛出异常，如果 .result() 失败)
            self._future and self._future.result()
            # 提交新的写入任务
            self._future = self._worker.submit(self._callback, summaries)
        else:
            # 同步模式：直接调用回调函数
            self._callback(summaries)


class TerminalOutput:
    """
  将标量指标格式化并打印到终端的输出后端。
  """

    def __init__(self, pattern=r'.*', name=None, limit=50):
        # 可选的正则表达式，用于筛选要显示的指标
        self._pattern = (pattern != r'.*') and re.compile(pattern)
        self._name = name  # 终端输出的名称/标题
        self._limit = limit  # 限制显示的指标数量

    @timer.section('terminal')
    def __call__(self, summaries):
        """处理并打印指标。"""
        # 找到所有指标中的最大步数
        step = max(s for s, _, _, in summaries)
        # 筛选出标量指标
        scalars = {
            k: float(v) for _, k, v in summaries
            if isinstance(v, np.ndarray) and len(v.shape) == 0}

        # 根据 pattern 筛选或根据 limit 截断
        if self._pattern:
            scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
        else:
            truncated = 0
            if len(scalars) > self._limit:
                truncated = len(scalars) - self._limit
                scalars = dict(list(scalars.items())[:self._limit])  # 截断

        # 格式化标量值 (例如，转换为科学计数法或保留两位小数)
        formatted = {k: self._format_value(v) for k, v in scalars.items()}

        # 构造头部标题
        if self._name:
            header = f'{"-" * 20}[{self._name} Step {step:_}]{"-" * 20}'
        else:
            header = f'{"-" * 20}[Step {step:_}]{"-" * 20}'

        # 构造内容
        content = ''
        if self._pattern:
            content += f"Metrics filtered by: '{self._pattern.pattern}'"
        elif truncated:
            content += f'{truncated} metrics truncated, filter to see specific keys.'
        content += '\n'
        if formatted:
            # 将指标以 'key value / key2 value2' 格式连接
            content += ' / '.join(f'{k} {v}' for k, v in formatted.items())
        else:
            content += 'No metrics.'

        printing.print_(f'\n{header}\n{content}\n', flush=True)

    def _format_value(self, value):
        """自定义数值格式化，旨在简洁和可读性。"""
        value = float(value)
        if value == 0:
            return '0'
        # 范围在 [0.01, 10000] 内的数保留两位小数，并移除末尾的零
        elif 0.01 < abs(value) < 10000:
            value = f'{value:.2f}'
            value = value.rstrip('0').rstrip('.')
            return value
        else:
            # 否则使用科学计数法 (1.0e+01)
            value = f'{value:.1e}'
            value = value.replace('.0e', 'e')
            value = value.replace('+0', '').replace('+', '')
            value = value.replace('-0', '-')
        return value


class JSONLOutput(AsyncOutput):
    """
  将标量和字符串指标写入到 JSON Lines (.jsonl) 文件中，支持异步写入。
  """

    def __init__(
            self, logdir, filename='metrics.jsonl', pattern=r'.*',
            strings=False, parallel=True):
        # 异步执行 _write 方法
        super().__init__(self._write, parallel)
        self._pattern = re.compile(pattern)
        self._strings = strings  # 是否记录字符串/文本指标
        logdir = path.Path(logdir)
        logdir.mkdir()
        self._filename = logdir / filename  # 完整的 JSONL 文件路径

    @timer.section('jsonl')
    def _write(self, summaries):
        """实际执行文件写入的方法。"""
        bystep = collections.defaultdict(dict)
        # 按步数组织标量指标和字符串
        for step, name, value in summaries:
            if not self._pattern.search(name):
                continue
            if isinstance(value, str) and self._strings:
                bystep[step][name] = value
            # 只记录 0 维 (标量) 的 numpy 数组
            if isinstance(value, np.ndarray) and len(value.shape) == 0:
                bystep[step][name] = float(value)

        # 将每个步数的字典转换为 JSON 字符串并添加换行符
        lines = ''.join([
            json.dumps({'step': step, **scalars}) + '\n'
            for step, scalars in bystep.items()])

        printing.print_(f'Writing metrics: {self._filename}')

        # 以追加模式写入文件
        with self._filename.open('a') as f:
            f.write(lines)


class TensorBoardOutput(AsyncOutput):
    """
  将各种指标写入 TensorBoard 事件文件，支持异步写入。
  依赖于 TensorFlow 库。
  """

    def __init__(
            self, logdir, fps=20, videos=True, maxsize=1e9, parallel=True):
        super().__init__(self._write, parallel)
        self._logdir = str(path.Path(logdir))
        # 兼容 Google Cloud Storage (GCS) 路径
        if self._logdir.startswith('/gcs/'):
            self._logdir = self._logdir.replace('/gcs/', 'gs://')
        self._fps = fps
        self._writer = None  # TensorBoard FileWriter 实例
        # GCS 路径特有的最大文件大小限制和检查器
        self._maxsize = self._logdir.startswith('gs://') and maxsize
        self._videos = videos  # 是否写入视频
        if self._maxsize:
            # 用于检查文件大小的单线程执行器
            self._checker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._promise = None

        # 禁用 GPU/TPU 设备，因为 TensorBoard 写入通常是 CPU 密集型操作
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices([], 'TPU')

    @timer.section('tensorboard_write')
    def _write(self, summaries):
        """实际执行 TensorBoard 写入的方法。"""
        import tensorflow as tf
        reset = False

        # 检查是否达到最大文件大小限制，如果达到则重置 writer
        if self._maxsize:
            result = self._promise and self._promise.result()
            reset = (self._promise and result >= self._maxsize)
            # 提交下一次文件大小检查任务
            self._promise = self._checker.submit(self._check)

        if not self._writer or reset:
            print('Creating new TensorBoard event file writer.')
            # 创建新的 FileWriter，使用 _retry 机制处理潜在的 I/O 错误
            self._writer = self._retry(functools.partial(
                tf.summary.create_file_writer,
                self._logdir, max_queue=int(1e9), flush_millis=int(1e9)))

        self._writer.set_as_default()  # 设置为默认 writer

        for step, name, value in summaries:
            try:
                if isinstance(value, str):
                    self._retry(tf.summary.text, name, value, step)
                elif len(value.shape) == 0:
                    self._retry(tf.summary.scalar, name, value, step)
                elif len(value.shape) == 1:
                    # 向量/直方图：如果太长则随机采样 1024 个点
                    if len(value) > 1024:
                        value = value.copy()
                        np.random.shuffle(value)
                        value = value[:1024]
                    self._retry(tf.summary.histogram, name, value, step)
                elif len(value.shape) == 2:
                    # 2D 图像 (H, W): 扩展到 (1, H, W, 1)
                    self._retry(tf.summary.image, name, value[None, ..., None], step)
                elif len(value.shape) == 3:
                    # 3D 图像 (H, W, C): 扩展到 (1, H, W, C)
                    self._retry(tf.summary.image, name, value[None], step)
                elif len(value.shape) == 4 and self._videos:
                    # 4D 视频 (T, H, W, C): 专门处理
                    self._video_summary(name, value, step)
            except Exception:
                print('Error writing summary:', name)
                raise

        self._writer.flush()  # 强制写入磁盘

    @timer.section('tensorboard_check')
    def _check(self):
        """检查当前 TensorBoard 事件文件的大小。"""
        import tensorflow as tf
        # 查找所有事件文件
        events = tf.io.gfile.glob(self._logdir.rstrip('/') + '/events.out.*')
        # 返回最新事件文件的大小，如果没有则返回 0
        return tf.io.gfile.stat(sorted(events)[-1]).length if events else 0

    def _retry(self, fn, *args, attempts=3, delay=(3, 10)):
        """处理 TF I/O 操作中的权限错误 (PermissionDeniedError) 的重试机制。"""
        import tensorflow as tf
        for retry in range(attempts):
            try:
                return fn(*args)
            except tf.errors.PermissionDeniedError as e:
                if retry >= attempts - 1:
                    raise
                print(f'Retrying after exception: {e}')
                # 随机等待一段时间后重试
                delay and time.sleep(float(np.random.uniform(*delay)))

    @timer.section('tensorboard_video')
    def _video_summary(self, name, video, step):
        """将视频数据编码为 GIF 并作为图像摘要写入 TensorBoard。"""
        import tensorflow as tf
        import tensorflow.compat.v1 as tf1
        name = name if isinstance(name, str) else name.decode('utf-8')
        assert video.dtype in (np.float32, np.uint8), (video.shape, video.dtype)

        # 浮点数视频转换为 uint8 (0-255)
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)

        try:
            T, H, W, C = video.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=W, colorspace=C)
            # 核心：使用外部函数 _encode_gif 将帧编码为 GIF 字节串
            image.encoded_image_string = _encode_gif(video, self._fps)
            summary.value.add(tag=name, image=image)
            content = summary.SerializeToString()
            # 写入原始 protobuf 摘要
            self._retry(tf.summary.experimental.write_raw_pb, content, step)
        except (IOError, OSError) as e:
            # 如果 GIF 编码失败 (通常是缺少 ffmpeg)，则退回到写入普通图像摘要
            print('GIF summaries require ffmpeg in $PATH.', e)
            self._retry(tf.summary.image, name, video, step)


class WandBOutput:
    """
  将指标写入 Weights & Biases (WandB) 平台。
  """

    def __init__(self, name, pattern=r'.*', **kwargs):
        self._pattern = re.compile(pattern)
        import wandb
        # 初始化 WandB run
        wandb.init(name=name, **kwargs)
        self._wandb = wandb

    def __call__(self, summaries):
        """将收集到的指标按类型转换为 WandB 对象，并记录。"""
        bystep = collections.defaultdict(dict)
        wandb = self._wandb

        for step, name, value in summaries:
            if not self._pattern.search(name):
                continue

            # 根据数据类型转换为相应的 WandB 对象
            if isinstance(value, str):
                bystep[step][name] = value
            elif len(value.shape) == 0:
                bystep[step][name] = float(value)  # 标量
            elif len(value.shape) == 1:
                bystep[step][name] = wandb.Histogram(value)  # 向量转直方图
            elif len(value.shape) in (2, 3):
                # 图像
                value = value[..., None] if len(value.shape) == 2 else value
                assert value.shape[3] in [1, 3, 4], value.shape
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                # 转换为 WandB 要求的 C, H, W 格式
                value = np.transpose(value, [2, 0, 1])
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 4:
                # 视频
                assert value.shape[3] in [1, 3, 4], value.shape
                # 转换为 WandB 要求的 T, C, H, W 格式
                value = np.transpose(value, [0, 3, 1, 2])
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                bystep[step][name] = wandb.Video(value)

        for step, metrics in bystep.items():
            # 调用 wandb.log 记录数据
            self._wandb.log(metrics, step=step)


class ScopeOutput(AsyncOutput):
    """
  将指标写入 Scope 平台。
  """

    def __init__(self, logdir, fps=20, pattern=r'.*'):
        super().__init__(self._write, parallel=True)
        import scope
        logdir = path.Path(logdir)
        self.writer = scope.Writer(logdir, fps=fps)
        self.pattern = (pattern != r'.*') and re.compile(pattern)

    @timer.section('scope')
    def _write(self, summaries):
        """将数据写入 Scope Writer。"""
        for step, name, value in summaries:
            if self.pattern and not self.pattern.search(name):
                continue
            # Scope Writer 接受 step 和一个包含指标的字典
            self.writer.add(step, {name: value})
        self.writer.flush()


class MLFlowOutput:
    """
  将标量指标写入 MLFlow 平台。
  """

    def __init__(self, run_name=None, resume_id=None, config=None, prefix=None):
        import mlflow
        self._mlflow = mlflow
        self._prefix = prefix
        self._pattern = re.compile(r'.*')  # MLFlow 默认记录所有标量
        self._setup(run_name, resume_id, config)

    def __call__(self, summaries):
        """只记录标量指标。"""
        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            # 只处理 0 维 (标量) 的值
            if len(value.shape) == 0 and self._pattern.search(name):
                name = f'{self._prefix}/{name}' if self._prefix else name
                bystep[step][name] = float(value)

        for step, metrics in bystep.items():
            # 调用 MLFlow API 记录指标
            self._mlflow.log_metrics(metrics, step=step)

    def _setup(self, run_name, resume_id, config):
        """设置 MLFlow run，包括追踪 URI、恢复 run 以及记录配置。"""
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'local')
        run_name = run_name or os.environ.get('MLFLOW_RUN_NAME')
        resume_id = resume_id or os.environ.get('MLFLOW_RESUME_ID')
        print('MLFlow Tracking URI:', tracking_uri)
        print('MLFlow Run Name:    ', run_name)
        print('MLFlow Resume ID:   ', resume_id)

        if resume_id:
            # 如果提供了 resume_id，尝试恢复已存在的 run
            runs = self._mlflow.search_runs(None, f'tags.resume_id="{resume_id}"')
            assert len(runs), ('No runs to resume found.', resume_id)
            # 启动已存在的 run
            self._mlflow.start_run(run_name=run_name, run_id=runs['run_id'].iloc[0])
            # 恢复 run 时，重新记录配置参数
            for key, value in config.items():
                self._mlflow.log_param(key, value)
        else:
            # 启动一个新的 run
            tags = {'resume_id': resume_id or ''}
            self._mlflow.start_run(run_name=run_name, tags=tags)


@timer.section('gif')
def _encode_gif(frames, fps):
    """
  使用外部程序 ffmpeg 将视频帧序列编码为 GIF 字节串。
  """
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    # 确定像素格式 (灰度或 RGB24)
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]

    # 构造 ffmpeg 命令：
    # 1. -f rawvideo -vcodec rawvideo ... -i - : 从 stdin 读取原始视频数据
    # 2. -filter_complex ... paletteuse : 使用调色板优化，生成高质量 GIF
    # 3. -f gif - : 输出为 GIF 格式到 stdout
    cmd = ' '.join([
        'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])

    # 启动 ffmpeg 进程
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)

    # 写入所有帧的原始字节数据到 stdin
    for image in frames:
        proc.stdin.write(image.tobytes())

    # 等待进程完成并获取 stdout 和 stderr
    out, err = proc.communicate()

    if proc.returncode:
        # 如果 ffmpeg 返回非零代码，则抛出 I/O 错误
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))

    del proc
    return out  # 返回编码后的 GIF 字节串