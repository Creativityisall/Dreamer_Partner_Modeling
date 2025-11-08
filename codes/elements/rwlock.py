import contextlib
import threading


class RWLock:
    """
  读写锁（Reader-Writer Lock）实现。
  允许并发读取，但在写入时提供独占访问。
  """

    def __init__(self):
        # 核心互斥锁，用于保护 RWLock 自身的内部状态变量（如计数器）
        self.lock = threading.Lock()

        # 【注意】这个锁（active_writer_lock）用于实现对写入操作的独占访问。
        # 在 acquire_write 中，它被用于阻止多个写入线程同时进入临界区。
        self.active_writer_lock = threading.Lock()

        # 记录当前尝试获取写入锁的线程数量（或已获取写入意图的线程数量）
        self.writer_count = 0
        # 记录当前正在等待，希望获取读取锁的线程数量
        self.waiting_reader_count = 0
        # 记录当前正在执行读取操作的线程数量
        self.active_reader_count = 0

        # 条件变量：当所有读者完成时，用于唤醒等待的写者
        self.readers_finished_cond = threading.Condition(self.lock)
        # 条件变量：当写者完成时，用于唤醒等待的读者
        self.writers_finished_cond = threading.Condition(self.lock)

    @property
    @contextlib.contextmanager
    def reading(self):
        """
    提供一个上下文管理器，用于方便地以读取模式访问共享资源 (with rwlock.reading: ...)。
    """
        try:
            self.acquire_read()
            yield
        finally:
            self.release_read()

    @property
    @contextlib.contextmanager
    def writing(self):
        """
    提供一个上下文管理器，用于方便地以写入模式访问共享资源 (with rwlock.writing: ...)。
    """
        try:
            self.acquire_write()
            yield
        finally:
            self.release_write()

    def acquire_read(self):
        """
    获取读取锁。
    如果当前有写者活跃（writer_count > 0），则读者必须等待。
    """
        with self.lock:
            if self.writer_count:
                self.waiting_reader_count += 1
                # 等待，直到 writer_count 变为 0（写者完成）
                while self.writer_count:
                    self.writers_finished_cond.wait()
                self.waiting_reader_count -= 1
            # 允许读取
            self.active_reader_count += 1

    def release_read(self):
        """
    释放读取锁。
    如果这是最后一个活跃的读者，并且有写者正在等待，则通知等待的写者。
    """
        with self.lock:
            assert self.active_reader_count > 0
            self.active_reader_count -= 1
            # 如果当前没有活跃读者，并且有写者在等待（通过 writer_count 表示）
            if not self.active_reader_count and self.writer_count:
                # 通知所有等待的写者（通过 writers_finished_cond 间接通知）
                self.readers_finished_cond.notify_all()

    def acquire_write(self):
        """
    获取写入锁。
    写者必须等待所有活跃的读者完成。
    【注意】这个实现有两个锁：self.lock (状态锁) 和 self.active_writer_lock (独占锁)。
    """
        with self.lock:
            # 标记有一个写者意图获取锁
            self.writer_count += 1
            # 等待，直到没有活跃的读者
            while self.active_reader_count:
                self.readers_finished_cond.wait()

        # 在没有任何活跃读者之后，写者尝试获取独占锁。
        # 这样可以阻止多个写者同时进入临界区，也确保写操作的独占性。
        self.active_writer_lock.acquire()

    def release_write(self):
        """
    释放写入锁。
    释放独占锁，然后通知等待的读者。
    """
        # 释放独占锁
        self.active_writer_lock.release()

        with self.lock:
            assert self.writer_count > 0
            # 减少写者计数
            self.writer_count -= 1

            # 如果当前没有写者（即当前写者是最后一个）
            # 并且有等待的读者，则通知他们
            if not self.writer_count and self.waiting_reader_count:
                self.writers_finished_cond.notify_all()