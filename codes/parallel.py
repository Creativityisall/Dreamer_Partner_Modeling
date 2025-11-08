import multiprocessing as mp
from enum import Enum

class Dummy:
    def __init__(self, make_env, config, i):
        self._env = make_env(config, i)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        return lambda: self._env.step(actions)

    def reset(self):
        return lambda: self._env.reset()
    
    def close(self):
        self._env.close()

class Message(Enum):
    ATTR = 1
    RESET = 2
    STEP = 3
    CLOSE = 4

# TODO: add error handling
class Remote:
    def __init__(self, make_env, config, index):
        self._index = index
        ctx = mp.get_context("spawn")
        self._pipe_local, pipe_remote = ctx.Pipe()
        self._process = ctx.Process(target=env_server, args=(make_env, config, index, pipe_remote))
        self._process.start()

    def _submit(self, msg, data):
        self._pipe_local.send((msg, data))
        return lambda: self._pipe_local.recv()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"Attribute {name} is not available")
        return self._submit(Message.ATTR, name)()

    def reset(self):
        return self._submit(Message.RESET, None)

    def step(self, actions):
        return self._submit(Message.STEP, actions)

    def close(self):
        self._submit(Message.CLOSE, None)
        self._process.join()
        print(f"Remote process {self._index} exited gracefully")

def env_server(make_env, config, index, pipe):
    env = make_env(config, index)
    while True:
        msg, data = pipe.recv()
        if msg == Message.ATTR:
            value = getattr(env, data)
            pipe.send(value)
        elif msg == Message.RESET:
            value = env.reset()
            pipe.send(value)
        elif msg == Message.STEP:
            value = env.step(data)
            pipe.send(value)
        elif msg == Message.CLOSE:
            env.close()
            break
        else:
            raise ValueError(f"Unknown message {msg}")
