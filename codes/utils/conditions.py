class Condition:

    def __init__(self):
        pass

    def __call__(self, step: int) -> bool:
        """
        This method should not modify the state of the condition if update is False.
        """
        pass

class Every(Condition):

    def __init__(self, every, after=0, initial=True):
        assert isinstance(every, int) and every > 0
        self._every = every
        self._after = after
        self._initial = initial
        self._prev = None

    def __call__(self, step) -> bool:
        step = int(step)
        if step < self._after:
            return False

        if self._prev is None:
            result = self._initial
            self._prev = self._after
        elif step >= self._prev + self._every:
            result = True
            self._prev += self._every
        else:
            result = False

        return result

    def save(self):
        return self._prev

    def load(self, value):
        self._prev = value

class Ratio(Condition):

    def __init__(self, ratio, after=0, initial=1):
        assert isinstance(ratio, float) and ratio > 0
        self._ratio = ratio
        self._after = after
        self._initial = initial
        self._prev = None

    def __call__(self, step) -> int:
        step = int(step)
        if step < self._after:
            return 0

        if self._prev is None:
            reps = self._initial
            self._prev = self._after
        else:
            reps = int(self._ratio * (step - self._prev))
            self._prev += int(reps / self._ratio)
        
        return reps

    def save(self):
        return self._prev

    def load(self, value):
        self._prev = value

class After(Condition):

    def __init__(self, after, inclusive=False):
        self._after = after
        self._inclusive = inclusive

    def __call__(self, step, update=False):
        step = int(step)
        result = False
        if self._inclusive:
            result = step >= self._after
        else:
            result = step > self._after
        return result
