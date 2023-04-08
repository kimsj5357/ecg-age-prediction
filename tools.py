

class ETA:
    def __init__(self, total_iter):
        self.total_iter = total_iter
        self.cur_iter = 0
        self.total_time = 0

    def __call__(self, time, count=1):
        self.total_time += time
        self.cur_iter += count

        avg_time = self.total_time / self.cur_iter
        eta = avg_time * (self.total_iter - self.cur_iter)
        return eta