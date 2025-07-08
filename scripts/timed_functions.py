import signal
import sys
import time
import threading
from multiprocessing import Pool, Process, TimeoutError


# class FunctionTimeout(RuntimeError):
#     ...


# def raise_timeout(x, y):
#     raise FunctionTimeout(f'Timed out {type(x)} {type(y)}')


# def timed_function(sleep_time: int) -> bool:
#     signal.signal(signal.SIGALRM, raise_timeout)
#     signal.alarm(5)

#     try:
#         time.sleep(sleep_time)
#         return True
#     except FunctionTimeout as err:
#         print('%d timed out returning false, %s' % (sleep_time, err))
#     return False


# if __name__ == '__main__':
#     num_data = 15
#     data = [i + 1 for i in range(15)]
#     results = [False] * num_data
#     with Pool(10) as p:
#         fut = p.map(timed_function, data)
#         for n in range(num_data):
#             results[n] = fut[n]

#     print(results)


# def set_timeout(event):
#     event.set()


# class TimedOperation:
#     def __init__(self, ident: int):
#         self.ident = ident
#         self.t = False

#     def __call__(self, dur: int):
#         time.sleep(dur)
#         self.t = True
#         print('ident=%d, dur=%d' % (self.ident, dur))


# ops = [TimedOperation(ident=i + 1) for i in range(15)]
# threads = [threading.Timer(5, op, args=(i,)) for i, op in enumerate(ops)]
# for thread in threads:
#     thread.start()
# for thread in threads:
#     thread.join()

# print('result=%s' % ([op.t for op in ops]))


# def is_isomorphic(dur):
#     time.sleep(dur)
#     return True


# class TimedIsomorphicCheck:
#     def __init__(self):
#         self.result = False

#     def __call__(self, dur: int):
#         self.result = is_isomorphic(dur)


# class Wrapper:
#     def __init__(self, dur: int):
#         self._check = TimedIsomorphicCheck()
#         self._p = Process(target=self._check, args=(dur,))

#     @property
#     def result(self) -> bool:
#         self._p.start()
#         self._p.join(5)
#         if self._p.is_alive():
#             self._p.terminate()
#             return False
#         return self._check.result


# # def is_isomorphic_timed(dur):
# #     return Wrapper(dur)


# def main():
#     procs = [Wrapper(i + 1) for i in range(10)]
#     results = [proc.result for proc in procs]
#     print(results)


# main()


def compute(dur) -> bool:
    time.sleep(dur)
    print('completed %d' % (dur))
    return True, dur


with Pool(processes=4) as p:
    procs = [p.apply_async(compute, (i + 1,)) for i in range(10)]
    # results = [proc.get(timeout=5) for proc in procs]
    results = [False] * 10
    for proc in procs:
        try:
            r, d = proc.get(timeout=3)
            results[d] = r
        except TimeoutError as err:
            print(err, file=sys.stderr, end='')
            # results[i] = False

    print(results)
