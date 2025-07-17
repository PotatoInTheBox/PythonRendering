# profiler.py

import time

# Accumulates total time spent in named segments
_profile_accumulators = {}
# keep track of our named profilers
_profile_timers = {}

enabled_profiler = True

class Profiler:
    @staticmethod
    def profile_accumulate_start(name: str):
        if enabled_profiler:
            if name not in _profile_accumulators:
                _profile_accumulators[name] = [0.0, 0, None]  # [total_time, count, start_time]
            _profile_accumulators[name][2] = time.perf_counter()  # Reset start time

    @staticmethod
    def profile_accumulate_end(name: str):
        if enabled_profiler:
            if name not in _profile_accumulators or _profile_accumulators[name][2] is None:
                return  # ignore unmatched end
            start = _profile_accumulators[name][2]
            elapsed = time.perf_counter() - start
            _profile_accumulators[name][0] += elapsed
            _profile_accumulators[name][1] += 1
            _profile_accumulators[name][2] = None  # clear start

    @staticmethod
    def profile_accumulate_report(intervals=1):
        if enabled_profiler:
            print("\n////////==== Report Start ====\\\\\\\\\\\\\\\\")
            grand_total = sum(total for total, count, _ in _profile_accumulators.values())

            # Sort keys: normal entries first (alphabetical), then those starting with "f:"
            sorted_items = sorted(_profile_accumulators.items(), key=lambda x: (x[0].startswith("f:"), x[0]))

            for name, (total, count, _) in sorted_items:
                if count == 0:
                    continue
                total_ms = total * 1000
                avg_ms = (total / (count / intervals)) * 1000
                percent = (total / grand_total) * 100 if grand_total > 0 else 0
                if percent >= 100:
                    percent_str = "100%"
                elif percent >= 10:
                    percent_str = f"{percent:4.1f}%"
                else:
                    percent_str = f"{percent:4.2f}%"
                print(f"{percent_str} — {name}: {total_ms/intervals:.3f}ms total over {count/intervals} calls (avg {avg_ms/intervals:.3f}ms)")

            _profile_accumulators.clear()
            print("\\\\\\\\\\\\\\\\==== Report End   ====////////")

    @staticmethod
    def timed(name=""):
        def wrapper(fn):
            def inner(*args, **kwargs):
                if enabled_profiler:
                    label = name or fn.__name__
                    label = "f:" + label
                    start = time.perf_counter()
                    result = fn(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    if label not in _profile_accumulators:
                        _profile_accumulators[label] = [0.0, 0, None]
                    _profile_accumulators[label][0] += elapsed
                    _profile_accumulators[label][1] += 1
                    return result
                else:
                    return fn(*args, **kwargs)
            return inner
        return wrapper

    # @staticmethod
    # def profile_start(name: str, frame_count, n=60):
    #     if frame_count % n == 0:
    #         _profile_timers[name] = time.perf_counter()

    # @staticmethod
    # def profile_end(name: str, frame_count, n=60):
    #     if frame_count % n == 0:
    #         if name in _profile_timers:
    #             elapsed = (time.perf_counter() - _profile_timers.pop(name)) * 1000
    #             print(f"{name}: {elapsed:.3f}ms")
    #         else:
    #             print(f"Warning: profile_end called for '{name}' without matching profile_start")
