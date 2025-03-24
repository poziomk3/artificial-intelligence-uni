import time
import functools


import functools
import time

import functools
import time

def print_info(algo_name="Unknown"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start = args[1]
                end = args[2]
                start_time = args[3]
                heuristic_func = args[4] if len(args) > 4 and callable(args[4]) else None
            except IndexError:
                start = end = start_time = "N/A"
                heuristic_func = None

            print(f"\n[ALGORITHM: {algo_name}]")
            print(f"  Start:         {start}")
            print(f"  End:           {end}")
            print(f"  Start time:    {start_time.strftime('%H:%M:%S') if hasattr(start_time, 'strftime') else start_time}")
            if heuristic_func:
                print(f"  Heuristic:     {heuristic_func.__name__}")
            print("  Computing...")

            start_clock = time.time()
            result = func(*args, **kwargs)
            end_clock = time.time()

            print(f"  Execution time: {end_clock - start_clock:.4f} seconds")

            # Try to extract and print the number of checked nodes (assumed to be last element in tuple)
            if isinstance(result, tuple) and isinstance(result[-1], int):
                print(f"  Checked nodes: {result[-1]}")

            return result

        return wrapper

    return decorator




def print_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # return result
        if result is None:
            print("⚠️ No valid route found.")
            return None

        path_edges, cost,checked_nodes = result

        if not path_edges:
            print("⚠️ No valid path found.")
            return result

        previous_line = None
        print("\n✅ Result:\n")
        for edge in path_edges:
            from_stop, to_stop, line, departure, arrival = edge

            if line != previous_line:
                if previous_line is not None:
                    print(f"🚏 Change at: {from_stop} at {departure.strftime('%H:%M:%S')}\n")
                print(f"🚌 Line {line} | Board at {from_stop} at {departure.strftime('%H:%M:%S')}")

            print(f"   → {to_stop} (Arrives at {arrival.strftime('%H:%M:%S')})")

            previous_line = line

        print(f"\n⏳ Arrived at {to_stop} at {arrival.strftime('%H:%M:%S')}")
        print(f"🕒 Total cost: {cost}\n")

        return result  # Still return result if further processing is needed

    return wrapper
