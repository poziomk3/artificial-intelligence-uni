import time
import functools


def log_route_info(algo_name="Unknown"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start = args[1]  # 2nd argument
                end = args[2]  # 3rd argument
                start_time = args[3]  # 4th argument
            except IndexError:
                start = end = start_time = "N/A"

            print(f"\n[ALGORITHM: {algo_name}]")
            print(f"  Start:      {start}")
            print(f"  End:        {end}")
            print(f"  Start time: {start_time.strftime('%H:%M:%S')}")
            print("  Computing...")

            start_clock = time.time()
            result = func(*args, **kwargs)
            end_clock = time.time()

            print(f"  Execution time: {end_clock - start_clock:.4f} seconds")
            return result

        return wrapper

    return decorator


def format_algo_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if result is None:
            print("⚠️ No valid route found.")
            return None

        path_edges, cost = result

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
