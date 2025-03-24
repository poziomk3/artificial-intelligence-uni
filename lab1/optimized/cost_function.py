def time_cost(g, arrival_time, route, prev_line):
    wait = (route.departure_time - arrival_time).total_seconds() / 60
    travel = route.time_delta.total_seconds() / 60
    return g + wait + travel


def line_change_cost(g, arrival_time, route, prev_line):
    return g + (1 if prev_line and prev_line != route.line else 0)