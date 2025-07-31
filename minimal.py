from math import floor, ceil, log2
from problem import (n, qn, ancilla_index, problem_comp_num,
                     get_greedy_components_list,
                     history_add)


def minimal_init(qc, problem):
    qc.x(ancilla_index(problem))
    qc.x(ancilla_index(problem) + 1)


def minimal_qubit_count(problem):
    return (n(problem) * qn(problem)
            + ceil(log2(problem_comp_num(problem))) + 1
            + 1)


def put(qc, problem, i, components):
    if len(components) != 2 ** max(0, i-1):
        raise Exception("Put at level %d requires %d constraints."
                        " Got %d instead."
                        % (i, 2 ** max(0, i-1), len(components)))

    if i < 2:  # first two lines are direct
        target = ancilla_index(problem) + i
        components[0](target)
        history_add(problem, lambda c=components[0], target=target: c(target))
        return

    end = 0
    intervals = []
    for j in reversed(range(i)):
        start = end
        intervals.append(start)
        end += 2 ** max(0, j - 1)
        put(qc, problem, j, components[start:end])
    sources = [ancilla_index(problem) + r for r in range(i)]
    target = ancilla_index(problem) + i
    qc.mcx(sources, target)
    history_add(problem,
                lambda sources=sources, target=target: qc.mcx(sources, target))
    end = len(components)
    for j, start in zip(range(i), reversed(intervals)):
        put(qc, problem, j, components[start:end])
        end = start


# def setup(qc, problem, components):
#     output_indexes = []
#     end = 0
#     print(len(components))
#     while end != len(components):
#         start = end
#         i = floor(log2(len(components) - start)) + 1
#         end += 2 ** max(0, i - 1)
#         print("putting", end-start, "components at", i)
#         put(qc, problem, i, components[start:end])
#         output_indexes.append(ancilla_index(problem) + i)
#     return output_indexes


# The actual way it should be done (by fixing put)
# def setup(qc, problem, components):
#     output_indexes = []
#     values = []
#     i = 0

#     while sum(values) != len(components):
#         chunk = min(2 ** max(i-1, 0), len(components) - sum(values))
#         values.append(chunk)
#         i += 1

#     c_i = 0
#     for i, chunk in reversed(list(enumerate(values))):
#         put(qc, problem, i, components[c_i:c_i + chunk])
#         c_i += chunk
#         output_indexes.append(ancilla_index(problem) + i)
#     return output_indexes


def setup(qc, problem, components):
    output_indexes = []
    values = []
    indexes = []
    i = ceil(log2(problem_comp_num(problem)))  # last ancilla

    while sum(values) != len(components):
        chunk = 2 ** max(i-1, 0)
        if chunk <= len(components) - sum(values):
            values.append(chunk)
            indexes.append(i)
        i -= 1

    c_i = 0
    for i, chunk in zip(indexes, values):
        put(qc, problem, i, components[c_i:c_i + chunk])
        c_i += chunk
        output_indexes.append(ancilla_index(problem) + i)
    return output_indexes



def minimal_compose(qc, problem):
    components = get_greedy_components_list(qc, problem)
    return setup(qc, problem, components)


minimal_system = (minimal_qubit_count, minimal_init, minimal_compose)
