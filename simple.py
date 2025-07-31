from problem import (n, qn,
                     history_add, ancilla_index, problem_comp_num,
                     get_components_list,
                     get_greedy_components_list
                     )


def simple_init(qc, problem):
    for i in range(problem_comp_num(problem)):
        qc.x(ancilla_index(problem) + i)


def simple_qubit_count(problem):
    return (n(problem) * qn(problem)
            + problem_comp_num(problem) + 1)


def simple_compose(qc, problem):
    components = get_greedy_components_list(qc, problem)
    for i in range(len(components)):
        components[i](ancilla_index(problem) + i)
        history_add(problem,
                    lambda c=components[i], x=ancilla_index(problem)+i: c(x))
    return [ancilla_index(problem) + i for i in range(len(components))]


simple_system = (simple_qubit_count, simple_init, simple_compose)
