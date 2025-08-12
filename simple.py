from problem import (n, qn,
                     history_add, ancilla_index, problem_comp_num,
                     get_components_list,
                     get_greedy_components_list
                     )


def simple_init(qc, problem):
    # no ancilla initialization in ancilla-free design
    return


def simple_qubit_count(problem):
    return (n(problem) * qn(problem))


def simple_compose(qc, problem):
    components = get_greedy_components_list(qc, problem)
    for builder in components:
        try:
            builder(None)
        except TypeError:
            builder()
        history_add(problem, (lambda b=builder: (lambda: (b(None) if (lambda: True)() else None)))())
    return None


simple_system = (simple_qubit_count, simple_init, simple_compose)
