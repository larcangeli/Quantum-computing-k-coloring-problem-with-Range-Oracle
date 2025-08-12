from problem import (n, qn, ancilla_index, problem_comp_num,
                     get_greedy_components_list,
                     history_add)


def original_init(qc, problem):
    # no ancillas in ancilla-free design
    return


def original_qubit_count(problem):
    return n(problem) * qn(problem)


def original_compose(qc, problem):
    components = get_greedy_components_list(qc, problem)
    for builder in components:
        try:
            builder(None)
        except TypeError:
            builder()
        history_add(problem, (lambda b=builder: (lambda: (b(None) if (lambda: True)() else None)))())
    return None


original_system = (original_qubit_count, original_init, original_compose)
