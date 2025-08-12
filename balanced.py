from math import ceil, sqrt
from problem import (n, qn, ancilla_index, problem_comp_num,
                     get_greedy_components_list,
                     history_add)


def balanced_qubits(values):
    return ceil((1 + sqrt(8 * values - 7)) / 2)


def balanced_values(qubits):
    return 1 + qubits * (qubits - 1) / 2


def balanced_qubit_count(problem):
    return (n(problem) * qn(problem))


def balanced_init(qc, problem):
    # no ancilla initialization in ancilla-free design
    return


def balanced_compose(qc, problem):
    components = get_greedy_components_list(qc, problem)
    for builder in components:
        try:
            builder(None)
        except TypeError:
            builder()
        history_add(problem, (lambda b=builder: (lambda: (b(None) if (lambda: True)() else None)))())
    return None


balanced_system = (balanced_qubit_count, balanced_init, balanced_compose)
