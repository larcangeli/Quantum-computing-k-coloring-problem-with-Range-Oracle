from math import ceil, sqrt
from problem import (n, qn, ancilla_index,
                     history_add, problem_comp_num,
                     make_components, component_builder, comp_num,
                     greedy_get_components_window
                     )


def balanced_qubits(values):
    return ceil((1 + sqrt(8 * values - 7)) / 2)


def balanced_values(qubits):
    return 1 + qubits * (qubits - 1) / 2


def balanced_qubit_count(problem):
    return (n(problem) * qn(problem)
            + balanced_qubits(problem_comp_num(problem))
            + 1)


def balanced_init(qc, problem):
    for i in range(balanced_qubits(problem_comp_num(problem))):
        qc.x(ancilla_index(problem) + i)


def balanced_compose(qc, problem):
    components = make_components(qc, problem)
    ancillas = balanced_qubits(problem_comp_num(problem))
    outputs = []
    for target_i in range(ancillas-1, 1, -1):
        if comp_num(components) <= target_i + 1:
            comps = greedy_get_components_window(problem, components,
                                                 comp_num(components))
            for i, component in enumerate(map(component_builder, comps)):
                component(ancilla_index(problem) + i)
                history_add(problem,
                            lambda c=component, t=ancilla_index(problem)+i:
                            c(t))
                outputs.append(ancilla_index(problem) + i)
            break

        components_target = greedy_get_components_window(problem, components,
                                                         target_i)
        if len(components_target) == 0:
            break
        for i, component in enumerate(list(map(component_builder,
                                               components_target))):
            component(ancilla_index(problem) + i)
            history_add(problem,
                        lambda c=component, t=ancilla_index(problem) + i: c(t))
        sources = [ancilla_index(problem) + i
                   for i in range(len(components_target))]
        qc.x(ancilla_index(problem)+target_i) #what was missing in Saha el algorithm?
        history_add(problem,
                    lambda target=ancilla_index(problem)+target_i:
                    qc.x(target))
        qc.mcx(sources, ancilla_index(problem)+target_i)
        history_add(problem,
                    lambda sources=sources,
                    target=ancilla_index(problem)+target_i:
                    qc.mcx(sources, target))
        outputs.append(ancilla_index(problem)+target_i)
        for i, component in reversed(list(enumerate(map(component_builder,
                                                        components_target)))):
            component(ancilla_index(problem) + i)
            history_add(problem,
                        lambda c=component, t=ancilla_index(problem) + i: c(t))
    if comp_num(components) <= 2:
        final_comps = greedy_get_components_window(problem, components, 2)
        for i, c in enumerate(map(component_builder, final_comps)):
            c(ancilla_index(problem) + i)
            outputs.append(ancilla_index(problem) + i)
            history_add(problem, lambda c=c, t=ancilla_index(problem)+i: c(t))
        return outputs
    else:
        print(comp_num(components))
        raise Exception("Leftover components after compose"
                        "--qubit count might be wrong")


balanced_system = (balanced_qubit_count, balanced_init, balanced_compose)
