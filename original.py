from problem import (n, qn, k, invalid_color, graph,
                     history_add, ancilla_index,
                     invalid_colors, comparator)


def original_init(qc, problem):
    for i in range(n(problem)):
        qc.x(ancilla_index(problem) + i)


def original_qubit_count(problem):
    return (n(problem) * qn(problem)
            + n(problem) + invalid_colors(problem)
            + 1)


def original_compose(qc, problem):
    # check for invalid colors
    for color in range(k(problem), k(problem) + invalid_colors(problem)):
        for node in range(n(problem)):
            invalid_color(qc, problem,
                          color, node, ancilla_index(problem) + node)
            history_add(problem,
                        lambda problem=problem, color=color, node=node,
                        t=ancilla_index(problem) + node:
                        invalid_color(qc, problem, color, node, t))

        sources = [ancilla_index(problem) + node for node in range(n(problem))]
        target = ancilla_index(problem) + n(problem) + color - k(problem)
        qc.mcx(sources, target)
        history_add(problem,
                    lambda sources=sources, target=target:
                    qc.mcx(sources, target))

        for node in reversed(range(n(problem))):
            invalid_color(qc, problem,
                          color, node, ancilla_index(problem) + node)
            history_add(problem,
                        lambda problem=problem, color=color, node=node,
                        t=ancilla_index(problem) + node:
                        invalid_color(qc, problem, color, node, t))

    # check arcs
    l = n(problem) - 1
    f = 0
    for i in range(0, n(problem) - 1):
        r = f
        m = f
        for j in range(i + 1, n(problem)):
            if [i, j] in graph(problem).edges:
                # make comp
                comparator(qc, problem, i, j, ancilla_index(problem) + r)
                history_add(problem,
                            lambda problem=problem, i=i, j=j,
                            t=ancilla_index(problem) + r:
                            comparator(qc, problem, i, j, t))
                r += 1
        if r > f + 1:
            # make mcx
            sources = [ancilla_index(problem) + i for i in range(r)]
            target = ancilla_index(problem) + l
            qc.x(target)
            qc.mcx(sources, target)
            history_add(problem, lambda sources=sources, target=target:
                        qc.mcx(sources, target))
            history_add(problem, lambda target=target: qc.x(target))
            l -= 1
            for j in reversed(range(i + 1, n(problem))):
                if [i, j] in graph(problem).edges:
                    # undo comps
                    comparator(qc, problem, i, j, ancilla_index(problem) + m)
                    history_add(problem,
                                lambda problem=problem, i=i, j=j,
                                t=ancilla_index(problem) + m:
                                comparator(qc, problem, i, j, t))
                    m += 1
        elif r == f + 1:
            f += 1

    return [ancilla_index(problem) + i
            for i in range(n(problem) + invalid_colors(problem))]


original_system = (original_qubit_count, original_init, original_compose)
