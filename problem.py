from math import ceil, log2
import Oracles as oracles


def make_problem(graph, k):
    return (graph, k, [])


def graph(problem):
    return problem[0]


def k(problem):
    return problem[1]


def n(problem):
    return graph(problem).order()


def history(problem):
    return problem[2]


def history_add(problem, c):
    history(problem).append(c)


def qn(problem):  # qubits per node
    return ceil(log2(k(problem)))


def decompose(problem):
    while history(problem):
        history(problem).pop()()


def node_qubits(problem, i):
    return list(range(qn(problem)*i, qn(problem)*(i+1)))


def ancilla_index(problem):
    return n(problem) * qn(problem)


def invalid_colors(problem):
    return 2 ** qn(problem) - k(problem)

'''
def problem_comp_num(problem):
    return (invalid_colors(problem) * n(problem)
            + len(graph(problem).edges))
'''
#New version using oracle greater-than (no need to create one component for each invalid color, just one per node)
def problem_comp_num(problem):
    return (n(problem)
            + len(graph(problem).edges))

def comparator(qc, problem, i, j, f):
    if i > j:
        i, j = j, i
    a = node_qubits(problem, i)
    b = node_qubits(problem, j)
    for i, j in zip(a, b):
        qc.cx(i, j)
        qc.x(j)
    qc.mcx(b, f)
    for i, j in reversed(list(zip(a, b))):
        qc.x(j)
        qc.cx(i, j)


def invalid_color(qc, problem, color, i, dest):
    a = node_qubits(problem, i)
    x_gates = [not bool(int(x)) for x in bin(color)[2:]]
    qnum = len(x_gates)
    for j in range(qnum):
        if x_gates[j]:
            qc.x(a[0] + j)
    qc.mcx(a, dest)
    for j in range(qnum):
        if x_gates[j]:
            qc.x(a[0] + j)

        
# New version of ancilla marking for invalid colors on node i (using a greater-than oracle).
def invalid_color_greater_than(qc, problem, color, i, dest):
    """
    New version of ancilla marking for invalid colors on node i (using a greater-than oracle).
    """
    oracles.invalid_color_greater_than(qc, problem, color, i, dest)


def diffusion(qc, problem):
    for i in range(ancilla_index(problem)-1):
        qc.h(i)
        qc.x(i)
    qc.z(ancilla_index(problem) - 1)
    qc.mcx(list(range(ancilla_index(problem)-1)), ancilla_index(problem) - 1)
    qc.z(ancilla_index(problem) - 1)
    for i in range(ancilla_index(problem)-1):
        qc.x(i)
        qc.h(i)

'''
def get_components_list(qc, problem):
    components = []
    for color in range(k(problem), 2 ** qn(problem)):
        for i in range(n(problem)):
            components.append(
                lambda x, i=i, color=color:
                invalid_color(qc, problem, color, i, x))
    for i, j in graph(problem).edges:
        components.append(
            lambda x, i=i, j=j:
            comparator(qc, problem, i, j, x))
    return components


def make_components(qc, problem):
    arcs = [[] for i in range(n(problem))]
    for i, j in graph(problem).edges:
        def c(x, i=i, j=j):
            return comparator(qc, problem, i, j, x)
        arcs[i].append(((i, j), c))
        arcs[j].append(((i, j), c))

    colors = [[((i,),
                lambda x, i=i, color=color:
                invalid_color(qc, problem, color, i, x))
               for color in range(k(problem), 2 ** qn(problem))]
              for i in range(n(problem))]
    return arcs, colors
'''

#New version using greater-than oracle
def get_components_list(qc, problem):
    components = []
    for i in range(n(problem)):
        components.append(
            lambda x, i=i:
            invalid_color_greater_than(qc, problem, k(problem), i, x))
    for i, j in graph(problem).edges:
        components.append(
            lambda x, i=i, j=j:
            comparator(qc, problem, i, j, x))
    return components

#New version using greater-than oracle
def make_components(qc, problem):
    arcs = [[] for i in range(n(problem))]
    for i, j in graph(problem).edges:
        def c(x, i=i, j=j):
            return comparator(qc, problem, i, j, x)
        arcs[i].append(((i, j), c))
        arcs[j].append(((i, j), c))

    # Single component per node instead of per invalid color
    colors = [[((i,),
                lambda x, i=i:
                invalid_color_greater_than(qc, problem, k(problem), i, x))]
              for i in range(n(problem))]
    return arcs, colors

def arcs_comp(components):
    return components[0]


def colors_comp(components):
    return components[1]


def node_arcs_comp(components, i):
    return arcs_comp(components)[i]


def node_colors_comp(components, i):
    return colors_comp(components)[i]


def component_nodes(component):
    return component[0]


def component_builder(component):
    return component[1]


def is_arc(component):
    return len(component_nodes(component)) == 2


def comp_num(components):
    return (sum(1
                for node_color_comps in colors_comp(components)
                for i in node_color_comps)
            + int(sum(1
                      for node_arcs_comps in arcs_comp(components)
                      for i in node_arcs_comps)/2))


def remove_component(components, component):
    for i in component_nodes(component):
        if is_arc(component):
            node_arcs_comp(components, i).remove(component)
        else:
            node_colors_comp(components, i).remove(component)


def greedy_get_components_window(problem, components, amount):
    # sort nodes by number of components
    def ranking():
        return sorted(range(n(problem)),
                      key=lambda x: -len(node_arcs_comp(components, x)))

    def next_component(nodes):
        for node in nodes:
            for arc in node_arcs_comp(components, node):
                one, other = component_nodes(arc)
                if one in nodes and other in nodes:
                    return arc
            if len(node_colors_comp(components, node)):
                return node_colors_comp(components, node)[-1]

    window = []
    while len(window) < amount and comp_num(components) != 0:
        # gather fully-parallelized lists of components until we have
        # "amount" components in total
        nodes = ranking()  # prioritize nodes with most components
        while len(window) < amount:
            # choose as many components as possible to run in parallel
            chosen_component = next_component(nodes)
            if chosen_component is None:
                break
            window.append(chosen_component)
            remove_component(components, chosen_component)
            for i in component_nodes(chosen_component):
                nodes.remove(i)
    return window


def get_greedy_components_list(qc, problem):  # optimized for parallelism
    components = make_components(qc, problem)
    comp_list = greedy_get_components_window(problem,
                                             components,
                                             comp_num(components))
    return list(map(component_builder, comp_list))
