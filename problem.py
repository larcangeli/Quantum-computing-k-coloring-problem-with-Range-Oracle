from math import ceil, log2
from platform import node
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


def problem_comp_num(problem):
    return (invalid_colors(problem) * n(problem)
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


def oracle_same_color_penalty(qc, problem, i, j):
    """
    Apply a phase penalty to configurations where nodes i,j have same color
    """
    
    a = node_qubits(problem, i)
    b = node_qubits(problem, j)
    
    # XOR the color bits (same as current comparator logic)
    for qi, qj in zip(a, b):
        qc.cx(qi, qj)
        qc.x(qj)
    
    # Now b contains the XOR result - all zeros if colors are equal
    # Apply a controlled phase when all b bits are zero
    qc.append(oracles.multi_control_z(len(b)).to_gate(), b)
    
    # Undo the XOR operation
    for qi, qj in reversed(list(zip(a, b))):
        qc.x(qj)
        qc.cx(qi, qj)



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

def invalid_color_lt_oracle(qc, problem, color, i, dest):
    node_bits = node_qubits(problem, i)
    k_value = k(problem)
    less = oracles.oracle_less_than(color, qn(problem))
    qc.append(less, node_bits)

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

def diffusion_operator(qc, problem, theta):
    """Applies the diffusion operator adapted for non-uniform superposition using adaptive Grover rotations."""
    n_qubits = total_qubits(problem) - 1
    
    for i in range(n_qubits):
        qc.h(i)
        qc.x(i)
    
    qc.rz(2 * theta, n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.rz(2 * theta, n_qubits - 1)
    
    for i in range(n_qubits):
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
'''
#updated function using the new oracle-based comparator and invalid color functions
def get_components_list(qc, problem):
    components = []
    for color in range(k(problem), 2 ** qn(problem)):
        for i in range(n(problem)):
            components.append(
                lambda x, i=i, color=color:
                invalid_color_lt_oracle(qc, problem, color, i, x))
    for i, j in graph(problem).edges:
        components.append(
            lambda x, i=i, j=j:
            oracle_same_color_penalty(qc, problem, i, j))
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
