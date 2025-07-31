from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# gate == qiskit instruction
def gate_equal(gate1, gate2):
    return (gate1.operation.name == gate2.operation.name
            and gate1.qubits == gate2.qubits)


def gate_qubits(qc, gate):
    return list(map(lambda q: qc.find_bit(q).index, gate.qubits))


def gate_print(qc, gate):
    print(gate.operation.name, gate_qubits(qc, gate))


def gate_write_qubit(qc, gate):
    return max(gate_qubits(qc, gate))


def gate_read_qubits(qc, gate):
    qubits = gate_qubits(qc, gate)
    qubits.remove(max(qubits))
    return qubits


def make_lines(qc, num_qubits):
    return (qc, [[] for i in range(num_qubits)])


def line_gate(lines_q, q, i):
    qc, lines = lines_q
    return lines[q][i][1]


def line_index(lines_q, q, i):
    qc, lines = lines_q
    return lines[q][i][0]


def lines_add(lines_q, index, gate):
    qc, lines = lines_q
    for q in gate_qubits(qc, gate):
        lines[q].append((index, gate))


def lines_remove(lines_q, gate, gate_index):
    qc, lines = lines_q
    for q in gate_qubits(qc, gate):
        indexes = list(zip(*lines[q]))[0]
        i = indexes.index(gate_index)
        lines[q].pop(i)


def line_back_till(lines_q, q, circuit_index):
    qc, lines = lines_q
    for index, gate in reversed(lines[q]):
        if index <= circuit_index:
            break
        yield index, gate


def empty_line(lines_q, q):
    qc, lines = lines_q
    return len(lines[q]) == 0


# check for writes from i onwards on qubit q
def line_clean(lines_q, q, circuit_index):
    qc, lines = lines_q
    for index, gate in line_back_till(lines_q, q, circuit_index):
        if gate_write_qubit(qc, gate) == q:
            return False
    return True


def get_duplicate(lines_q, gate):
    qc, lines = lines_q
    if empty_line(lines_q, gate_write_qubit(qc, gate)):
        return None, None

    duplicate_index = line_index(lines_q, gate_write_qubit(qc, gate), -1)
    duplicate_gate = line_gate(lines_q, gate_write_qubit(qc, gate), -1)

    if (
            gate.operation.name != "h"
            and gate_equal(gate, duplicate_gate)
            and all(map(lambda q: line_clean(lines_q, q, duplicate_index),
                        gate_read_qubits(qc, gate)))):
        return duplicate_gate, duplicate_index
    return None, None


def deduplicate(qc):
    # previous gates as seen by each qubit
    lines = make_lines(qc, len(qc.qubits))

    for index, gate in enumerate(qc.data):
        duplicate_gate, duplicate_index = get_duplicate(lines, gate)
        if duplicate_gate is not None:
            gate.operation.label = "delenda"
            duplicate_gate.operation.label = "delenda"

            lines_remove(lines, duplicate_gate, duplicate_index)
        elif gate.operation.name != "barrier":
            lines_add(lines, index, gate)

    qc_opt = QuantumCircuit(QuantumRegister(qc.num_qubits),
                            ClassicalRegister(qc.num_clbits))

    for ins in qc.data:
        if ins.operation.label != "delenda":
            #  This is simpler than figuring out which arguments
            #  ins.operation.__class__() requires
            name = ins.operation.name
            qs = gate_qubits(qc, ins)
            if name == "h":
                qc_opt.h(qs[0])
            elif name == "z":
                qc_opt.z(qs[0])
            elif name == "x":
                qc_opt.x(qs[0])
            elif name == "cx":
                qc_opt.cx(qs[0], qs[1])
            elif name == "ccx":
                qc_opt.ccx(qs[0], qs[1], qs[2])
            elif name == "mcx_gray" or ins.operation.name == "mcx":
                qc_opt.mcx(qs[:-1], qs[-1])

    for i in range(qc.num_clbits):
        qc_opt.measure(i, i)

    return qc_opt
