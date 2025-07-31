# Imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, UnitaryGate

import numpy as np

def diffuser(n_qubits):
    """
    Standard diffuser (inversion about the average) for n_qubits.
    """
    
    qc = QuantumCircuit(n_qubits, name=' Diffuser ')
    # Apply Hadamard gates to all qubits
    qc.h(range(n_qubits))
    # Apply X gates to all qubits
    qc.x(range(n_qubits))
    # Apply multi-controlled Z gate (equivalent to inversion about the average)
    qc.append(multi_control_z(n_qubits), range(n_qubits))
    # Apply X gates to all qubits
    qc.x(range(n_qubits))
    # Apply Hadamard gates to all qubits
    qc.h(range(n_qubits))

    return qc

def to_binary(number, nbits=None):
    
    '''
    This fucntion transforms an integer to its binary form (string).
    If a determined number of bits is required (more than the needed ones),
    it can be passed as a parameter too, nbits, None by default.
    It is needed that the number of bits passed as a parameter is larger
    than the number of bits needed to write the number in binary. 

    Input:
    number: integer (int).
    nbits: integer (int), None by default

    Output:
    binary: string (str) containing the number in its binary form.
    It writes 0s in front if nbits is larger than the number of bits needed
    to write the binary form.
    '''

    if nbits is None:
        return bin(number)[2:]
    else:
        binary = bin(number)[2:]
        if nbits < len(binary):
            print('Error, nbits must be larger than %d.'%(len(binary)))
        else:
            return '0' * (nbits - len(binary)) + binary


def multi_control_z(nqubits):
    '''
    Function to create a multi-controlled Z gate.

    Input:
    nqubits: Integer (int) of the number of qubits in the gate (controls and target)
       This means that the gate has nqubits-1 controls and 1 target.

    Output:
    circuit: QuantumCircuit containing a multi-controlled Z gate.
      It has to be transformed with method .to_gate() to append to a QuantumCircuit larger.

    Example:

    main_circuit = QuantumCircuit(nqubits)

    gate_multi_z = multi_control_z(nqubits)

    main_circuit.append(gate_multi_z.to_gate(), range(nqubits))
    '''
    circuit=QuantumCircuit(nqubits,name=' CZ (%d)' %(nqubits))
    circuit.h(nqubits-1)
    gate = MCXGate(nqubits-1)
    circuit.append(gate, range(nqubits))
    circuit.h(nqubits-1)
    return circuit


def oracle_less_than(number, nqubits, name=None):

    '''
    This function builds a quantum circuit, an oracle, which marks with a pi-phase
    those states which represent numbers strictly smaler than the number given by parameter.

    The procedure is almost the same for all numbers, with the only exception of a difference
    if the first bit of the number in binary is 1 or 0.

    Input:
    number: integer (int) containing the objective number,
       or a string (str) with the binary representation of such number.
    nqubits: integer (int) number of qubits of the circuit.
       It must be larger than the number of digits of the binary representation of number.
    name: string (str), default None, name of the circuit.

    Output:
    circuit: QuantumCircuit which marks with fase pi the states which
    represent in binary the numbers strictly smaller than number.
    '''

    # Construction of the circuit
    if name:# If name is provided give such name to the circuit
        circuit = QuantumCircuit(nqubits, name=name)
    else: # Otherwise, the name is just " < number"
        circuit = QuantumCircuit(nqubits, name = ' < %d '%number)

    # Binary representation of the number
    num_binary = to_binary(number, nqubits)
    
    # Discard the 0s at the end, as they will not be used and save
    # unnecessary X gates
    num_binary = num_binary.rstrip('0')
    
    if num_binary[0] == '1':
        # If the first digit is 1
        # Mark all the states of the form |0q1...>
        circuit.x(nqubits-1)
        circuit.z(nqubits-1)
        circuit.x(nqubits-1)
    else:
        # If first digit is 0
        # Apply X gate to first qubit
        circuit.x(nqubits-1)
    
    # For loop on the remaining digits  
    for position1, value in enumerate(num_binary[1:]):
        # Rename the position as it starts with 0 in the second bit and
        # we want it to be 1.
        position = position1 + 1

        if value == '0':
            # If the digit is 0
            # Just apply a X gate
            circuit.x(nqubits-1)
        else:
            # If the digit bi is 1
            # Apply a multi-controlled Z gate to mark states of the shape:
            # |bn...bi+1 0 qi-1...q1>
            # where bn,...,bi+1 are the first n-i bits of m, which is of the shape bn...bi+1 1 bi-1...b1
            # because we just checked that bi is 1.
            # Hence, the numbers of the form bn...bi+1 0 qi-1...q1 are smaller than m.                
            circuit.x(nqubits-position-1)
            multi_z = multi_control_z(position + 1)
            circuit.append(multi_z.to_gate(), range(nqubits-1, nqubits-position-2, -1))
            circuit.x(nqubits-position-1)

    
    for position, value in enumerate(num_binary):
        # Apply X gates to qubits in position of bits with a 0 value
        if value == '0':
            circuit.x(nqubits-position-1)
        else:
            pass
    
    return circuit

def oracle_less_than_sandwich(number, nqubits, name=None):

    '''
    This function builds a quantum circuit, an oracle, which marks with a pi-phase
    those states which represent numbers strictly smaler than the number given by parameter.

    The procedure is almost the same for all numbers, with the only exception of a difference
    if the first bit of the number in binary is 1 or 0.

    Input:
    number: integer (int) containing the objective number,
       or a string (str) with the binary representation of such number.
    nqubits: integer (int) number of qubits of the circuit.
       It must be larger than the number of digits of the binary representation of number.
    name: string (str), default None, name of the circuit.

    Output:
    circuit: QuantumCircuit which marks with fase pi the states which
    represent in binary the numbers strictly smaller than number.
    '''

    # Construction of the circuit
    if name:# If name is provided give such name to the circuit
        circuit = QuantumCircuit(nqubits, name=name)
    else: # Otherwise, the name is just " < number"
        circuit = QuantumCircuit(nqubits, name = ' < %d '%number)

    # Binary representation of the number
    num_binary = to_binary(number, nqubits)
    
    # Discard the 0s at the end, as they will not be used and save
    # unnecessary X gates
    num_binary = num_binary.rstrip('0')

    zero_bits = []
    
    if num_binary[0] == '1':
        # If the first digit is 1
        # Mark all the states of the form |0q1...>
        circuit.x(nqubits-1)
        circuit.z(nqubits-1)
        circuit.x(nqubits-1)
    else:
        # If first digit is 0
        # Apply X gate to first qubit
        zero_bits.append(nqubits-1)
    
    # For loop on the remaining digits  
    for position1, value in enumerate(num_binary[1:]):
        # Rename the position as it starts with 0 in the second bit and
        # we want it to be 1.
        position = position1 + 1

        if value == '0':
            # If the digit is 0
            # Just apply a X gate
            zero_bits.append(nqubits-position-1)
        else:
            # If the digit bi is 1
            # Apply a multi-controlled Z gate to mark states of the shape:
            # |bn...bi+1 0 qi-1...q1>
            # where bn,...,bi+1 are the first n-i bits of m, which is of the shape bn...bi+1 1 bi-1...b1
            # because we just checked that bi is 1.
            # Hence, the numbers of the form bn...bi+1 0 qi-1...q1 are smaller than m.
            for qbit in zero_bits:
                circuit.x(qbit)
                
            circuit.x(nqubits-position-1)
            multi_z = multi_control_z(position + 1)
            circuit.append(multi_z.to_gate(), range(nqubits-1, nqubits-position-2, -1))
            circuit.x(nqubits-position-1)
            
            for qbit in zero_bits:
                circuit.x(qbit)
    
    return circuit

def oracle_less_than_sandwich_v1(number: int, nqubits: int) -> QuantumCircuit:
    """
    Creates a CORRECT quantum oracle that marks states |x> where x < number.
    This version FIXES the qubit ordering bug.
    """
    circuit = QuantumCircuit(nqubits, name=f"< {number}")
    num_binary = format(number, f'0{nqubits}b')

    for i, bit in enumerate(num_binary):
        
        if bit == '0':
            continue

        target_qubit = nqubits - 1 - i
        
        # The prefix is all bits before the current one.
        prefix = num_binary[:i]
        prefix_qubits = [nqubits - 1 - j for j in range(i)]
        
        # Top Slice: Wrap the prefix bits in X-gates where the prefix has a '0'.
        prefix_x_wrap_qubits = [nqubits - 1 - j for j, prefix_bit in enumerate(prefix) if prefix_bit == '0']
        if prefix_x_wrap_qubits:
            circuit.x(prefix_x_wrap_qubits)
            
        circuit.x(target_qubit)

        # The controls are the prefix qubits. The target is the current bit.
        mcz_qubits = prefix_qubits + [target_qubit]
        
        if len(mcz_qubits) > 1:
            mcz_gate = multi_control_z(len(mcz_qubits))
            circuit.append(mcz_gate.to_gate(), mcz_qubits)
        else:
            circuit.z(mcz_qubits[0])

        circuit.x(target_qubit)
        if prefix_x_wrap_qubits:
            circuit.x(prefix_x_wrap_qubits)
            
    return circuit


#################################################
###### ORACLES BUILT WITH LESS-THAN ORACLE ######
#################################################

def my_oracle_greater_than(number, nqubits, name=None):

    
    # Construction of the circuit
    if name:# If name is provided give such name to the circuit
        circuit = QuantumCircuit(nqubits, name=name)
    else: # Otherwise, the name is just " < number"
        circuit = QuantumCircuit(nqubits, name = ' < %d '%number)

    # Binary representation of the number
    num_binary = to_binary(number, nqubits)

    
    if num_binary[0] == '0':
        circuit.z(nqubits-1)
        circuit.x(nqubits-1)


    
    # For loop on the remaining digits
    for position1, value in enumerate(num_binary[1:]):
        position = position1 + 1

        if value == '0':
            print("flipping at position %d ",  position)
            multi_z = multi_control_z(position + 1)
            circuit.append(multi_z.to_gate(), range(nqubits-1, nqubits-position-2, -1))
            circuit.x((nqubits-1)-position)


    for position, value in enumerate(num_binary):
        # Apply X gates to qubits in position of bits with a 0 value
        if value == '0':
            circuit.x(nqubits-position-1)
        else:
            pass
    
    return circuit

def oracle_greater_than(number, nqubits, name=None):
    '''
    This function builds a quantum circuit, an oracle, which marks with a pi-phase
    those states which represent numbers strictly larger than the number given by parameter.

    The procedure uses the function oracle_less_than and introduces a global phase.

    Input:
    number: integer (int) containing the objective number,
       or a string (str) with the binary representation of such number.
    nqubits: integer (int) number of qubits of the circuit.
       It must be larger than the number of digits of the binary representation of number.
    name: string (str), default None, name of the circuit.

    Output:
    circuit: QuantumCircuit which marks with fase pi the states which
    represent in binary the numbers strictly greater than number.
    '''
    if name is None:
        name = ' > %d'%(number)
    else:
        name = name
    
    circuit = oracle_less_than(number=number+1, nqubits=nqubits, name=name)

    circuit.z(0)
    circuit.x(0)
    circuit.z(0)
    circuit.x(0)
    
    return circuit
    


def oracle_interval(lower_boundary, upper_boundary, nqubits, name=None):

    '''
    This function builds a quantum circuit, an oracle, which marks with a pi-phase
    those states which represent numbers in the interval (lower_boundary, upper_boundary).

    The procedure uses the function oracle_less_than and oracle_greater_than.

    Input:
    lower_boundary: integer (int) containing the lower boundary,
       or a string (str) with the binary representation of such number.
    upper_boundary: integer (int) containing the upper boundary,
       or a string (str) with the binary representation of such number.
    nqubits: integer (int) number of qubits of the circuit.
       It must be larger than the number of digits of the binary representation of number.
    name: string (str), default None, name of the circuit.

    Output:
    circuit: QuantumCircuit which marks with fase pi the states which
    represent in binary the numbers in the interval (lower_boundary, upper_boundary).
    '''

    if name is None:
        name = '(%d,%d)'%(lower_boundary, upper_boundary)
    else:
        name = name

    circuit = QuantumCircuit(nqubits, name=name)

    circuit.append(oracle_less_than(number=upper_boundary, nqubits=nqubits), range(nqubits))
    circuit.append(oracle_greater_than(number=lower_boundary, nqubits=nqubits), range(nqubits))
    
    circuit.z(0)
    circuit.x(0)
    circuit.z(0)
    circuit.x(0)

    return circuit


#############################################
###### ORACLES BUILT WITH UNITARY GATE ######
#############################################

def matrix_less_than(number:int, N:int):
    '''
    Creates a squared matrix of the form
    -1 0 ... 0 0 ... 0
    0 -1 ... 0 0 ... 0
    .
    .
    .     0 -1 0 ... 0
             0 1 ... 0
                     .


                    1
    With 0s outside of the diagonal, and the first $number elements
    of the diagonal are -1 and the rest are 1.

    Input:
        - number (int): number for less than oracle
        - N (int): Total size of the squared matrix, with N = 2^nqubits.
    
    Output:
        - matrix (np.array): diagonal matrix with first $number elements are -1
    '''
    v = np.array([-1]*number + [1]*(N-number))
    matrix = np.diag(v)
    return matrix


def unitary_oracle_less_than(number:int, nqubits:int, name=None):

    '''
    This function builds a quantum circuit, an oracle, which marks with a pi-phase
    those states which represent numbers strictly smaler than the number given by parameter
    by using the method UnitaryGate from Qiskit.

    Input:
    number: integer (int) containing the objective number,
       or a string (str) with the binary representation of such number.
    nqubits: integer (int) number of qubits of the circuit.
       It must be larger than the number of digits of the binary representation of number.
    name: string (str), default None, name of the circuit.

    Output:
    circuit: QuantumCircuit which marks with fase pi the states which
    represent in binary the numbers strictly smaller than number.
    '''

    # Construction of the circuit
    if name:# If name is provided give such name to the circuit
        circuit = QuantumCircuit(nqubits, name=name)
    else: # Otherwise, the name is just " < number Uni"
        circuit = QuantumCircuit(nqubits, name = ' < %d Uni'%number)
    
    
    circuit.append(UnitaryGate(matrix_less_than(number, 2**nqubits)), range(nqubits))

    return circuit




################################################
###### CALCULATE DEPTH WITH GIVEN BACKEND ######
################################################

def decompose_circuit(circuit:QuantumCircuit, backend, reps:int=50, opt_level=3):
    '''
    Decomposes a circuit according to a backend.

    Input:
        - circuit (QuantumCircuit): circuit to be decomposed

        - backend: backend to use. It may be a str (when statevector), or a backend object

        - reps (int): number of repetitions in decompose. By default uses 50 which is
        usually enough to decompose any circuit.

        - opt_level (int in 0, 1, 2, 3): optimization level of the circuit. By default 3 to
        obtain the maximum optimization.
    '''

    circuit_transpiled = transpile(circuits=circuit, backend=backend, optimization_level=opt_level)
    circuit_decomposed = circuit_transpiled.decompose(reps=reps)
    
    return circuit_decomposed