#!/bin/python
#Taken from https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html and modified
# initialization
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile

# import basic plot tools
from qiskit.visualization import plot_histogram


def oracle(val, n):
    # We need to make a QuantumCircuit object to return
    # This circuit has n+1 qubits: the size of the input,
    # plus one output qubit
    oracle_qc = QuantumCircuit(n+1)
    
    # Next, format 'val' as a binary string of length 'n', padded with zeros:
    b_str = format(val, '0'+str(n)+'b')
    # Next, we place the first X-gates. Each digit in our binary string 
    # corresponds to a qubit, if the digit is 0, we do nothing, if it's 1
    # we apply an X-gate to that qubit:
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            oracle_qc.x(qubit)
    # Do the controlled-NOT gates for each qubit, using the output qubit 
    # as the target:
    for qubit in range(n):
        oracle_qc.cx(qubit, n)
    # Next, place the final X-gates
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            oracle_qc.x(qubit)

    
    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle" # To show when we display the circuit
    return oracle_gate
    
#TODO: Implementation is being tested
def grover_algorithm(oracle, n):
    grover_circuit = QuantumCircuit(n+1, n)
    # Set up the output qubit:
    grover_circuit.x(n)
    grover_circuit.h(n)
    # And set up the input register:
    for qubit in range(n):
        grover_circuit.h(qubit)
    # Let's append the oracle gate to our circuit:
    for iteration in range(int(np.sqrt(n))):
        grover_circuit.append(oracle, range(n+1))
    # Finally, perform the H-gates again and measure:
    for qubit in range(n):
        grover_circuit.h(qubit)
    
    for i in range(n):
        grover_circuit.measure(i, i)
    
    return grover_circuit
