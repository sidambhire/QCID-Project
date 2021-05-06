import math
import random
from numpy.random import choice

import numpy as np
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua.algorithms.amplitude_amplifiers.grover import Grover
from qiskit.circuit.library import QuadraticForm

from qiskit.optimization.algorithms.optimization_algorithm import OptimizationAlgorithm
from qiskit.optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit.optimization.problems.quadratic_program import QuadraticProgram


class GroverOptimizer(OptimizationAlgorithm):
    """Uses Grover Adaptive Search (GAS) to find the minimum of a QUBO function."""

    def __init__(self, num_value_qubits: int, num_iterations: int = 3):
        """
        Args:
            num_value_qubits: The number of value qubits.
            num_iterations: The number of iterations the algorithm will search with
                no improvement.
            quantum_instance: Instance of selected backend, defaults to Aer's statevector simulator.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit.optimization.converters.QuadraticProgramToQubo` will be used.
            penalty: The penalty factor used in the default
                :class:`~qiskit.optimization.converters.QuadraticProgramToQubo` converter

        Raises:
            TypeError: When there one of converters is an invalid type.
        """
        self._num_value_qubits = num_value_qubits
        self._num_key_qubits = None
        self._n_iterations = num_iterations
        #self._quantum_instance = Aer.get_backend('statevector_simulator')

        #self._converters = self._prepare_converters(converters, penalty)
        self._converters = QuadraticProgramToQubo()
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """
        must define abstract method from parent
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    def _get_a_operator(self, qr_key_value, problem):
        quadratic = problem.objective.quadratic.to_array()
        linear = problem.objective.linear.to_array()
        offset = problem.objective.constant

        # Get circuit requirements from input.
        quadratic_form = QuadraticForm(self._num_value_qubits, quadratic, linear, offset,
                                       little_endian=False)

        a_operator = QuantumCircuit(qr_key_value)
        a_operator.h(list(range(self._num_key_qubits)))
        a_operator.compose(quadratic_form, inplace=True)
        return a_operator

    def _get_oracle(self, qr_key_value):
        # Build negative value oracle O.
        if qr_key_value is None:
            qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)

        oracle_bit = QuantumRegister(1, "oracle")
        oracle = QuantumCircuit(qr_key_value, oracle_bit)
        oracle.z(self._num_key_qubits)  # recognize negative values.

        def is_good_state(self, measurement):
            """Check whether ``measurement`` is a good state or not."""
            value = measurement[self._num_key_qubits:self._num_key_qubits + self._num_value_qubits]
            return value[0] == '1'

        return oracle, is_good_state

    def solve(self, problem: QuadraticProgram):
        """Tries to solves the given problem using the grover optimizer.

        Runs the optimizer to try to solve the optimization problem. If the problem cannot be,
        converted to a QUBO, this optimizer raises an exception due to incompatibility.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            AttributeError: If the quantum instance has not been set.
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """

        # convert problem to QUBO
        problem_ = self._convert(problem, self._converters)
        #problem_ = problem
        #problem_ = QuadraticProgramToQubo().convert(problem_)

        # convert to minimization problem

        self._num_key_qubits = len(problem_.objective.linear.to_array())  # type: ignore

        # Variables for tracking the optimum.
        optimum_found = False
        optimum_key = math.inf
        optimum_value = math.inf
        threshold = 0
        n_key = len(problem_.variables)
        n_value = self._num_value_qubits

        # Variables for tracking the solutions encountered.

        # Variables for result object.

        # Variables for stopping if we've hit the rotation max.
        rotations = 0

        # Initialize oracle helper object.
        qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)
        orig_constant = problem_.objective.constant
        oracle, is_good_state = self._get_oracle(qr_key_value)

        while not optimum_found:
            m = 1
            improvement_found = False

            # Get oracle O and the state preparation operator A for the current threshold.
            problem_.objective.constant = orig_constant - threshold
            a_operator = self._get_a_operator(qr_key_value, problem_)

            # Iterate until we measure a negative.
            loops_with_no_improvement = 0
            while not improvement_found:
                # Determine the number of rotations.
                loops_with_no_improvement += 1
                #rotation_count = int(np.ceil(aqua_globals.random.uniform(0, m - 1)))
                rotation_count = random.randint(0, m - 1)
                rotations += rotation_count
                # Apply Grover's Algorithm to find values below the threshold.
                # TODO: Utilize Grover's incremental feature - requires changes to Grover.
                grover = Grover(oracle,
                                state_preparation=a_operator,
                                good_state=is_good_state)
                circuit = grover.construct_circuit(rotation_count, True)
                
                simulator = Aer.get_backend('aer_simulator')
                circ = transpile(circuit, simulator)
                result = simulator.run(circ).result()
                #result = self._quantum_instance.execute(circuit)
                temp = result.get_counts(circ)
                print(temp)
                draw = choice(list(temp.keys()))
                outcome = draw[::-1]
                # Get the next outcome.
                #outcome = self._measure(circuit)
                print(outcome, "\n")
                k = int(outcome[0:n_key], 2)
                v = outcome[n_key:n_key + n_value]
                int_v = self._bin_to_int(v, n_value) + threshold
                print(k, " ", v, " ", int_v)

                # If the value is an improvement, we update the iteration parameters (e.g. oracle).
                if int_v < optimum_value:
                    optimum_key = k
                    optimum_value = int_v
                    improvement_found = True
                    threshold = optimum_value
                else:
                    # Using Durr and Hoyer method, increase m.
                    m = int(np.ceil(min(m * 8 / 7, 2 ** (n_key / 2))))

                    # Check if we've already seen this value.

                    # Assume the optimal if any of the stop parameters are true.
                    if loops_with_no_improvement >= 10:
                        improvement_found = True
                        optimum_found = True

                # Track the operation count.

        # If the constant is 0 and we didn't find a negative, the answer is likely 0.
        if optimum_value >= 0 and orig_constant == 0:
            optimum_key = 0

        opt_x = np.array([1 if s == '1' else 0 for s in ('{0:%sb}' % n_key).format(optimum_key)])
        print(opt_x)
        return 


    @staticmethod
    def _bin_to_int(v: str, num_value_bits: int) -> int:
        """Converts a binary string of n bits using two's complement to an integer."""
        if v.startswith("1"):
            int_v = int(v, 2) - 2 ** num_value_bits
        else:
            int_v = int(v, 2)

        return int_v

