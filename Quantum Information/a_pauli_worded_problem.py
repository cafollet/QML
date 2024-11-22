import json
import logging

import pennylane as qml
import pennylane.numpy as np
import scipy

""" Sourced from Pennylane Challenge: 'A Pauli-Worded Problem' (https://pennylane.ai/challenges/pauli_worded)"""

def abs_dist(rho, sigma):
    """A function to compute the absolute value |rho - sigma|."""
    polar = scipy.linalg.polar(rho - sigma)
    return polar[1]

def word_dist(word):
    """A function which counts the non-identity operators in a Pauli word"""
    return sum(word[i] != "I" for i in range(len(word)))


# Produce the Pauli density for a given Pauli word and apply noise

def noisy_Pauli_density(word, lmbda):
    """
       A subcircuit which prepares a density matrix (I + P)/2**n for a given Pauli
       word P, and applies depolarizing noise to each qubit. Nothing is returned.

    Args:
            word (str): A Pauli word represented as a string with characters I,  X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.
    """
    p_matrix = [[1]]  # Initialize Pauli Tensor Product for for-loop

    # For-loop to iterate through the string and find matrix representation of the Pauli Word
    for i, letter in enumerate(word):

        # if I, apply Identity operator
        if letter == "I":
            matrix = np.array([[1., 0.], [0., 1.]])

        # if X, apply Pauli X
        elif letter == "X":
            matrix = np.array([[0., 1.], [1., 0.]])

        # if Y, apply Pauli Y
        elif letter == "Y":
            matrix = np.array([[0., -1.j], [1.j, 0.]], dtype=complex)

        # if Z, apply Pauli Z
        elif letter == "Z":
            matrix = np.array([[1., 0.], [0., -1.]])

        # if other character, raise exception
        else:
            logging.exception("Pauli Word not correct")
            exit()

        # Calculate the tensor product
        p_matrix = np.kron(matrix, p_matrix)

    # Calculate the Pauli density matrix
    d_matrix = (1/(2**len(word))) * (np.identity(2**len(word)) + p_matrix)

    # Apply density matrix
    qml.QubitDensityMatrix(d_matrix, wires=range(len(word)))

    # Apply the Depolarizing Channel to each Qubit
    for i, letter in enumerate(word):
        qml.DepolarizingChannel(lmbda, wires=i)
    return None


# Compute the trace distance from a noisy Pauli density to the maximally mixed density

def maxmix_trace_dist(word, lmbda):
    """
       A function compute the trace distance between a noisy density matrix, specified
       by a Pauli word, and the maximally mixed matrix.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The trace distance between two matrices encoding Pauli words.
    """
    n_wires = len(word)  # no. of qubits
    bin_n = 2 ** n_wires  # no. of possible states

    # Defining the quantum circuit
    dev = qml.device("default.mixed", wires=n_wires)
    @qml.qnode(dev)
    def get_rho(word, lmbda):
        noisy_Pauli_density(word, lmbda)
        return qml.density_matrix(range(n_wires))


    p_0 = np.identity(bin_n) / bin_n
    p = get_rho(word, lmbda)

    # Calculating trace distance
    T = abs_dist(p, p_0)
    T = 0.5 * (np.trace(T))
    return T

def bound_verifier(word, lmbda):
    """
       A simple check function which verifies the trace distance from a noisy Pauli density
       to the maximally mixed matrix is bounded by (1 - lambda)^|P|.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The difference between (1 - lambda)^|P| and T(rho_P(lambda), rho_0).
    """
    # calculating trace distance, which is the RHS of the verifier inequality
    T = maxmix_trace_dist(word, lmbda)

    # print("T:", T)  # Trace distance console output, for debugging

    # Calculating the LHS of the verifier inequality
    abs_P = word_dist(word)
    ver = 1 - lmbda
    ver = ver**abs_P

    # print("ver:",ver)  # LHS value console output, for debugging

    return ver-T

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    word, lmbda = json.loads(test_case_input)
    output = np.real(bound_verifier(word, lmbda))

    return str(output)


def check(solution_output: str, expected_output: str) -> None:

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your trace distance isn't quite right!"

# These are the public test cases
test_cases = [
    ('["XXI", 0.7]', '0.0877777777777777'),
    ('["XXIZ", 0.1]', '0.4035185185185055'),
    ('["YIZ", 0.3]', '0.30999999999999284'),
    ('["ZZZZZZZXXX", 0.1]', '0.22914458207245006')
]
# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")