import json
import pennylane as qml
from pennylane import numpy as np

""" Sourced from Pennylane Challenge: 'Mid-Circuit Measurements' 
(https://pennylane.ai/challenges/mid_circuit_measurements)"""


n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(angles):
    """A quantum circuit made from the quantum function U.

    Args:
        angles (list(float)): A list of angles containing theta_0, theta_1, theta_2, and theta_3 in that order.
    Returns:
        (numpy.tensor): The probability of the fourth qubit.
    """

    # Apply Hadamard to all |0> states
    for wire in range(4):
        qml.Hadamard(wires=wire)

    # Rotation of theta_0 around x
    qml.RX(angles[0], wires=0)

    # CNOT Gates
    qml.CNOT(wires=(0, 3))
    qml.CNOT(wires=(2, 1))

    # Mid-Circuit measure
    m_0 = qml.measure(wires=0)
    m_2 = qml.measure(wires=2)

    # Creating the Conditional Unitary Operator
    theta_1, theta_2, theta_3 = tuple(angles[1:])
    qml.cond(m_0 + m_2 >= 1, qml.U3)(theta_1, theta_2, theta_3, wires=3)

    qml.PauliZ(wires=3)
    return qml.probs(wires=3)

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    angles = json.loads(test_case_input)
    output = circuit(angles).tolist()
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-4)

# These are the public test cases
test_cases = [
    ('[1.0, 1.5, 2.0, 2.5]', '[0.79967628, 0.20032372]')
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