import json
import pennylane as qml
import pennylane.numpy as np

""" Sourced from Pennylane Challenge: 'An Itch to Switch' (https://pennylane.ai/challenges/an_itch_to_switch)"""

""" Explanation: We know that there are 4 distinct bell states, which, when a specific 2-qubit operator is applied, 
can result in 4 seperate states (|00>, |01>, |10>, and |11>). Knowing this, we can purposefully apply a control gate 
that maps each different initial state to a different bell state, eg. ZR|0>|phi+>|0> = |0>|phi+>|0>, 
ZR|0>|phi+>|1> = |0>|phi->|1>, ZR|1>|phi+>|0> = |1>|psi+>|0>, ZR|1>|phi+>|1> = |1>|psi->|1>. 

We can see that if we apply the pauli_z gate to one of the qubits in the phi+ bell state it will change the sign of the
|11> term, thus flipping the state to |phi->. If the pauli_x gate is applied to to one of the qubits in the phi+ bell 
state, then both states will have a 1-qubit flip, resulting in the psi+ state. Finally, if pauli_x is applied to one 
qubit, and pauli_z is applied to the other, then there is a bit flip and a sign change, leading to the psi- state. 
This give us 4 individual operations to apply to the bell state to make it differentiable: apply pauli_x on qubit m, 
apply pauli_z on qubit n, apply both pauli_x and and pauli_z on qubit m and n, respectively, or do nothing. 
We can then see that these can be controlled by the 4 possible basis states of z0 and r0. 
 
Lets let zenda's operator be the control-X gate, with the z0 controlling the z1 state. Therefore reeces operator must
be the control-Z operator by our logic above. this provides the following transformations:

ZR|0>|phi+>|0> = |0>|phi+>|0>
ZR|0>|phi+>|1> = |0>|phi->|1>
ZR|1>|phi+>|0> = |1>|psi+>|0>
ZR|1>|phi+>|1> = |1>|psi->|1>

Just as predicted from teh logic above. From this we can create a matrix that will transform each bell state to the
states indicated by z0 and r0. 

M|phi+> = |00>
M|phi-> = |10>
M|psi+> = |01>
M|psi-> = |11>

This is a simple linear algebra problem, giving the matrix:

M = 1/sqrt(2)   [ 1 0  0  1]
                [ 1 0  0 -1]
                [ 0 1  1  0]
                [ 0 1 -1  0]

Thus, we give zenda the CX operator, give reece the CZ operator, and then apply the M operator described above.
"""

def zenda_operator():
    """
    Quantum function corresponding to the operator to be applied by
    Zenda in her qubits.This function does not return anything,
    you must simply write the necessary gates.
    """

    qml.CNOT(["z0", "z1"])

def reece_operator():
    """
    Quantum function corresponding to the operator to be applied by
    Reece in his qubits.This function does not return anything,
    you must simply write the necessary gates.
    """

    qml.CZ(["r0", "r1"])

def magic_operator():
    """
    Quantum function corresponding to the operator to be applied on the "z1"
    and "r1" qubits. This function does not return anything, you must
    simply write the necessary gates.

    """

    M = 1 / np.sqrt(2) * np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]])
    qml.QubitUnitary(M, wires=["r1", "z1"])


def bell_generator():
    """
    Quantum function preparing bell state shared by Reece and Zenda.
    """

    qml.Hadamard(wires=["z1"])
    qml.CNOT(wires=["z1", "r1"])


dev = qml.device("default.qubit", wires=["z0", "z1", "r1", "r0"])


@qml.qnode(dev)
def circuit(j, k):
    bell_generator()

    # j encoding and Zenda operation
    qml.BasisEmbedding([j], wires="z0")
    zenda_operator()

    # k encoding and Reece operation
    qml.BasisEmbedding([k], wires="r0")
    reece_operator()

    magic_operator()

    return qml.probs(wires=dev.wires)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    return None


def check(solution_output: str, expected_output: str) -> None:
    try:
        dev1 = qml.device("default.qubit", wires=["z0", "z1"])

        @qml.qnode(dev1)
        def circuit1():
            zenda_operator()
            return qml.probs(dev1.wires)

        circuit1()
    except:
        assert False, "zenda_operator can only act on z0 and z1 wires"

    try:
        dev2 = qml.device("default.qubit", wires=["r0", "r1"])

        @qml.qnode(dev2)
        def circuit2():
            reece_operator()
            return qml.probs(dev2.wires)

        circuit2()
    except:
        assert False, "reece_operator can only act on r0 and r1 wires"
    try:
        dev3 = qml.device("default.qubit", wires=["z1", "r1"])

        @qml.qnode(dev3)
        def circuit3():
            magic_operator()
            return qml.probs(dev3.wires)

        circuit3()
    except:
        assert False, "magic_operator can only act on r1 and z1 wires"

    for j in range(2):
        for k in range(2):
            assert np.isclose(circuit(j, k)[10 * j + 5 * k], 1), "The output is not correct"


# These are the public test cases
test_cases = [
    ('No input', 'No output')
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