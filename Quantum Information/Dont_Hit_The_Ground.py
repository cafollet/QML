import json
import pennylane as qml
import pennylane.numpy as np

""" Sourced from Pennylane Challenge: 'Don't Hit the Ground' (https://pennylane.ai/challenges/dont_hit_the_ground)"""


def half_life(gamma, p):
    """Calculates the relaxation half-life of a quantum system that exchanges energy with its environment.
    This process is modeled via Generalized Amplitude Damping.

    Args:
        gamma (float):
            The probability per unit time of the system losing a quantum of energy
            to the environment.
        p (float): The de-excitation probability due to environmental effect

    Returns:
        (float): The relaxation haf-life of the system, as explained in the problem statement.
    """

    num_wires = 1

    dev = qml.device("default.mixed", wires=num_wires)

    # Creating the GAD channel circuit
    @qml.qnode(dev)
    def gad_circuit(g, pr, delta_t, repeat):
        qml.Hadamard(wires=0)
        for _ in range(repeat):
            gamma_t = g*delta_t
            qml.GeneralizedAmplitudeDamping(gamma_t, p, wires=0)
        return qml.probs(wires=0)

    # Initializing all loop variables
    prob = [0.5, 0.5]
    del_t = 0.01  # Adjust this to increase runtime, but decrease accuracy
    T = 0
    n = 0
    print("\t\t", "|0>\t\t", "|1>\t\t","Time:")

    # While-loop that iterates through time evolutions, simulating that
    # evolution by running the circuit with n = t/del_t GAD channels
    while prob[1] > 0.25:
        prob = gad_circuit(gamma, p, del_t, n)

        # Print progress (for potential benchmarking?)
        if T - int(T) < del_t:
            print("\t\t", round(prob[0]*100), "%\t\t", round(prob[1]*100), "%\t\t", round(T), "s")

        T += del_t
        n += 1
    return T



# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = half_life(*ins)

    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, atol=2e-1
    ), "The relaxation half-life is not quite right."

#  ('[0.1,0.92]', '9.05')
# These are the public test cases
test_cases = [
    ('[0.1,0.92]', '9.05'),
    ('[0.2,0.83]', '7.09')
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