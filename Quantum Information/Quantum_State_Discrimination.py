import json
import pennylane as qml
import pennylane.numpy as np

""" Sourced from Pennylane Challenge: 'Quantum State Discrimination' 
(https://pennylane.ai/challenges/quantum_state_discrimination)"""


def maximal_probability(theta_1, theta_2, p_1, p_2):
    """
    This function calculates the maximal probability of distinguishing
    the states

    Args:
        theta_1 (float): Angle parametrizing the state |phi_1>.
        theta_2 (float): Angle parametrizing the state |phi_2>.
        p_1 (float): Probability that the state was |phi_1>.
        p_2 (float): Probability that the state was |phi_2>.

    Returns:
        (Union[float, np.tensor]): Maximal probability of distinguishing the states.

    """

    # create a circuit which applies a rotation gate to a state
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def circuit(phi_state, rotation_lambda):
        # Prepare the state
        qml.StatePrep(phi_state, wires=0)
        # Apply the rotation matrix to optimize
        qml.RY(rotation_lambda, wires=0)
        return qml.probs(0)

    # Defining the cost function to minimize
    def cost(rot_lambdas, phis):
        rot_lambda = rot_lambdas[0]
        phi_1 = phis[0]
        phi_2 = phis[1]
        # Run the circuit defined above for both initial states, and retrieve the Probabilities of achieving each state
        sig_1 = circuit(phi_1, rot_lambda)
        sig_2 = circuit(phi_2, rot_lambda)
        q_1 = sig_1[0]
        q_2 = sig_2[1]
        p_succ = (p_1 * q_1) + (p_2 * q_2)
        # cost function returns value to be minimized, namely the complement to the probability we want to maximize
        return 1-p_succ

    # Initial States |phi_1> and |phi_2>
    phi_1 = np.asarray([np.cos(theta_1),np.sin(theta_1)])
    phi_2 = np.asarray([np.cos(theta_2), np.sin(theta_2)])


    # First guess lambda
    rotation_lambda_guess = np.array([1.])
    # Minimize cost function, which maximizes total probability
    optim = qml.AdagradOptimizer(stepsize=0.1)
    conv_tol = 1e-10
    phi_states = [phi_1, phi_2]
    rotation_lambda_guess, conv = optim.step_and_cost(lambda x: cost(x, phi_states), rotation_lambda_guess)
    prev_cost = conv

    while conv > conv_tol:
        rotation_lambda_guess, cost_var = optim.step_and_cost(lambda x: cost(x, phi_states), rotation_lambda_guess)
        conv = abs(cost_var - prev_cost)
        prev_cost = cost_var
        print(1-cost_var)

    # Return the highest probability of distinguishing the states
    return 1-cost(rotation_lambda_guess, phi_states)



# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    theta1, theta2, p_1, p_2 = json.loads(test_case_input)
    prob = np.array(maximal_probability(theta1, theta2, p_1, p_2))

    return str(prob)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    print(solution_output)
    # I changed this assertion from rtol=1e-4, as
    assert np.allclose(solution_output, expected_output, rtol=1e-4)


# These are the public test cases
test_cases = [
    ('[0, 0.7853981633974483, 0.25, 0.75]', '0.8952847075210476'),
    ('[1.83259571459, 1.88495559215, 0.5, 0.5]', '0.52616798')
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