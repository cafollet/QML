import json
import pennylane as qml
import pennylane.numpy as np

""" Sourced from Pennylane Challenge: 'Reaching for the Ratio' 
(https://pennylane.ai/challenges/reaching_for_the_ratio) """

def cost_hamiltonian(edges):
    """
    This function build the QAOA cost Hamiltonian for a graph, encoded in its edges

    Args:
    - Edges (list(list(int))): A list of ordered pairs of integers, representing
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - pennylane.Operator: The cost Hamiltonian associated with the graph.
    """

    # Create the coefficient list based on the cost hamiltonian
    coeffs = [0.75 for _ in range(3*len(edges))]
    obs = []
    node_list = []

    # For-loop, appending the first sum over the edges to the observables list
    for edge in edges:

        # Creating the node list to iterate through for the next sum
        if edge[0] not in node_list:
            node_list.append(edge[0])
        if edge[1] not in node_list:
            node_list.append(edge[1])

        # First LC Term of Hamiltonian
        obs.append(qml.Z(edge[0]) @ qml.Z(edge[1]))
        obs.append(qml.Z(edge[0]))
        obs.append(qml.Z(edge[1]))

    # For-loop appending the second sum over the vertices to the observables list
    for node in node_list:
        obs.append(qml.Z(node))
        coeffs.append(-1)  # -1 appended to the observables respective coefficient

    return qml.Hamiltonian(coeffs, obs)


def mixer_hamiltonian(edges):
    """
    This function build the QAOA mixer Hamiltonian for a graph, encoded in its edges

    Args:
    - edges (list(list(int))): A list of ordered pairs of integers, representing
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - pennylane.Operator: The mixer Hamiltonian associated with the graph.
    """


    node_list = []
    obs = []

    # For-loop to create list of observables that match the hamiltonian equation
    for edge in edges:
        if edge[0] not in node_list:
            node_list.append(edge[0])
            obs.append(qml.X(edge[0]))
        if edge[1] not in node_list:
            node_list.append(edge[1])
            obs.append(qml.X(edge[1]))

    coeffs = np.ones(len(node_list))
    return qml.Hamiltonian(coeffs, obs)


def qaoa_circuit(params, edges):
    """
    This quantum function (i.e. a list of gates describing a circuit) implements the QAOA algorithm
    You should only include the list of gates, and not return anything

    Args:
    - params (np.array): A list encoding the QAOA parameters. You are free to choose any array shape you find
    convenient.
    - edges (list(list(int))): A list of ordered pairs of integers, representing
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - This function does not return anything. Do not write any return statements.

    """
    lambdas = params[0]  # both the length of this and the alphas list give the number of layers to be applied
    alphas = params[1]
    node_list = []

    # For-loop to determine the wires (vertices) to iterate through for the hadamard
    for edge in edges:
        if edge[0] not in node_list:
            node_list.append(edge[0])
        if edge[1] not in node_list:
            node_list.append(edge[1])

    # Apply Hadamard gate to each wire
    for wire in node_list:
        qml.Hadamard(wires=wire)

    # Apply Cost and mixer parametrized-evolution layers
    for i, lmbda in enumerate(lambdas):
        qml.ApproxTimeEvolution(cost_hamiltonian(edges), lmbda, n=1)
        qml.ApproxTimeEvolution(mixer_hamiltonian(edges), alphas[i], n=1)

# This function runs the QAOA circuit and returns the expectation value of the cost Hamiltonian

dev = qml.device("default.qubit")


@qml.qnode(dev)
def qaoa_expval(params, edges):
    qaoa_circuit(params, edges)
    return qml.expval(cost_hamiltonian(edges))


def optimize(edges):
    """
    This function returns the parameters that minimize the expectation value of
    the cost Hamiltonian after applying QAOA

    Args:
    - edges (list(list(int))): A list of ordered pairs of integers, representing
    the edges of the graph for which we need to solve the minimum vertex cover problem.


    """

    # Adjustable Hyperparams (Public tests pass at default: layers=6, step=0.01, tested with layers=6, step=0.1 as well)
    layers = 6  # Default: 6
    step = 0.01 # Default: 0.01, passes with 0.1

    # Create the parameters to be optimized
    params = np.ones((2, layers))  # Initial guess
    params.requires_grad = True  # So the params can be backpropped

    # Define the optimizer (default: Adagrad)
    opt = qml.AdagradOptimizer(stepsize=step)

    # Initialize variables
    loss_tol = 1e-05
    loss = 2 * loss_tol
    prev_cost = 0
    count = 0

    # Keeps updating the params until 1000 iterations or the "loss" is smaller than the set tolerance
    while loss > loss_tol and count < 1000:
        loss = np.abs(qaoa_expval(params, edges) - prev_cost)  # NOT ACTUALLY LOSS, JUST LOSS-LIKE VALUES FOR BENCHMARKING

        # Run cost function and update the parameters
        params, prev_cost = opt.step_and_cost(lambda p: qaoa_expval(p, edges), params)

        # Print updated loss and approx_val every 10 counts
        if not count % 10:
            approx_val=approximation_ratio(params, edges)
            print("", end='\r')
            print(f"Epoch: {count}\tloss: {loss}\t\tApprox_Val: {approx_val}", end='')

        count += 1


    # Run the qaoa_probs circuit to get the probabilities
    # (in |ab...cd>, a represents the 0th qubit and d represents the last wire)
    probs = qaoa_probs(params, edges)
    bin_pad = len(str(bin(len(probs)-1))[2:])  # max index to know how to pad the binary


    # Find the indices of all elements in the list with the highest probability,
    # if more than two ways of solving specific graph, will be a list of indices
    max_vals = np.argwhere(np.isclose(probs, np.amax(probs), rtol=0.001)).flatten().tolist()


    # Printing out the vertices that were found (high time complexity so could slow down program at larger graph sizes)
    print("\n\nFinal Guess: Vertices (", end="")
    num_solns = len(max_vals)

    for k, x in enumerate(max_vals):
        # Convert index to binary
        val = format(x, f'0{bin_pad}b')
        is_first=True

        # Convert the index value to vertex value and print
        for i, y in enumerate(val):
            if y == "1":
                if is_first:
                    print(i, end="")
                    is_first = False
                else:
                    print(", ", i, end="")
            if y == "b":
                break
        if k < num_solns-1:
            print("), (", end="")
        else:
            print(")")


    return params



@qml.qnode(dev)
def qaoa_probs(params, edges):
    qaoa_circuit(params, edges)
    return qml.probs()


def approximation_ratio(params, edges):
    true_min = np.min(qml.eigvals(cost_hamiltonian(edges)))

    approx_ratio = qaoa_expval(params, edges) / true_min
    return approx_ratio


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    params = optimize(ins)
    output = approximation_ratio(params, ins)

    ground_energy = np.min(qml.eigvals(cost_hamiltonian(ins)))

    index = np.argmax(qaoa_probs(params, ins))
    vector = np.zeros(len(qml.matrix(cost_hamiltonian(ins))))
    vector[index] = 1

    calculate_energy = np.real_if_close(np.dot(np.dot(qml.matrix(cost_hamiltonian(ins)), vector), vector))
    verify = np.isclose(calculate_energy, ground_energy)
    print()
    if verify:
        return str(output)

    return "QAOA failed to find right answer"


def check(solution_output: str, expected_output: str) -> None:
    assert not solution_output == "QAOA failed to find right answer", "QAOA failed to find the ground eigenstate."

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert solution_output >= expected_output - 0.01, "Minimum approximation ratio not reached"


# These are the public test cases
test_cases = [
    ('[[0, 1], [1, 2], [0, 2], [2, 3]]', '0.55'),
    ('[[0, 1], [1, 2], [2, 3], [3, 0]]', '0.92'),
    ('[[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]]', '0.55')
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