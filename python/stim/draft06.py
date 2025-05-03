import stim
from typing import List


def distance_of_stabilizer_code(
        stabilizers: List[stim.PauliString],
        logical_x: stim.PauliString,
        logical_z: stim.PauliString) -> int:
    circuit = stabilizer_code_to_phenomenological_noise_circuit(stabilizers, logical_x, logical_z)
    error_mechanisms = circuit.search_for_undetectable_logical_errors(
        # For bigger codes, you will likely want to do a truncated search where these parameters
        # are tuned to only explore smaller potential errors.
        dont_explore_edges_increasing_symptom_degree=False,
        dont_explore_detection_event_sets_with_size_above=9999,
        dont_explore_edges_with_degree_above=9999,

        canonicalize_circuit_errors=True,
    )
    return len(error_mechanisms)


def stabilizer_code_to_phenomenological_noise_circuit(
        stabilizers: List[stim.PauliString],
        logical_x: stim.PauliString,
        logical_z: stim.PauliString) -> stim.Circuit:
    num_qubits = len(logical_x)
    assert len(logical_z) == len(logical_x)
    assert all(len(stabilizer) == num_qubits for stabilizer in stabilizers)

    circuit = stim.Circuit()

    # Entangle observables with a noiseless ancilla so that they can be simultaneously tested.
    logical_xx = logical_x + stim.PauliString("X")
    logical_zz = logical_z + stim.PauliString("Z")
    measure_stabilizer(logical_xx, out=circuit)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    measure_stabilizer(logical_zz, out=circuit)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    circuit.append("TICK")

    # Project stabilizers.
    for stabilizer in stabilizers:
        measure_stabilizer(stabilizer, out=circuit)
    circuit.append("TICK")

    # Apply noise.
    circuit.append("DEPOLARIZE1", range(num_qubits), 1e-3)
    circuit.append("TICK")

    # Measure after noise and compare to before noise.
    for stabilizer in stabilizers:
        measure_stabilizer(stabilizer, out=circuit)
    for k in range(len(stabilizers)):
        circuit.append("DETECTOR", [stim.target_rec(-1 - k),
                                    stim.target_rec(-1 - k - len(stabilizers))])
    circuit.append("TICK")

    measure_stabilizer(logical_xx, out=circuit)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    measure_stabilizer(logical_zz, out=circuit)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)

    return circuit


def measure_stabilizer(stabilizer: stim.PauliString, *, out: stim.Circuit) -> None:
    targets = []
    for q, p in enumerate(stabilizer):
        if p == 1:
            targets.append(stim.target_x(q))
        elif p == 2:
            targets.append(stim.target_y(q))
        elif p == 3:
            targets.append(stim.target_z(q))
        if p != 0:
            targets.append(stim.target_combiner())
    targets.pop()
    out.append("MPP", targets)


distance_of_perfect_5_qubit_code = distance_of_stabilizer_code(
    stabilizers=[
        stim.PauliString("XZZ_Z"),
        stim.PauliString("XXXZ_"),
        stim.PauliString("_ZXXX"),
        stim.PauliString("Z_ZZX"),
    ],
    logical_x=stim.PauliString("X_X_X"),
    logical_z=stim.PauliString("_ZZZ_"),
)
print(distance_of_perfect_5_qubit_code)
