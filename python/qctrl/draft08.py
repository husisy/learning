# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-evaluate-control-susceptibility-to-quasi-static-noise
import numpy as np
import matplotlib.pyplot as plt
import dotenv

import qctrl
import qctrlvisualizer
import qctrlopencontrols

plt.style.use(qctrlvisualizer.get_qctrl_style())

QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)

# Define control parameters
omega_max = 2 * np.pi * 1e6  # Hz
total_rotation = np.pi / 2

# Define coefficient array for the noise values
dephasing_coefficients = np.linspace(-1.0, 1.0, 101) * omega_max

fig,ax = plt.subplots(figsize=(10,5))
fig.suptitle("Dephasing noise susceptibility of different pulses")

# For each scheme, compute and plot the results of the quasi-static scan
for scheme_name, function in [("CORPSE", qctrlopencontrols.new_corpse_control), ("primitive", qctrlopencontrols.new_primitive_control)]:
    # Define pulse objects using pulses from Q-CTRL Open Controls
    pulse = function(rabi_rotation=total_rotation, azimuthal_angle=0.0, maximum_rabi_rate=omega_max)

    graph = QCTRL_HANDLE.create_graph()
    rabi_signal = graph.pwc(durations=pulse.durations, values=pulse.rabi_rates * np.exp(1j * pulse.azimuthal_angles))
    rabi_coupling_term = graph.hermitian_part(rabi_signal * graph.pauli_matrix("M"))

    # Define dephasing term, a [101] batch of operators, created by multiplying
    # a [101] batch of (constant) PWC signals by the corresponding σz/2 operator
    dephasing_signal = graph.constant_pwc(
        constant=dephasing_coefficients,
        duration=pulse.duration,
        batch_dimension_count=1,
    )
    dephasing_term = dephasing_signal * graph.pauli_matrix("Z") / 2

    # Build total Hamiltonian, a [101] batch of operators
    hamiltonian = rabi_coupling_term + dephasing_term

    # Calculate infidelities, a [101] tensor Rx(pi/2)
    sqrt_sigma_x = (0.5 + 0.5j) * graph.pauli_matrix("I") + (0.5 - 0.5j) * graph.pauli_matrix("X")

    graph.infidelity_pwc(
        hamiltonian=hamiltonian, target=graph.target(sqrt_sigma_x), name="infidelities"
    )

    result = QCTRL_HANDLE.functions.calculate_graph(graph=graph, output_node_names=["infidelities"])

    # Extract and plot infidelities
    infidelities = result.output["infidelities"]["value"]
    ax.plot(dephasing_coefficients / omega_max, infidelities, label=scheme_name)
ax.set_ylabel("Infidelity")
ax.set_xlabel(r"Relative dephasing coefficient $\eta/\Omega_\mathrm{max}$")
ax.legend()
