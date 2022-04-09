import numpy as np

import qecsim
import qecsim.models.basic
import qecsim.models.generic
import qecsim.paulitools
import qecsim.app

np_rng = np.random.default_rng()

# pt = qecsim.paulitools

my_code = qecsim.models.basic.FiveQubitCode()
my_error_model = qecsim.models.generic.DepolarizingErrorModel()
my_decoder = qecsim.models.generic.NaiveDecoder()


error_probability = 0.1

error = my_error_model.generate(my_code, error_probability, np_rng)
print(error, qecsim.paulitools.bsf_to_pauli(error))


syndrome = qecsim.paulitools.bsp(error, my_code.stabilizers.T)
print(syndrome)


recovery = my_decoder.decode(my_code, syndrome)
print(recovery, qecsim.paulitools.bsf_to_pauli(recovery))


print(qecsim.paulitools.bsp(recovery ^ error, my_code.stabilizers.T))


print(qecsim.paulitools.bsp(recovery ^ error, my_code.logicals.T))

z0 = qecsim.app.run_once(my_code, my_error_model, my_decoder, error_probability)
