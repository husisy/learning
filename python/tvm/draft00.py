# https://tvm.apache.org/docs/tutorials/get_started/autotvm_relay_x86.html
import os
import onnx
import PIL
import PIL.Image
import timeit
import numpy as np
import scipy.special

import tvm
import tvm.relay
import tvm.contrib.graph_executor
# import tvm.auto_scheduler as auto_scheduler
# from tvm.autotvm.tuner import XGBTuner
# from tvm import autotvm

# from PIL import Image
# from tvm.contrib.download import download_testdata
# import tvm.relay as relay
# import tvm
# from tvm.contrib import graph_executor

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


target = 'llvm -mcpu=core-avx2'

# https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx
onnx_model = onnx.load(hf_file('resnet50-v2-7.onnx'))

# https://s3.amazonaws.com/model-server/inputs/kitten.jpg
resized_image = PIL.Image.open(hf_file('kitten.jpg')).resize((224, 224))
tmp0 = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
tmp1 = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
img_data = np.asarray(resized_image).transpose(2,0,1).astype(np.float32)
img_data = (img_data[np.newaxis] / 255 - tmp0) / tmp1

input_name = 'data'
mod, params = tvm.relay.frontend.from_onnx(onnx_model, shape={input_name:img_data.shape})
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target=target, params=params)

dev = tvm.device(target, 0)
module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))


dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

timing_number = 10
tmp0 = np.array(timeit.Timer(lambda: module.run()).repeat(repeat=10, number=timing_number))
print('unoptimized time: ', np.array(tmp0).mean() / timing_number)
# p720-cpu 0.0368

# https://s3.amazonaws.com/onnx-model-zoo/synset.txt
with open(hf_file('synset.txt'), "r") as f:
    ind_to_label = [x.strip() for x in f]
predict_probability = scipy.special.softmax(tvm_output)[0]
for ind0 in np.argsort(predict_probability)[::-1][:5]:
    print(f'label={ind_to_label[ind0]}, probability={predict_probability[ind0]}')



# Set up some basic parameters for the runner. The runner takes compiled code
# that is generated with a specific set of parameters and measures the
# performance of it. ``number`` specifies the number of different
# configurations that we will test, while ``repeat`` specifies how many
# measurements we will take of each configuration. ``min_repeat_ms`` is a value
# that specifies how long need to run configuration test. If the number of
# repeats falls under this time, it will be increased. This option is necessary
# for accurate tuning on GPUs, and is not required for CPU tuning. Setting this
# value to 0 disables it. The ``timeout`` places an upper limit on how long to
# run training code for each tested configuration.

number = 10
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

# create a TVM runner
runner = tvm.autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

# Create a simple structure for holding tuning options. We use an XGBoost
# algorithim for guiding the search. For a production job, you will want to set
# the number of trials to be larger than the value of 10 used here. For CPU we
# recommend 1500, for GPU 3000-4000. The number of trials required can depend
# on the particular model and processor, so it's worth spending some time
# evaluating performance across a range of values to find the best balance
# between tuning time and model optimization. Because running tuning is time
# intensive we set number of trials to 10, but do not recommend a value this
# small. The ``early_stopping`` parameter is the minimum number of trails to
# run before a condition that stops the search early can be applied. The
# measure option indicates where trial code will be built, and where it will be
# run. In this case, we're using the ``LocalRunner`` we just created and a
# ``LocalBuilder``. The ``tuning_records`` option specifies a file to write
# the tuning data to.

tuning_option = {
    "tuner": "xgb",
    "trials": 10,
    "early_stopping": 100,
    "measure_option": tvm.autotvm.measure_option(builder=tvm.autotvm.LocalBuilder(build_func="default"), runner=runner),
    "tuning_records": "resnet-50-v2-autotuning.json",
}

# begin by extracting the taks from the onnx model
tasks = tvm.autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = tvm.autotvm.tuner.XGBTuner(task, loss_type="rank")
    tmp0 = [
            tvm.autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            tvm.autotvm.callback.log_to_file(tuning_option["tuning_records"]),
    ]
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=tmp0,
    )

with tvm.autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = tvm.relay.build(mod, target=target, params=params)

dev = tvm.device(target, 0)
module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))


timing_number = 10
tmp0 = np.array(timeit.Timer(lambda: module.run()).repeat(repeat=10, number=timing_number))
print('optimized time: ', np.array(tmp0).mean() / timing_number)
# p720: 0.0268775780199212
