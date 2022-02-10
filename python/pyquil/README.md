# pyquil

1. link
   * [documentation](http://docs.rigetti.com/en/stable/)
2. 安装`pip install pyquil`
3. 运行单元测试需`cp ../my_quantum_circuit/np_quantum_circuit.py .`

## 安装qvm/quilc

1. 测试
   * ubuntu apt install from `.deb`文件：失败，需要`glibc2.28`；应该无法解决，ubuntu19.04方才用上了`glibc2.29`
   * windows install from `.msi`文件：失败，原因不详
   * ubuntu install from source：失败，原因不详
   * ubuntu install with docker：成功（操作见下）
   * windows install with docker：成功（类似于ubuntu install with docker）
2. docker安装流程
   * 安装docker（见相应文档）
   * `docker pull rigetti/quilc`
   * `docker container run --rm --detach -p 5555:5555 rigetti/quilc -R`
   * `docker pull rigetti/qvm`
   * `docker container run --rm --detach -p 5000:5000 rigetti/qvm -S`
3. `quilc`与`qvm`运行独立，虽然quilc container同时也暴露了6000端口，两者没有关系

## minimum working example

```Python
from pyquil import Program, get_qc, list_quantum_computers
from pyquil.gates import H
from pyquil.api import WavefunctionSimulator, ForestConnection

FC = ForestConnection(sync_endpoint='http://127.0.0.1:5000', compiler_endpoint='tcp://127.0.0.1:5555')
WF_SIM = WavefunctionSimulator(connection=FC)
QC = get_qc('2q-qvm', connection=FC)
circuit = Program(H(0))
print('state: ', WF_SIM.wavefunction(circuit))
print('run and measure: ', QC.run_and_measure(circuit, trials=10))
```
