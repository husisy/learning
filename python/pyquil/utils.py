from pyquil.api import WavefunctionSimulator, ForestConnection

def setup_qvm_quilc_connection(qvm_ip='127.0.0.1', qvm_port=5000, quilc_ip=None, quilc_port=5555):
    '''
    qvm_ip(str/NoneType): if None, will be set as quilc_ip
    qvm_port(int)
    quilc_ip(str/NoneType): if None, will be set as qvm_ip
    quilc_port(int)
    (ret0)FC(pyquil.api.ForestConnection)
    (ret1)WF_SIM(pyquil.api.WavefunctionSimulator)
    '''
    if qvm_ip is None:
        qvm_ip = quilc_ip
    if quilc_ip is None:
        quilc_ip = qvm_ip
    assert isinstance(qvm_ip, str) and isinstance(quilc_ip, str)
    FC = ForestConnection(
        sync_endpoint = 'http://{}:{}'.format(qvm_ip, qvm_port),
        compiler_endpoint = 'tcp://{}:{}'.format(quilc_ip, quilc_port),
    )
    WF_SIM = WavefunctionSimulator(connection=FC)
    return FC, WF_SIM
