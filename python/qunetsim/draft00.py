import qunetsim

# qunetsim.objects.Logger.DISABLED = True


network = qunetsim.components.Network.get_instance()
network.start()

alice = qunetsim.components.Host('Alice')
bob = qunetsim.components.Host('Bob')

alice.add_connection(bob.host_id)
bob.add_connection(alice.host_id)

alice.start()
bob.start()

network.add_hosts([alice, bob])

# Block Alice to wait for qubit arrive from Bob
alice.send_epr(bob.host_id, await_ack=True)
q_alice = alice.get_epr(bob.host_id)
q_bob = bob.get_epr(alice.host_id)

print("EPR is in state: %d, %d" % (q_alice.measure(), q_bob.measure()))
network.stop(True)



backend = qunetsim.backends.EQSNBackend()


def protocol_1(host, receiver):
    # Here we write the protocol code for a host.
    for i in range(5):
        s = 'Hi {}.'.format(receiver)
        print("{} sends: {}".format(host.host_id, s))
        host.send_classical(receiver, s, await_ack=True)
    for i in range(5):
        q = qunetsim.objects.Qubit(host)
        q.X()
        print("{} sends qubit in the |1> state".format(host.host_id))
        host.send_qubit(receiver, q, await_ack=True)


def protocol_2(host, sender):
    # Here we write the protocol code for another host.
    for i in range(5):
        sender_message = host.get_classical(sender, wait=5, seq_num=i)
        print("{} Received classical: {}".format(host.host_id, sender_message.content))
    for i in range(5):
        q = host.get_data_qubit(sender, wait=10)
        m = q.measure()
        print("{} measured: {}".format(host.host_id, m))


network = qunetsim.components.Network.get_instance()
nodes = ['A', 'B', 'C']
network.generate_topology(nodes, 'star')
network.start(nodes)

host_a = network.get_host('A')
host_b = network.get_host('B')
host_c = network.get_host('C')

t1 = host_a.run_protocol(protocol_1, (host_c.host_id,))
t2 = host_c.run_protocol(protocol_2, (host_a.host_id,), blocking=True)
network.stop(True)
