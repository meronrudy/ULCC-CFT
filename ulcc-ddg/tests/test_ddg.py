from ulcc_ddg.metric_graph import MetricGraph, Edge
from ulcc_ddg.holonomy import holonomy_loop, holonomy_loop_with_transport
from ulcc_ddg.transport import parallel_transport_rotation

def test_zero_holonomy_identity_transport():
    g = MetricGraph({Edge(0,1): 1.0, Edge(1,2): 1.0, Edge(2,0): 1.0})
    loop = [Edge(0,1), Edge(1,2), Edge(2,0)]
    assert holonomy_loop(loop) == 0.0

def test_positive_holonomy_on_curved_patch():
    # Triangular loop with constant positive edge rotation
    loop = [Edge(0,1), Edge(1,2), Edge(2,0)]
    alpha = 0.05
    angle_map = {e: alpha for e in loop}
    T = parallel_transport_rotation(angle_map)
    h = holonomy_loop_with_transport(loop, T)
    assert h > 1e-6

def test_flat_zero_holonomy_general():
    # Angles that cancel around the loop: alpha, beta, -(alpha+beta)
    loop = [Edge(0,1), Edge(1,2), Edge(2,0)]
    alpha = 0.07
    beta = 0.03
    angle_map = {loop[0]: alpha, loop[1]: beta, loop[2]: -(alpha + beta)}
    T = parallel_transport_rotation(angle_map)
    h = holonomy_loop_with_transport(loop, T)
    assert abs(h) < 1e-6
