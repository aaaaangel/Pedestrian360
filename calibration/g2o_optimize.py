import numpy as np
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        # solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().set_verbose(True)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex_pose(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
        return v_se3


    def add_edge(self, id, vertices, measurement, parm_id,
            information=np.identity(2),
            robust_kernel=None):

        # edge = g2o.EdgeSE3()
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_id(id)
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_parameter_id(0, parm_id)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


