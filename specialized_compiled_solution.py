from problem import HASH_STAGES

from DAG_kernel_builder import DAGKernelBuilder, Instruction
import inner_loop_opt


class SpecializedCompiledSolution(DAGKernelBuilder):
    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        # Scratch space addresses
        tmp0 = self.alloc_scratch("tmp0")
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp0, i))
            self.add("load", ("load", self.scratch[v], tmp0))

        # n_nodes_v = self.alloc_scratch("n_nodes_v", 8)
        # self.instrs.append({"valu": [("vbroadcast", n_nodes_v, self.scratch["n_nodes"])]})

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        inner_loop_opt.kernel(self)
