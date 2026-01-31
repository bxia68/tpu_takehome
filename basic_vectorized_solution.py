from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

from kernel_builder import (
    KernelBuilder
)


class BasicVectorizedKernelBuilder(KernelBuilder):
    def build_vhash(self, val_hash_addr, tmp1, tmp2, round, i, tmp_broad1, tmp_broad2):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(
                (
                    "valu", [
                        ("vbroadcast", tmp_broad1, self.scratch_const(val1)),
                        ("vbroadcast", tmp_broad2, self.scratch_const(val3)), 
                    ]
                )
            )

            slots.append(("valu", (op1, tmp1, val_hash_addr, tmp_broad1)))
            slots.append(("valu", (op3, tmp2, val_hash_addr, tmp_broad2)))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            
            for j in range(8):
                slots.append(("debug", ("compare", val_hash_addr + j, (round, i + j, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1", 8)
        tmp2 = self.alloc_scratch("tmp2", 8)
        tmp3 = self.alloc_scratch("tmp3", 8)
        tmp_broad1 = self.alloc_scratch("tmp_broad1", 8)
        tmp_broad2 = self.alloc_scratch("tmp_broad2", 8)
        # tmp_broad3 = self.alloc_scratch("tmp_broad3", 8)
        # tmp_broad4 = self.alloc_scratch("tmp_broad4", 8)
        
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        
        # vector const registers
        zero_const_v = self.alloc_scratch("zero_const_v", 8)
        one_const_v = self.alloc_scratch("one_const_v", 8)
        two_const_v = self.alloc_scratch("two_const_v", 8)
        self.instrs.append(
            {
                "valu": [
                    ("vbroadcast", zero_const_v, zero_const),
                    ("vbroadcast", one_const_v, one_const), 
                    ("vbroadcast", two_const_v, two_const)
                ]
            }
        )
              
        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # vector/scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx", 8)
        tmp_val = self.alloc_scratch("tmp_val", 8)
        tmp_node_val = self.alloc_scratch("tmp_node_val", 8)
        tmp_addr = self.alloc_scratch("tmp_addr")
        
        tmp_tree_addr1 = self.alloc_scratch("tmp_tree_addr1")
        tmp_tree_addr2 = self.alloc_scratch("tmp_tree_addr2")
        # tmp_tree_idx = self.alloc_scratch("tmp_tree_idx")
        # tmp_tree_node_val = self.alloc_scratch("tmp_tree_node_val")

        for round in range(rounds):
            for i in range(0, batch_size, 8):
                # each iteration will complete 8 inputs at a time
                
                i_const = self.scratch_const(i)
                
                # idx = mem[inp_indices_p + i] (vectorized)
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))) 
                body.append(("load", ("vload", tmp_idx, tmp_addr)))
                for j in range(8):
                    body.append(("debug", ("compare", tmp_idx + j, (round, i + j, "idx"))))
                
                # val = mem[inp_values_p + i] (vectorized)
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("vload", tmp_val, tmp_addr)))
                for j in range(8):
                    body.append(("debug", ("compare", tmp_val + j, (round, i + j, "val"))))

                # node_val = mem[forest_values_p + idx] (need to load each tree val individually)
                # body.append(("alu", ("|", tmp_tree_idx, tmp_idx, zero_const)))           # tree_idx = idx
                # body.append(("alu", ("|", tmp_tree_node_val, tmp_node_val, zero_const))) # tree_node_val = node_val
                # body.append(("load", ("load_offset", tmp_node_val, self.scratch["forest_values_p"], tmp_idx))) # allowed to add with "j"? 
                # body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                
                # body.append(("alu", ("|", tmp_tree_idx, tmp_idx, zero_const))) # tree_idx = idx
                next_instr = {}
                for j in range(0, 8, 2):
                    next_instr["alu"] = [("+", tmp_tree_addr1, self.scratch["forest_values_p"], tmp_idx + j),
                                         ("+", tmp_tree_addr2, self.scratch["forest_values_p"], tmp_idx + j + 1)]  # allowed to add with "j"? 
                    body.append(next_instr)
                    next_instr = {"load": [("load", tmp_node_val + j, tmp_tree_addr1), 
                                          ("load", tmp_node_val + j + 1, tmp_tree_addr2)]}
                body.append(next_instr)
                
                for j in range(8):
                    body.append(("debug", ("compare", tmp_node_val + j, (round, i + j, "node_val"))))

                # val = myhash(val ^ node_val) (vectorized)
                body.append(("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_vhash(tmp_val, tmp1, tmp2, round, i, tmp_broad1, tmp_broad2))
                for j in range(8):
                    body.append(("debug", ("compare", tmp_val + j, (round, i + j, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1, tmp_val, two_const_v)))
                body.append(("valu", ("==", tmp1, tmp1, zero_const_v)))
                body.append(("flow", ("vselect", tmp3, tmp1, one_const_v, two_const_v)))
                body.append(("valu", ("*", tmp_idx, tmp_idx, two_const_v)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp3)))
                for j in range(8):
                    body.append(("debug", ("compare", tmp_idx + j, (round, i + j, "next_idx"))))

                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("vbroadcast", tmp_broad1, self.scratch["n_nodes"])))
                body.append(("valu", ("<", tmp1, tmp_idx, tmp_broad1)))
                body.append(("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_const_v)))
                for j in range(8):
                    body.append(("debug", ("compare", tmp_idx + j, (round, i + j, "wrapped_idx"))))

                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, tmp_idx)))

                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})