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

from DAG_kernel_builder import (
    DAGKernelBuilder,
    Instruction
)


class BasicCompiledSolution(DAGKernelBuilder):
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
        # Scratch space addresses
        tmp0 = self.alloc_scratch("tmp0")
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
            self.add("load", ("const", tmp0, i))
            self.add("load", ("load", self.scratch[v], tmp0))

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
        
        hash_array1 = self.alloc_scratch("hash_array1", 48)
        hash_array2 = self.alloc_scratch("hash_array2", 48)
        for i in range(len(HASH_STAGES)):
            self.instrs.append(
                {
                    "valu": [
                        ("vbroadcast", hash_array1 + i * 8, self.scratch_const(HASH_STAGES[i][1])),
                        ("vbroadcast", hash_array2 + i * 8, self.scratch_const(HASH_STAGES[i][4]))
                    ]
                }
            )
        
        n_nodes_v = self.alloc_scratch("n_nodes_v", 8)
        self.instrs.append({
            "valu": [("vbroadcast", n_nodes_v, self.scratch["n_nodes"])]
        })
              
        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # TODO: start
        block_size = 4
        vector_size = 8
        
        # vector/scalar scratch registers
        tmp1_v = self.alloc_scratch("tmp1", block_size * vector_size)
        tmp2_v = self.alloc_scratch("tmp2", block_size * vector_size)
        tmp3_v = self.alloc_scratch("tmp3", block_size * vector_size)
        # tmp4_v = self.alloc_scratch("tmp4", block_size * vector_size)
        # tmp5_v = self.alloc_scratch("tmp5", block_size * vector_size)
        tmp_idx_v = self.alloc_scratch("tmp_idx", block_size * vector_size)
        tmp_val_v = self.alloc_scratch("tmp_val", block_size * vector_size)
        tmp_node_val_v = self.alloc_scratch("tmp_node_val", block_size * vector_size)
        tmp_addr_v = self.alloc_scratch("tmp_addr", block_size * vector_size)
        
        for round in range(rounds):
            for group_id in range(batch_size // (block_size * vector_size)):
                for block_id in range(block_size): # complete block_size * vector_size after loop finishes
                    # each iteration will complete 8 inputs at a time
                    i = group_id * block_size * vector_size + block_id * vector_size
                    i_const = self.scratch_const(i)
                    
                    # assign block registers
                    tmp1 = tmp1_v + block_id * vector_size
                    tmp2 = tmp2_v + block_id * vector_size
                    tmp3 = tmp3_v + block_id * vector_size
                    # tmp4 = tmp4_v + block_id * vector_size
                    # tmp5 = tmp5_v + block_id * vector_size
                    tmp_idx = tmp_idx_v + block_id * vector_size
                    tmp_val = tmp_val_v + block_id * vector_size 
                    tmp_node_val = tmp_node_val_v + block_id * vector_size
                    tmp_addr = tmp_addr_v + block_id * vector_size
                    
                    # TODO: cache at start
                    # idx = mem[inp_indices_p + i] (vectorized)
                    self.add_node(Instruction("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))) 
                    self.add_node(Instruction("load", ("vload", tmp_idx, tmp_addr)))
                    
                    # val = mem[inp_values_p + i] (vectorized)
                    self.add_node(Instruction("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    self.add_node(Instruction("load", ("vload", tmp_val, tmp_addr)))

                    # pull tree nodes
                    # for j in range(0, 8, 2):
                    #     self.add_node(Instruction("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx + j)))
                    #     self.add_node(Instruction("alu", ("+", tmp_addr + 1, self.scratch["forest_values_p"], tmp_idx + j + 1))) # TODO: can vectorize
                    #     self.add_node(Instruction("load", ("load", tmp_node_val + j, tmp_addr)))
                    #     self.add_node(Instruction("load", ("load", tmp_node_val + j + 1, tmp_addr + 1)))
                    for j in range(8):
                        self.add_node(Instruction("alu", ("+", tmp_addr + j, self.scratch["forest_values_p"], tmp_idx + j)))
                        self.add_node(Instruction("load", ("load", tmp_node_val + j, tmp_addr + j)))

                    # body_instrs = self.compile_kernel()
                    # self.instrs.extend(body_instrs)
                    # self.clear()
                    
                    # self.instrs.append(
                    #     {"debug": [("vcompare", tmp_node_val, [(round, i + j, "node_val") for j in range(8)])]}
                    # )

                    # val = myhash(val ^ node_val) (vectorized)
                    self.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                        self.add_node(Instruction("valu", (op1, tmp1, tmp_val, hash_array1 + hi * vector_size)))
                        self.add_node(Instruction("valu", (op3, tmp2, tmp_val, hash_array2 + hi * vector_size)))
                        self.add_node(Instruction("valu", (op2, tmp_val, tmp1, tmp2)))
                        
                    # body_instrs = self.compile_kernel()
                    # self.instrs.extend(body_instrs)
                    # self.clear()

                    # self.instrs.append(
                    #     {"debug": [("vcompare", tmp_val, [(round, i + j, "hashed_val") for j in range(8)])]}
                    # )

                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    self.add_node(Instruction("valu", ("%", tmp1, tmp_val, two_const_v)))
                    self.add_node(Instruction("valu", ("==", tmp1, tmp1, zero_const_v)))
                    self.add_node(Instruction("flow", ("vselect", tmp3, tmp1, one_const_v, two_const_v)))
                    self.add_node(Instruction("valu", ("*", tmp_idx, tmp_idx, two_const_v)))
                    self.add_node(Instruction("valu", ("+", tmp_idx, tmp_idx, tmp3)))

                    # idx = 0 if idx >= n_nodes else idx
                    self.add_node(Instruction("valu", ("<", tmp1, tmp_idx, n_nodes_v)))
                    self.add_node(Instruction("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_const_v)))

                    # body_instrs = self.compile_kernel()
                    # self.instrs.extend(body_instrs)
                    # self.clear()

                    # self.instrs.append(
                    #     {"debug": [("vcompare", tmp_idx, [(round, group_id * block_size * vector_size + j, "wrapped_idx") for j in range(8)])]}
                    # )

                    # mem[inp_indices_p + i] = idx
                    self.add_node(Instruction("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                    self.add_node(Instruction("store", ("vstore", tmp_addr, tmp_idx)))

                    # mem[inp_values_p + i] = val
                    self.add_node(Instruction("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                    self.add_node(Instruction("store", ("vstore", tmp_addr, tmp_val)))
                    
                    # body_instrs = self.compile_kernel()
                    # self.instrs.extend(body_instrs)
                    # self.clear()
                    
                    # self.instrs.append(
                    #     {"debug": [("vcompare", tmp_idx, [(round, group_id * block_size * vector_size + j, "wrapped_idx") for j in range(8)])]}
                    # )

                body_instrs = self.compile_kernel()
                self.instrs.extend(body_instrs)
                self.clear()
                # for i in range(block_size):
                self.instrs.append(
                    {"debug": 
                        [("vcompare", tmp_idx_v, 
                            [(round, group_id * block_size * vector_size + j, "wrapped_idx") for j in range(vector_size)]
                        )]
                    }
                )

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})