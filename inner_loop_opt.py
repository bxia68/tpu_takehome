from problem import HASH_STAGES

from DAG_kernel_builder import DAGKernelBuilder, Instruction

CHARACTER_COUNT = 256
ROUNDS = 16
BLOCK_SIZE = 8
VECTOR_SIZE = 8


def kernel(kb: DAGKernelBuilder):
    # vector const registers
    for i in range(5):
        const_v = kb.alloc_scratch(f"const_{i}", VECTOR_SIZE)
        kb.add_node(Instruction("valu", ("vbroadcast", const_v, kb.scratch_const(i))))

    tree_ptr_v = kb.alloc_scratch("tree_ptr_v", VECTOR_SIZE)
    kb.add_node(Instruction("valu", ("vbroadcast", tree_ptr_v, kb.scratch["forest_values_p"])))

    hash_array1 = kb.alloc_scratch("hash_array1", 6 * VECTOR_SIZE)
    hash_array2 = kb.alloc_scratch("hash_array2", 6 * VECTOR_SIZE)
    for i in range(len(HASH_STAGES)):
        kb.add_node(Instruction("valu", ("vbroadcast", hash_array1 + i * VECTOR_SIZE, kb.scratch_const(HASH_STAGES[i][1]))))
        kb.add_node(Instruction("valu", ("vbroadcast", hash_array2 + i * VECTOR_SIZE, kb.scratch_const(HASH_STAGES[i][4]))))

    # Multiplier constants for fused multiply-add in hash stages 0, 2, 4
    # multiply_add computes: dest = a * b + c
    # We want: tmp_val = tmp_val * (2^shift + 1) + const1
    hash_mult0 = kb.alloc_scratch("hash_mult0", VECTOR_SIZE)  # 2^12 + 1 = 4097
    hash_mult2 = kb.alloc_scratch("hash_mult2", VECTOR_SIZE)  # 2^5 + 1 = 33
    hash_mult4 = kb.alloc_scratch("hash_mult4", VECTOR_SIZE)  # 2^3 + 1 = 9
    kb.add_node(Instruction("valu", ("vbroadcast", hash_mult0, kb.scratch_const(4097))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_mult2, kb.scratch_const(33))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_mult4, kb.scratch_const(9))))

    # vector/scalar scratch registers
    tmp1_v = kb.alloc_scratch("tmp1_v", BLOCK_SIZE * VECTOR_SIZE)
    tmp2_v = kb.alloc_scratch("tmp2_v", BLOCK_SIZE * VECTOR_SIZE)
    tmp3_v = kb.alloc_scratch("tmp3_v", BLOCK_SIZE * VECTOR_SIZE)

    # tmp4_v = kb.alloc_scratch("tmp4_v", BLOCK_SIZE * VECTOR_SIZE)
    # tmp5_v = kb.alloc_scratch("tmp5_v", BLOCK_SIZE * VECTOR_SIZE)
    # tmp_idx_v = kb.alloc_scratch("tmp_idx", BLOCK_SIZE * VECTOR_SIZE)
    # tmp_val_v = kb.alloc_scratch("tmp_val", BLOCK_SIZE * VECTOR_SIZE)

    # tmp_node_val_v = kb.alloc_scratch("tmp_node_val_v", BLOCK_SIZE * VECTOR_SIZE)
    # tmp_addr_v = kb.alloc_scratch("tmp_addr_v", BLOCK_SIZE * VECTOR_SIZE)
    tmp_node_val_v = kb.alloc_scratch("tmp_node_val_v", 256)
    tmp_addr_v = kb.alloc_scratch("tmp_addr_v", 256)

    kb.alloc_scratch("round0_cache", 8)
    kb.alloc_scratch("round1_cache", 16)

    idx_array = kb.alloc_scratch("idx_array", 256)
    val_array = kb.alloc_scratch("val_array", 256)
    for i in range(0, 256, VECTOR_SIZE):
        tmp_addr_1 = tmp_addr_v
        tmp_addr_2 = tmp_addr_v + 1
        kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("load", ("vload", idx_array + i, tmp_addr_1)))
        kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("load", ("vload", val_array + i, tmp_addr_2)))

    for round_num in range(ROUNDS):
        round_kernel(kb, round_num)
        # kb.compile_kernel()
        # for i in range(0, 256, 8):
        #     kb.instrs.append({"debug": [("vcompare", idx_array + i, [(round_num, i + j, "wrapped_idx") for j in range(8)])]})

    for i in range(0, 256, VECTOR_SIZE):
        tmp_addr_1 = tmp_addr_v
        tmp_addr_2 = tmp_addr_v + 1
        kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_1, idx_array + i)))
        kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_2, val_array + i)))

    kb.compile_kernel()

    # Required to match with the yield in reference_kernel2
    kb.instrs.append({"flow": [("pause",)]})


def round_kernel(kb: DAGKernelBuilder, round_num: int):
    if round_num == 0 or round_num == 11:
        round_0_kernel(kb, preloaded=(round_num > 0))
    elif round_num == 1 or round_num == 12:
        round_1_kernel(kb, preloaded=(round_num > 1))
    else:
        generic_round_kernel(kb, round_num)
    kb.add_node(Instruction("flow", ("trace_write", kb.scratch["idx_array"] + 248)))


def generic_round_kernel(kb: DAGKernelBuilder, round_num: int):
    for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * VECTOR_SIZE)):
        for block_id in range(BLOCK_SIZE):  # complete BLOCK_SIZE * VECTOR_SIZE after loop finishes
            # each iteration will complete 8 inputs at a time
            i = group_id * BLOCK_SIZE * VECTOR_SIZE + block_id * VECTOR_SIZE

            # assign block registers
            tmp1 = kb.scratch["tmp1_v"] + block_id * VECTOR_SIZE
            tmp2 = kb.scratch["tmp2_v"] + block_id * VECTOR_SIZE
            tmp3 = kb.scratch["tmp3_v"] + block_id * VECTOR_SIZE

            tmp_node_val = kb.scratch["tmp_node_val_v"] + i
            tmp_addr = kb.scratch["tmp_addr_v"] + i

            tmp_idx = kb.scratch["idx_array"] + i
            tmp_val = kb.scratch["val_array"] + i

            # pull tree nodes
            kb.add_node(Instruction("valu", ("+", tmp_addr, kb.scratch["tree_ptr_v"], tmp_idx)))
            for j in range(8):
                kb.add_node(Instruction("load", ("load", tmp_node_val + j, tmp_addr + j)))

            # val = myhash(val ^ node_val)
            kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
            vhash(kb, tmp1, tmp2, tmp_val)

            if round_num != 10:
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["const_2"])))
                kb.add_node(Instruction("flow", ("vselect", tmp3, tmp1, kb.scratch["const_2"], kb.scratch["const_1"])))
                kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["const_2"], tmp3)))
            else:
                # idx = root (0) if on level 10
                kb.add_node(Instruction("valu", ("&", tmp_idx, kb.scratch["const_0"], kb.scratch["const_0"])))


def round_0_kernel(kb: DAGKernelBuilder, preloaded=False):
    # pull tree nodes (all nodes are the same root node)
    root_val = kb.scratch["round0_cache"]
    if not preloaded:
        kb.add_node(Instruction("load", ("load", root_val, kb.scratch["forest_values_p"])))
        kb.add_node(Instruction("valu", ("vbroadcast", root_val, root_val)))

    for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * VECTOR_SIZE)):
        for block_id in range(BLOCK_SIZE):
            i = group_id * BLOCK_SIZE * VECTOR_SIZE + block_id * VECTOR_SIZE

            # assign block registers
            tmp1 = kb.scratch["tmp1_v"] + block_id * VECTOR_SIZE
            tmp2 = kb.scratch["tmp2_v"] + block_id * VECTOR_SIZE

            tmp_idx = kb.scratch["idx_array"] + i
            tmp_val = kb.scratch["val_array"] + i

            # val = myhash(val ^ node_val) (vectorized)
            kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, root_val)))
            vhash(kb, tmp1, tmp2, tmp_val)

            # idx = 0 if val % 2 == 0 else 1 (offset idx by -1 for easier select usage)
            kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["const_2"])))
            kb.add_node(Instruction("flow", ("vselect", tmp_idx, tmp1, kb.scratch["const_1"], kb.scratch["const_0"])))


def round_1_kernel(kb: DAGKernelBuilder, preloaded=False):
    # cache tree nodes
    tree_cache = kb.scratch["round1_cache"]
    left_node = tree_cache
    right_node = tree_cache + 8
    if not preloaded:
        kb.add_node(Instruction("load", ("vload", tree_cache, kb.scratch["forest_values_p"])))
        kb.add_node(Instruction("valu", ("vbroadcast", right_node, tree_cache + 2)))
        kb.add_node(Instruction("valu", ("vbroadcast", left_node, tree_cache + 1)))

    for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * VECTOR_SIZE)):
        for block_id in range(BLOCK_SIZE):
            i = group_id * BLOCK_SIZE * VECTOR_SIZE + block_id * VECTOR_SIZE

            # assign block registers
            tmp1 = kb.scratch["tmp1_v"] + block_id * VECTOR_SIZE
            tmp2 = kb.scratch["tmp2_v"] + block_id * VECTOR_SIZE
            tmp3 = kb.scratch["tmp3_v"] + block_id * VECTOR_SIZE

            tmp_idx = kb.scratch["idx_array"] + i
            tmp_val = kb.scratch["val_array"] + i

            tmp_node_val = kb.scratch["tmp_node_val_v"] + i

            # match with tree_node cache
            kb.add_node(Instruction("flow", ("vselect", tmp_node_val, tmp_idx, right_node, left_node)))

            # val = myhash(val ^ node_val) (vectorized)
            kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
            vhash(kb, tmp1, tmp2, tmp_val)

            # idx = 2*(idx + 1) + (1 if val % 2 == 0 else 2) (idx is offset by -1 for easier select usage)
            kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["const_2"])))
            kb.add_node(Instruction("flow", ("vselect", tmp3, tmp1, kb.scratch["const_4"], kb.scratch["const_3"])))
            kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["const_2"], tmp3)))


# TODO: fma the hash
def vhash(kb: DAGKernelBuilder, tmp1: int, tmp2: int, tmp_val: int):
    # Unrolled HASH_STAGES loop
    # Stage 0: ("+", 0x7ED55D16, "+", "<<", 12)
    kb.add_node(Instruction("valu", ("multiply_add", tmp_val, tmp_val, kb.scratch["hash_mult0"], kb.scratch["hash_array1"] + 0 * VECTOR_SIZE)))

    # Stage 1: ("^", 0xC761C23C, "^", ">>", 19)
    kb.add_node(Instruction("valu", ("^", tmp1, tmp_val, kb.scratch["hash_array1"] + 1 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", (">>", tmp2, tmp_val, kb.scratch["hash_array2"] + 1 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("^", tmp_val, tmp1, tmp2)))

    # Stage 2: ("+", 0x165667B1, "+", "<<", 5)
    kb.add_node(Instruction("valu", ("multiply_add", tmp_val, tmp_val, kb.scratch["hash_mult2"], kb.scratch["hash_array1"] + 2 * VECTOR_SIZE)))

    # Stage 3: ("+", 0xD3A2646C, "^", "<<", 9)
    kb.add_node(Instruction("valu", ("+", tmp1, tmp_val, kb.scratch["hash_array1"] + 3 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("<<", tmp2, tmp_val, kb.scratch["hash_array2"] + 3 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("^", tmp_val, tmp1, tmp2)))

    # Stage 4: ("+", 0xFD7046C5, "+", "<<", 3)
    kb.add_node(Instruction("valu", ("multiply_add", tmp_val, tmp_val, kb.scratch["hash_mult4"], kb.scratch["hash_array1"] + 4 * VECTOR_SIZE)))

    # Stage 5: ("^", 0xB55A4F09, "^", ">>", 16)
    kb.add_node(Instruction("valu", ("^", tmp1, tmp_val, kb.scratch["hash_array1"] + 5 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", (">>", tmp2, tmp_val, kb.scratch["hash_array2"] + 5 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("^", tmp_val, tmp1, tmp2)))
