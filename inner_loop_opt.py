from problem import HASH_STAGES

from DAG_kernel_builder import DAGKernelBuilder, Instruction

CHARACTER_COUNT = 256
ROUNDS = 16
BLOCK_SIZE = 8
TILE_SIZE = 2
VECTOR_SIZE = 8
LEVEL_COUNT = 11


def kernel(kb: DAGKernelBuilder):
    # vector const registers
    for i in range(6):
        const_v = kb.alloc_scratch(f"const_{i}", VECTOR_SIZE)
        kb.add_node(Instruction("valu", ("vbroadcast", const_v, kb.scratch_const(i))))

    tree_ptr_v = kb.alloc_scratch("tree_ptr_v", VECTOR_SIZE)
    kb.add_node(Instruction("valu", ("vbroadcast", tree_ptr_v, kb.scratch["forest_values_p"])))

    hash_array1 = kb.alloc_scratch("hash_array1", 6 * VECTOR_SIZE)
    hash_array2 = kb.alloc_scratch("hash_array2", 6 * VECTOR_SIZE)
    for i in range(len(HASH_STAGES)):
        kb.add_node(Instruction("valu", ("vbroadcast", hash_array1 + i * VECTOR_SIZE, kb.scratch_const(HASH_STAGES[i][1]))))
        kb.add_node(Instruction("valu", ("vbroadcast", hash_array2 + i * VECTOR_SIZE, kb.scratch_const(HASH_STAGES[i][4]))))

    # extra pre computed hash values for fma
    hash_mult0 = kb.alloc_scratch("hash_mult0", VECTOR_SIZE)  # 2^12 + 1 = 4097
    hash_mult2 = kb.alloc_scratch("hash_mult2", VECTOR_SIZE)  # 2^5 + 1 = 33
    hash_mult4 = kb.alloc_scratch("hash_mult4", VECTOR_SIZE)  # 2^3 + 1 = 9
    kb.add_node(Instruction("valu", ("vbroadcast", hash_mult0, kb.scratch_const(4097))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_mult2, kb.scratch_const(33))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_mult4, kb.scratch_const(9))))

    # vector/scalar scratch registers
    tmp1_v = kb.alloc_scratch("tmp1_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)
    tmp2_v = kb.alloc_scratch("tmp2_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)
    tmp3_v = kb.alloc_scratch("tmp3_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)

    tmp_node_val_v = kb.alloc_scratch("tmp_node_val_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)
    tmp_addr_v = kb.alloc_scratch("tmp_addr_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)

    kb.alloc_scratch("round0_cache", 8)
    kb.alloc_scratch("round1_cache", 16)
    kb.alloc_scratch("round2_cache", 32)

    kb.add_node(Instruction("load", ("load", kb.scratch["round0_cache"], kb.scratch["forest_values_p"])))
    kb.add_node(Instruction("valu", ("vbroadcast", kb.scratch["round0_cache"], kb.scratch["round0_cache"])))

    kb.add_node(Instruction("load", ("vload", kb.scratch["round1_cache"], kb.scratch["forest_values_p"])))
    kb.add_node(Instruction("valu", ("vbroadcast", kb.scratch["round1_cache"] + 8, kb.scratch["round1_cache"] + 2)))
    kb.add_node(Instruction("valu", ("vbroadcast", kb.scratch["round1_cache"], kb.scratch["round1_cache"] + 1)))

    idx_array = kb.alloc_scratch("idx_array", CHARACTER_COUNT)
    val_array = kb.alloc_scratch("val_array", CHARACTER_COUNT)
    for i in range(0, CHARACTER_COUNT, VECTOR_SIZE):
        tmp_addr_1 = tmp_addr_v
        tmp_addr_2 = tmp_addr_v + 1
        kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("load", ("vload", idx_array + i, tmp_addr_1)))
        kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("load", ("vload", val_array + i, tmp_addr_2)))

    # launch compute kernel
    generic_kernel(kb)

    for i in range(0, CHARACTER_COUNT, VECTOR_SIZE):
        tmp_addr_1 = tmp_addr_v
        tmp_addr_2 = tmp_addr_v + 1
        kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_1, idx_array + i)))
        kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_2, val_array + i)))

    kb.compile_kernel()

    # Required to match with the yield in reference_kernel2
    kb.instrs.append({"flow": [("pause",)]})


HANDOFF = 9


def generic_kernel(kb: DAGKernelBuilder):
    visited = [False] * LEVEL_COUNT
    for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)):
        for block_id in range(BLOCK_SIZE):
            # for round_num in range(ROUNDS):
            for round_num in range(HANDOFF):
                level = round_num % LEVEL_COUNT
                if level == 0:
                    round_0_kernel(kb, group_id, block_id, visited[level])
                elif level == 1:
                    round_1_kernel(kb, group_id, block_id, visited[level])
                elif level == 2:
                    round_2_kernel(kb, group_id, block_id, visited[level])
                else:
                    generic_round_kernel(kb, group_id, block_id, round_num)
                # generic_round_kernel(kb, group_id, block_id, round_num)
                visited[level] = True

    for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)):
        for block_id in range(BLOCK_SIZE):
            for round_num in range(HANDOFF, ROUNDS):
                level = round_num % LEVEL_COUNT
                if level == 0:
                    round_0_kernel(kb, group_id, block_id, visited[level])
                elif level == 1:
                    round_1_kernel(kb, group_id, block_id, visited[level])
                elif level == 2:
                    round_2_kernel(kb, group_id, block_id, visited[level])
                else:
                    generic_round_kernel(kb, group_id, block_id, round_num)
                # generic_round_kernel(kb, group_id, block_id, round_num)
                visited[level] = True


def generic_round_kernel(kb: DAGKernelBuilder, group_id: int, block_id: int, round_num: int):
    for tile_id in range(TILE_SIZE):
        # each iteration will complete 8 inputs at a time
        i = group_id * BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # assign block registers
        tmp1 = kb.scratch["tmp1_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp2 = kb.scratch["tmp2_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp3 = kb.scratch["tmp3_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        tmp_node_val = kb.scratch["tmp_node_val_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp_addr = kb.scratch["tmp_addr_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

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


def round_0_kernel(kb: DAGKernelBuilder, group_id: int, block_id: int, preloaded: bool):
    # pull tree nodes (all nodes are the same root node)
    root_val = kb.scratch["round0_cache"]
    # if not preloaded:
        # kb.add_node(Instruction("load", ("load", root_val, kb.scratch["forest_values_p"])))
        # kb.add_node(Instruction("valu", ("vbroadcast", root_val, root_val)))
        # kb.loaded_node_cache = [False] * (CHARACTER_COUNT // VECTOR_SIZE)

    for tile_id in range(TILE_SIZE):
        # each iteration will complete 8 inputs at a time
        i = group_id * BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # assign block registers
        tmp1 = kb.scratch["tmp1_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp2 = kb.scratch["tmp2_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        tmp_idx = kb.scratch["idx_array"] + i
        tmp_val = kb.scratch["val_array"] + i

        # if not kb.loaded_node_cache[i // VECTOR_SIZE]:
        #     tmp_addr_1 = kb.scratch["tmp_addr_v"] + (i // VECTOR_SIZE)
        #     tmp_addr_2 = kb.scratch["tmp_addr_v"] + (i // VECTOR_SIZE) + 1
        #     kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        #     kb.add_node(Instruction("load", ("vload", tmp_idx, tmp_addr_1)))
        #     kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        #     kb.add_node(Instruction("load", ("vload", tmp_val, tmp_addr_2)))
        #     kb.loaded_node_cache[i // VECTOR_SIZE] = True

        # val = myhash(val ^ node_val) (vectorized)
        kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, root_val)))
        vhash(kb, tmp1, tmp2, tmp_val)

        # idx = 0 if val % 2 == 0 else 1 (offset idx by -1 for easier select usage)
        kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["const_2"])))
        kb.add_node(Instruction("flow", ("vselect", tmp_idx, tmp1, kb.scratch["const_1"], kb.scratch["const_0"])))


def round_1_kernel(kb: DAGKernelBuilder, group_id: int, block_id: int, preloaded: bool):
    # cache tree nodes
    tree_cache = [kb.scratch["round1_cache"] + i * 8 for i in range(2)]
    # if not preloaded:
    #     kb.add_node(Instruction("load", ("vload", tree_cache, kb.scratch["forest_values_p"])))
    #     kb.add_node(Instruction("valu", ("vbroadcast", right_node, tree_cache + 2)))
    #     kb.add_node(Instruction("valu", ("vbroadcast", left_node, tree_cache + 1)))

    for tile_id in range(TILE_SIZE):
        # each iteration will complete 8 inputs at a time
        i = group_id * BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # assign block registers
        tmp1 = kb.scratch["tmp1_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp2 = kb.scratch["tmp2_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp3 = kb.scratch["tmp3_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        tmp_idx = kb.scratch["idx_array"] + i
        tmp_val = kb.scratch["val_array"] + i

        tmp_node_val = kb.scratch["tmp_node_val_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # match with tree_node cache
        kb.add_node(Instruction("flow", ("vselect", tmp_node_val, tmp_idx, tree_cache[1], tree_cache[0])))

        # val = myhash(val ^ node_val) (vectorized)
        kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
        vhash(kb, tmp1, tmp2, tmp_val)

        # idx = 2*(idx + 1) + (1 if val % 2 == 0 else 2) (idx is offset by -1 for easier select usage)
        kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["const_2"])))
        kb.add_node(Instruction("flow", ("vselect", tmp3, tmp1, kb.scratch["const_4"], kb.scratch["const_3"])))
        kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["const_2"], tmp3)))


def round_2_kernel(kb: DAGKernelBuilder, group_id: bool, block_id: int, preloaded: bool):
    # cache tree nodes
    tree_cache = [kb.scratch["round2_cache"] + i * 8 for i in range(4)]
    if not preloaded:
        # tileed load tree node values [0:8]
        kb.add_node(Instruction("load", ("vload", tree_cache[0], kb.scratch["forest_values_p"])))

        # broadcast tree node values to fill the cache
        kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[3], tree_cache[0] + 6)))
        kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[2], tree_cache[0] + 5)))
        kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[1], tree_cache[0] + 4)))
        kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[0], tree_cache[0] + 3)))

    for tile_id in range(TILE_SIZE):
        # each iteration will complete 8 inputs at a time
        i = group_id * BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # assign block registers
        tmp1 = kb.scratch["tmp1_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp2 = kb.scratch["tmp2_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp3 = kb.scratch["tmp3_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        tmp_idx = kb.scratch["idx_array"] + i
        tmp_val = kb.scratch["val_array"] + i

        tmp_node_val = kb.scratch["tmp_node_val_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # match with tree_node cache
        kb.add_node(Instruction("valu", ("%", tmp3, tmp_idx, kb.scratch["const_2"])))
        kb.add_node(Instruction("flow", ("vselect", tmp1, tmp3, tree_cache[0], tree_cache[1])))
        kb.add_node(Instruction("flow", ("vselect", tmp2, tmp3, tree_cache[2], tree_cache[3])))

        kb.add_node(Instruction("valu", ("<", tmp3, tmp_idx, kb.scratch["const_5"])))
        kb.add_node(Instruction("flow", ("vselect", tmp_node_val, tmp3, tmp1, tmp2)))

        # val = myhash(val ^ node_val) (vectorized)
        kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
        vhash(kb, tmp1, tmp2, tmp_val)

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["const_2"])))
        kb.add_node(Instruction("flow", ("vselect", tmp3, tmp1, kb.scratch["const_2"], kb.scratch["const_1"])))
        kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["const_2"], tmp3)))


# 12 valu instructions
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
