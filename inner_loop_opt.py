from problem import HASH_STAGES

from DAG_kernel_builder import DAGKernelBuilder, Instruction
import os

CHARACTER_COUNT = 256
ROUNDS = 16
VECTOR_SIZE = 8
LEVEL_COUNT = 11

BLOCK_SIZE = 4
TILE_SIZE = 4
BLOCK_SIZE1 = int(os.getenv("BLOCK_SIZE", 4))
TILE_SIZE1 = int(os.getenv("TILE_SIZE", 4))
SYNC_LIST = list(map(int, os.getenv("SYNC_LIST", "14,15").split(","))) + [16]


def kernel(kb: DAGKernelBuilder):
    # vector const registers
    for i in list(range(6)) + [9, 11, 13]:
        const_v = kb.alloc_scratch(f"const_{i}", VECTOR_SIZE)
        kb.add_node(Instruction("valu", ("vbroadcast", const_v, kb.scratch_const(i))))

    tree_ptr_v = kb.alloc_scratch("tree_ptr_v", VECTOR_SIZE)
    kb.add_node(Instruction("valu", ("vbroadcast", tree_ptr_v, kb.scratch["forest_values_p"])))

    hash_array1 = kb.alloc_scratch("hash_array1", 6 * VECTOR_SIZE)
    hash_array2 = kb.alloc_scratch("hash_array2", 6 * VECTOR_SIZE)
    # stage 0: val * 4097 + 0x7ED55D16
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array1, kb.scratch_const(0x7ED55D16))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array2, kb.scratch_const(4097))))
    # stage 1: (val ^ 0xC761C23C) ^ (val >> 19)
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array1 + 1 * VECTOR_SIZE, kb.scratch_const(0xC761C23C))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array2 + 1 * VECTOR_SIZE, kb.scratch_const(19))))
    # stage 2+3 optimized: tmp1 = val*33 + (const2 + const3), tmp2 = val*(33*512) + (const2 << 9) = new_val << 9
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array1 + 2 * VECTOR_SIZE, kb.scratch_const((0x165667B1 + 0xD3A2646C) % (2**32)))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array2 + 2 * VECTOR_SIZE, kb.scratch_const(33))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array1 + 3 * VECTOR_SIZE, kb.scratch_const((0x165667B1 << 9) % (2**32)))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array2 + 3 * VECTOR_SIZE, kb.scratch_const(33 * 512))))
    # stage 4: val * 9 + 0xFD7046C5
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array1 + 4 * VECTOR_SIZE, kb.scratch_const(0xFD7046C5))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array2 + 4 * VECTOR_SIZE, kb.scratch_const(9))))
    # stage 5: (val ^ 0xB55A4F09) ^ (val >> 16)
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array1 + 5 * VECTOR_SIZE, kb.scratch_const(0xB55A4F09))))
    kb.add_node(Instruction("valu", ("vbroadcast", hash_array2 + 5 * VECTOR_SIZE, kb.scratch_const(16))))

    # vector/scalar scratch registers
    tmp1_v = kb.alloc_scratch("tmp1_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)
    tmp2_v = kb.alloc_scratch("tmp2_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)
    tmp3_v = kb.alloc_scratch("tmp3_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)
    # tmp4_v = kb.alloc_scratch("tmp4_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)

    tmp_node_val_v = kb.alloc_scratch("tmp_node_val_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)
    tmp_addr_v = kb.alloc_scratch("tmp_addr_v", BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)

    kb.alloc_scratch("round0_cache", 8)
    kb.alloc_scratch("round1_cache", 16)
    kb.alloc_scratch("round2_cache", 32)
    kb.alloc_scratch("round3_cache", 64)

    # load tree caches

    # round 0 cache
    kb.add_node(Instruction("load", ("load", kb.scratch["round0_cache"], kb.scratch["forest_values_p"])))
    kb.add_node(Instruction("valu", ("vbroadcast", kb.scratch["round0_cache"], kb.scratch["round0_cache"])))

    # round 1 cache
    kb.add_node(Instruction("load", ("vload", kb.scratch["round1_cache"], kb.scratch["forest_values_p"])))
    kb.add_node(Instruction("valu", ("vbroadcast", kb.scratch["round1_cache"] + 8, kb.scratch["round1_cache"] + 2)))
    kb.add_node(Instruction("valu", ("vbroadcast", kb.scratch["round1_cache"], kb.scratch["round1_cache"] + 1)))

    # round 2 cache
    tree_cache = [kb.scratch["round2_cache"] + i * 8 for i in range(4)]
    kb.add_node(Instruction("load", ("vload", tree_cache[0], kb.scratch["forest_values_p"])))
    kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[3], tree_cache[0] + 6)))
    kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[2], tree_cache[0] + 5)))
    kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[1], tree_cache[0] + 4)))
    kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[0], tree_cache[0] + 3)))

    # round 3 cache
    tree_cache = [kb.scratch["round3_cache"] + i * 8 for i in range(8)]
    kb.add_node(Instruction("alu", ("+", tree_cache[0], kb.scratch["forest_values_p"], kb.scratch_const(7))))
    kb.add_node(Instruction("load", ("vload", tree_cache[0], tree_cache[0])))
    for i in reversed(range(8)):
        kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[i], tree_cache[0] + i)))

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
        tmp_addr_1 = tmp1_v + (i // VECTOR_SIZE)
        tmp_addr_2 = tmp2_v + (i // VECTOR_SIZE)
        kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_1, idx_array + i)))
        kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_2, val_array + i)))

    counts = {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0}
    for node in kb.instruction_list:
        if node.engine in counts:
            counts[node.engine] += 1

    cycles = {"alu+valu": counts["alu"] / 12 + counts["valu"] / 7.5, "load": counts["load"] / 2, "store": counts["store"] / 2, "flow": counts["flow"]}
    total_cycles = max(cycles.values())

    # ANSI color codes
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{CYAN}╔════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║     KERNEL CYCLE SUMMARY               ║{RESET}")
    print(f"{BOLD}{CYAN}╠════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}  ALU+VALU:  {cycles['alu+valu']:>8.1f} cycles            {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}  Load:      {cycles['load']:>8.1f} cycles            {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}  Store:     {cycles['store']:>8.1f} cycles            {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}  Flow:      {cycles['flow']:>8.1f} cycles            {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╠════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}  {BOLD}TOTAL:     {total_cycles:>8.1f} cycles{RESET}            {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════╝{RESET}")
    print(f"{DIM}  (ALU: {counts['alu']}, VALU: {counts['valu']}, Load: {counts['load']}, Store: {counts['store']}, Flow: {counts['flow']}){RESET}\n")

    kb.compile_kernel()

    # Required to match with the yield in reference_kernel2
    kb.instrs.append({"flow": [("pause",)]})


def generic_kernel(kb: DAGKernelBuilder):
    visited = [False] * LEVEL_COUNT
    prev_sync_point = 0
    for sync_point in SYNC_LIST:
        for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE)):
            for block_id in range(BLOCK_SIZE):
                # for round_num in range(ROUNDS):
                for round_num in range(prev_sync_point, sync_point):
                    level = round_num % LEVEL_COUNT
                    if level == 0:
                        round_0_kernel(kb, group_id, block_id, visited[level])
                    elif level == 1:
                        round_1_kernel(kb, group_id, block_id, visited[level])
                    elif level == 2:
                        round_2_kernel(kb, group_id, block_id, visited[level])
                    elif level == 3:
                        round_3_kernel(kb, group_id, block_id, visited[level])
                    else:
                        generic_round_kernel(kb, group_id, block_id, round_num)
                    # generic_round_kernel(kb, group_id, block_id, round_num)
                    visited[level] = True
        prev_sync_point = sync_point


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
        # else:
        #     # idx = root (0) if on level 10
        #     kb.add_node(Instruction("valu", ("&", tmp_idx, kb.scratch["const_0"], kb.scratch["const_0"])))


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

        # val = myhash(val ^ node_val) (vectorized)
        kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, root_val)))
        vhash(kb, tmp1, tmp2, tmp_val)

        # idx = 0 if val % 2 == 0 else 1 (offset idx by -1 for easier future select usage)
        kb.add_node(Instruction("valu", ("&", tmp_idx, tmp_val, kb.scratch["const_1"])))


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

        # idx = 2*(idx + 1) + (1 if val % 2 == 0 else 2) (idx is offset by -1 for easier future select usage)
        kb.add_node(Instruction("valu", ("&", tmp1, tmp_val, kb.scratch["const_1"])))
        kb.add_node(Instruction("valu", ("+", tmp3, tmp1, kb.scratch["const_3"])))
        kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["const_2"], tmp3)))


def round_2_kernel(kb: DAGKernelBuilder, group_id: bool, block_id: int, preloaded: bool):
    # cache tree nodes
    tree_cache = [kb.scratch["round2_cache"] + i * 8 for i in range(4)]
    # if not preloaded:
    #     # load tree node values [0:8]
    #     kb.add_node(Instruction("load", ("vload", tree_cache[0], kb.scratch["forest_values_p"])))

    #     # broadcast tree node values to fill the cache
    #     kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[3], tree_cache[0] + 6)))
    #     kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[2], tree_cache[0] + 5)))
    #     kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[1], tree_cache[0] + 4)))
    #     kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[0], tree_cache[0] + 3)))

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


def round_3_kernel(kb: DAGKernelBuilder, group_id: bool, block_id: int, preloaded: bool):
    # cache tree nodes
    tree_cache = [kb.scratch["round3_cache"] + i * 8 for i in range(8)]
    # if not preloaded:
    #     # load tree node values [7:15]
    #     kb.add_node(Instruction("alu", ("+", tree_cache[0], kb.scratch["forest_values_p"], kb.scratch_const(7))))
    #     kb.add_node(Instruction("load", ("vload", tree_cache[0], tree_cache[0])))

    #     # broadcast tree node values to fill the cache
    #     for i in reversed(range(8)):
    #         kb.add_node(Instruction("valu", ("vbroadcast", tree_cache[i], tree_cache[0] + i)))

    for tile_id in range(TILE_SIZE):
        # each iteration will complete 8 inputs at a time
        i = group_id * BLOCK_SIZE * TILE_SIZE * VECTOR_SIZE + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # assign block registers
        tmp1 = kb.scratch["tmp1_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp2 = kb.scratch["tmp2_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp3 = kb.scratch["tmp3_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        # tmp4 = kb.scratch["tmp4_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE
        tmp4 = kb.scratch["tmp_addr_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        tmp_idx = kb.scratch["idx_array"] + i
        tmp_val = kb.scratch["val_array"] + i

        tmp_node_val = kb.scratch["tmp_node_val_v"] + block_id * TILE_SIZE * VECTOR_SIZE + tile_id * VECTOR_SIZE

        # match with tree_node cache
        kb.add_node(Instruction("valu", ("%", tmp_node_val, tmp_idx, kb.scratch["const_2"])))
        kb.add_node(Instruction("flow", ("vselect", tmp1, tmp_node_val, tree_cache[0], tree_cache[1])))
        kb.add_node(Instruction("flow", ("vselect", tmp2, tmp_node_val, tree_cache[2], tree_cache[3])))
        kb.add_node(Instruction("flow", ("vselect", tmp3, tmp_node_val, tree_cache[4], tree_cache[5])))
        kb.add_node(Instruction("flow", ("vselect", tmp4, tmp_node_val, tree_cache[6], tree_cache[7])))

        kb.add_node(Instruction("valu", ("<", tmp_node_val, tmp_idx, kb.scratch["const_9"])))
        kb.add_node(Instruction("flow", ("vselect", tmp1, tmp_node_val, tmp1, tmp2)))
        kb.add_node(Instruction("valu", ("<", tmp_node_val, tmp_idx, kb.scratch["const_13"])))
        kb.add_node(Instruction("flow", ("vselect", tmp2, tmp_node_val, tmp3, tmp4)))

        kb.add_node(Instruction("valu", ("<", tmp_node_val, tmp_idx, kb.scratch["const_11"])))
        kb.add_node(Instruction("flow", ("vselect", tmp_node_val, tmp_node_val, tmp1, tmp2)))

        # val = myhash(val ^ node_val) (vectorized)
        kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
        vhash(kb, tmp1, tmp2, tmp_val)

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["const_2"])))
        kb.add_node(Instruction("flow", ("vselect", tmp3, tmp1, kb.scratch["const_2"], kb.scratch["const_1"])))
        kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["const_2"], tmp3)))


# 12 valu instructions
def vhash(kb: DAGKernelBuilder, tmp1: int, tmp2: int, tmp_val: int):
    # unrolled HASH_STAGES loop
    # stage 0: val = (val << 12) + val + 0x7ED55D16 = val * 4097 + 0x7ED55D16
    kb.add_node(Instruction("valu", ("multiply_add", tmp_val, tmp_val, kb.scratch["hash_array2"], kb.scratch["hash_array1"])))

    # stage 1: val = (val ^ 0xC761C23C) ^ (val >> 19)
    kb.add_node(Instruction("valu", ("^", tmp1, tmp_val, kb.scratch["hash_array1"] + 1 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", (">>", tmp2, tmp_val, kb.scratch["hash_array2"] + 1 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("^", tmp_val, tmp1, tmp2)))

    # stage 2+3 optimized: val = val * 33 + (0x165667B1 + 0xD3A2646C), then shift uses FMA to undo add
    kb.add_node(Instruction("valu", ("multiply_add", tmp1, tmp_val, kb.scratch["hash_array2"] + 2 * VECTOR_SIZE, kb.scratch["hash_array1"] + 2 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("multiply_add", tmp2, tmp_val, kb.scratch["hash_array2"] + 3 * VECTOR_SIZE, kb.scratch["hash_array1"] + 3 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("^", tmp_val, tmp1, tmp2)))

    # stage 4: val = (val << 3) + val + 0xFD7046C5 = val * 9 + 0xFD7046C5
    kb.add_node(Instruction("valu", ("multiply_add", tmp_val, tmp_val, kb.scratch["hash_array2"] + 4 * VECTOR_SIZE, kb.scratch["hash_array1"] + 4 * VECTOR_SIZE)))

    # stage 5: val = (val ^ 0xB55A4F09) ^ (val >> 16)
    kb.add_node(Instruction("valu", ("^", tmp1, tmp_val, kb.scratch["hash_array1"] + 5 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", (">>", tmp2, tmp_val, kb.scratch["hash_array2"] + 5 * VECTOR_SIZE)))
    kb.add_node(Instruction("valu", ("^", tmp_val, tmp1, tmp2)))
