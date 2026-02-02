from problem import Engine, DebugInfo, SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE, Machine, Tree, Input, HASH_STAGES, reference_kernel, build_mem_image, reference_kernel2

from dataclasses import dataclass
from collections import defaultdict, deque
import json

from kernel_builder import KernelBuilder


class Instruction:
    graph_id: int
    engine: str
    instruction: tuple
    dst: list[int]
    dep_list: list[int]

    def __init__(self, engine: str, instruction: tuple):
        self.engine = engine
        self.instruction = instruction

        if engine == "store" or instruction[0] == "trace_write":
            self.dst = []
            self.dep_list = instruction[1:]
        else:
            self.dst = [instruction[1]]
            self.dep_list = instruction[2:]

        if engine == "valu" or instruction[0] == "vselect" or instruction[0] == "vload":
            self.dst = getVectorAddrs(self.dst[0])
        if (engine == "valu" and instruction[0] != "vbroadcast") or instruction[0] == "vselect":
            # For valu (except vbroadcast) and vselect, dependencies are vectors
            tmp = self.dep_list
            self.dep_list = []
            for dep in tmp:
                self.dep_list.extend(getVectorAddrs(dep))
        elif instruction[0] == "vstore":
            # vstore: addr is scalar, src is vector
            # dep_list = (addr, src) -> addr stays scalar, src becomes vector
            addr = self.dep_list[0]  # scalar address
            src = self.dep_list[1]  # vector source
            self.dep_list = [addr] + getVectorAddrs(src)


def getVectorAddrs(addr: int) -> list[int]:
    return [addr + i for i in range(8)]


class DAGKernelBuilder(KernelBuilder):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.RAW_graph = defaultdict(list)  # (cache addr, version) (int, int) -> instruction ids (list[int])
        self.WAR_graph = defaultdict(list)  # (cache addr, version) (int, int) -> instruction ids (list[int])
        self.indegree = []  # instruction id (int) -> indegree (int)
        self.instruction_list = []  # instruction id (int) -> instruction
        self.dst_version = {}  # (instruction id, cache addr) (int, int) -> dst cache version (int)
        self.dep_version = {}  # (instruction id, cache addr) (int, int) -> dep cache version (int)
        self.WAR_indegree = defaultdict(int)
        self.cache_versions = [0] * 1536
        self.instruction_count = 0
        self.finished_set = set()

    def add_node(self, instruction: Instruction):
        # add instruction
        instruction.graph_id = self.instruction_count
        self.instruction_count += 1
        self.instruction_list.append(instruction)
        self.indegree.append(0)

        # populate dependency graph
        for dep in instruction.dep_list:
            if self.cache_versions[dep] > 0:
                self.RAW_graph[(dep, self.cache_versions[dep])].append(instruction.graph_id)
                self.indegree[instruction.graph_id] += 1
            self.WAR_indegree[(dep, self.cache_versions[dep])] += 1
            self.dep_version[(instruction.graph_id, dep)] = self.cache_versions[dep]

        for dst in instruction.dst:
            self.WAR_graph[(dst, self.cache_versions[dst])].append(instruction.graph_id)
            self.indegree[instruction.graph_id] += self.WAR_indegree[(dst, self.cache_versions[dst])]
            if dst in instruction.dep_list:
                self.indegree[instruction.graph_id] -= 1
            self.cache_versions[dst] += 1
            self.dst_version[(instruction.graph_id, dst)] = self.cache_versions[dst]

    def compile_kernel(self) -> list[dict]:
        compiled_instructions = []
        engine_queue = {"alu": [], "valu": [], "load": [], "store": [], "flow": []}

        job_queue = deque()
        for current_id, indegree in enumerate(self.indegree):
            if indegree == 0:
                job_queue.append(current_id)

        pending_valu_half = None
        while len(job_queue) > 0 or len([val for queue in engine_queue.values() for val in queue]) > 0:
            while len(job_queue) > 0:
                current_id = job_queue.popleft()
                instruction = self.instruction_list[current_id]
                engine_queue[instruction.engine].append(instruction)

            engine_list = ["alu", "valu", "load", "store", "flow"]
            cycle_instructions = {}
            finished_list = []

            # handle pending valu half from previous cycle (max priority)
            if pending_valu_half is not None:
                valu_instr, half_start = pending_valu_half
                op, dest, a1, a2 = valu_instr.instruction
                cycle_instructions["alu"] = [(op, dest + i, a1 + i, a2 + i) for i in range(half_start, 8)]
                finished_list.append(valu_instr)
                pending_valu_half = None

            # pack instructions from the queue into the current cycle
            for engine in engine_list:
                engine_queue[engine].sort(key=lambda x: x.graph_id)
                if len(engine_queue[engine]) > 0:
                    cycle_instructions[engine] = cycle_instructions.get(engine, []) + [x.instruction for x in engine_queue[engine][: SLOT_LIMITS[engine]]]
                    finished_list.extend(engine_queue[engine][: SLOT_LIMITS[engine]])
                    engine_queue[engine] = engine_queue[engine][SLOT_LIMITS[engine] :]

            # convert valu to alu ops if space available
            alu_slots_available = SLOT_LIMITS["alu"] - len(cycle_instructions.get("alu", []))
            while alu_slots_available >= 4 and len(engine_queue["valu"]) > 0:
                # find a convertible valu instruction
                found_idx = None
                for idx, valu_instr in enumerate(engine_queue["valu"]):
                    if valu_instr.instruction[0] not in ("vbroadcast", "multiply_add"):
                        found_idx = idx
                        break

                if found_idx is None:
                    break

                valu_instr = engine_queue["valu"].pop(found_idx)
                op, dest, a1, a2 = valu_instr.instruction

                if alu_slots_available >= 8:
                    # full convert valu to alu instruction
                    cycle_instructions["alu"] = cycle_instructions.get("alu", []) + [(op, dest + i, a1 + i, a2 + i) for i in range(8)]
                    finished_list.append(valu_instr)
                    alu_slots_available -= 8
                else:
                    # split value instruction
                    cycle_instructions["alu"] = cycle_instructions.get("alu", []) + [(op, dest + i, a1 + i, a2 + i) for i in range(4)]
                    pending_valu_half = (valu_instr, 4)
                    alu_slots_available -= 4

            compiled_instructions.append(cycle_instructions)

            # signal dependent instructions
            for finished in finished_list:
                dependents = []
                for dst in finished.dst:
                    dependents.extend(self.RAW_graph[(dst, self.dst_version[(finished.graph_id, dst)])])
                for dep in finished.dep_list:
                    dependents.extend(self.WAR_graph[(dep, self.dep_version[(finished.graph_id, dep)])])
                for adj_id in dependents:
                    self.indegree[adj_id] -= 1
                    if self.indegree[adj_id] == 0:
                        job_queue.append(adj_id)

        self.instrs.extend(compiled_instructions)
        self.clear()
