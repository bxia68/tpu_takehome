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
        elif engine == "load" and instruction[0] == "const":
            self.dst = [instruction[1]]
            self.dep_list = []
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
        self.cache_versions = [0] * SCRATCH_SIZE
        self.instruction_count = 0
        self.finished_set = set()

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add_node(Instruction("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

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
        engine_list = ["alu", "valu", "load", "store", "flow"]

        job_queue = deque()
        for current_id, indegree in enumerate(self.indegree):
            if indegree == 0:
                job_queue.append(current_id)

        def signal_war_dependents(instr_list):
            """Signal WAR dependents - they can run in the same cycle"""
            for instr in instr_list:
                for dep in instr.dep_list:
                    for adj_id in self.WAR_graph[(dep, self.dep_version[(instr.graph_id, dep)])]:
                        self.indegree[adj_id] -= 1
                        if self.indegree[adj_id] == 0:
                            job_queue.append(adj_id)

        def signal_raw_dependents(instr_list):
            """Signal RAW dependents - they must wait for next cycle"""
            for instr in instr_list:
                for dst in instr.dst:
                    for adj_id in self.RAW_graph[(dst, self.dst_version[(instr.graph_id, dst)])]:
                        self.indegree[adj_id] -= 1
                        if self.indegree[adj_id] == 0:
                            job_queue.append(adj_id)

        def drain_job_queue():
            """Move ready instructions from job_queue to engine_queue"""
            while len(job_queue) > 0:
                current_id = job_queue.popleft()
                instruction = self.instruction_list[current_id]
                engine_queue[instruction.engine].append(instruction)

        def pack_engines(cycle_instructions):
            """Pack instructions from engine queues into cycle, return newly added"""
            newly_added = []
            for engine in engine_list:
                engine_queue[engine].sort(key=lambda x: x.graph_id)
                avail_slots = SLOT_LIMITS[engine] - len(cycle_instructions.get(engine, []))
                if avail_slots > 0 and len(engine_queue[engine]) > 0:
                    to_add = engine_queue[engine][:avail_slots]
                    cycle_instructions[engine] = cycle_instructions.get(engine, []) + [x.instruction for x in to_add]
                    newly_added.extend(to_add)
                    engine_queue[engine] = engine_queue[engine][avail_slots:]
            return newly_added

        pending_valu_half = None
        while len(job_queue) > 0 or any(engine_queue.values()):
            drain_job_queue()

            cycle_instructions = {}
            finished_list = []

            # handle pending valu half from previous cycle (max priority)
            if pending_valu_half is not None:
                valu_instr, half_start = pending_valu_half
                op, dest, a1, a2 = valu_instr.instruction
                cycle_instructions["alu"] = [(op, dest + i, a1 + i, a2 + i) for i in range(half_start, 8)]
                finished_list.append(valu_instr)
                signal_war_dependents([valu_instr])
                drain_job_queue()
                pending_valu_half = None

            # pack instructions and process WAR dependents until no more can be added
            while True:
                newly_added = pack_engines(cycle_instructions)
                if not newly_added:
                    break
                finished_list.extend(newly_added)
                signal_war_dependents(newly_added)
                drain_job_queue()

            # convert valu to alu ops if space available
            alu_avail_slots = SLOT_LIMITS["alu"] - len(cycle_instructions.get("alu", []))
            while alu_avail_slots >= 4 and len(engine_queue["valu"]) > 0:
                found_idx = next((i for i, v in enumerate(engine_queue["valu"]) if v.instruction[0] not in ("vbroadcast", "multiply_add")), None)
                if found_idx is None:
                    break

                valu_instr = engine_queue["valu"].pop(found_idx)
                op, dest, a1, a2 = valu_instr.instruction

                if alu_avail_slots >= 8:
                    cycle_instructions["alu"] = cycle_instructions.get("alu", []) + [(op, dest + i, a1 + i, a2 + i) for i in range(8)]
                    finished_list.append(valu_instr)
                    signal_war_dependents([valu_instr])
                    alu_avail_slots -= 8
                else:
                    cycle_instructions["alu"] = cycle_instructions.get("alu", []) + [(op, dest + i, a1 + i, a2 + i) for i in range(4)]
                    pending_valu_half = (valu_instr, 4)
                    alu_avail_slots -= 4

            # convert load const to flow add_imm
            if len(cycle_instructions.get("flow", [])) == 0:
                found_idx = next((i for i, l in enumerate(engine_queue["load"]) if l.instruction[0] == "const"), None)
                if found_idx is not None:
                    load_instr = engine_queue["load"].pop(found_idx)
                    _, dest, val = load_instr.instruction
                    cycle_instructions["flow"] = [("add_imm", dest, self.const_map[0], val)]
                    finished_list.append(load_instr)
                    signal_war_dependents([load_instr])

            compiled_instructions.append(cycle_instructions)

            # Signal RAW dependents - they must wait for next cycle
            signal_raw_dependents(finished_list)

        self.instrs.extend(compiled_instructions)
        self.clear()
