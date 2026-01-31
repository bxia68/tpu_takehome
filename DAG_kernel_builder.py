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

from dataclasses import dataclass
from collections import defaultdict, deque
import json

from kernel_builder import (
    KernelBuilder
)


"""
instrs
{
    "engine 1": [slots],
    "engine 2": [slots],
}
"""

test1 = True
test2 = 0

class Instruction:
    graph_id: int
    engine: str
    instruction: tuple
    dst: list[int]
    dep_list: list[int]
    
    def __init__(self, engine: str, instruction: tuple):
        self.engine = engine
        self.instruction = instruction
        
        if engine == "store":
            self.dst = []
            self.dep_list = instruction[1:]
        else:
            self.dst = [instruction[1]]
            self.dep_list = instruction[2:]
        
        if (engine == "valu" or 
            instruction[0] == "vselect" or
            instruction[0] == "vload"):
            self.dst = getVectorAddrs(self.dst[0])
        if ((engine == "valu" and instruction[0] != "vbroadcast") or
            instruction[0] == "vselect"):
            tmp = self.dep_list
            self.dep_list = []
            for dep in tmp:
                self.dep_list.extend(getVectorAddrs(dep))
        elif engine == "vstore":
            self.dep_list.extend(self.dep_list.pop())
            
        # global test1
        # global test2
        # if test1 and engine == "store":
        #     test1 = False
        #     print("\n\n")
        #     print(instruction)
        #     print(self.dep_list)
        #     print(self.dst)
    
        # if test2 < 50 and engine == "alu":
        # if test2 < 50:
        #     test2 += 1
        #     print("\n\n")
        #     print(instruction)
        #     print(self.dep_list)
        #     print(self.dst)
    
def getVectorAddrs(addr: int) -> list[int]:
    return [addr + i for i in range(8)]

class DAGKernelBuilder(KernelBuilder):
    def __init__(self):
        super().__init__()
        self.RAW_graph = defaultdict(list)          # (cache addr, version) (int, int) -> instruction ids (list[int])
        self.WAR_graph = defaultdict(list)          # (cache addr, version) (int, int) -> instruction ids (list[int])
        self.indegree = []                          # instruction id (int) -> indegree (int)
        self.instruction_list = []                  # instruction id (int) -> instruction
        self.dst_version = {}                       # (instruction id, cache addr) (int, int) -> dst cache version (int)
        self.dep_version = {}                       # (instruction id, cache addr) (int, int) -> dep cache version (int)
        self.WAR_indegree = defaultdict(int)
        self.cache_versions = [0] * 1536
        self.instruction_count = 0

    def add_node(self, instruction: Instruction):
        # add instruction
        instruction.graph_id = self.instruction_count
        self.instruction_count += 1
        self.instruction_list.append(instruction)
        self.indegree.append(0)
        
        if instruction.graph_id == 22:
            pass
        
        # populate dependency graph for read dependencies
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
        engine_queue = {
            "alu": [],
            "valu": [],
            "load": [],
            "store": [],
            "flow": [],
        }
        
        job_queue = deque()
        for current_id, indegree in enumerate(self.indegree):
            if indegree == 0:
                job_queue.append(current_id)

        # print("############")
        while len(job_queue) > 0 or len([val for queue in engine_queue.values() for val in queue]) > 0:
            # print("############")
            while len(job_queue) > 0:
                current_id = job_queue.popleft()
                instruction = self.instruction_list[current_id]
                engine_queue[instruction.engine].append(instruction)
                
                dependents = []
                for dep in instruction.dep_list:
                    dependents.extend(self.WAR_graph[(dep, self.dep_version[(instruction.graph_id, dep)])])
                for adj_id in dependents:
                    self.indegree[adj_id] -= 1
                    if self.indegree[adj_id] == 0:
                        job_queue.append(adj_id)

            # add instructions to current cycle
            engine_list = ["alu", "valu", "load", "store", "flow"]
            cycle_instructions = {}
            finished_list = []
            for engine in engine_list:
                if len(engine_queue[engine]) > 0:
                    cycle_instructions[engine] = [x.instruction for x in engine_queue[engine][:SLOT_LIMITS[engine]]]
                    finished_list.extend(engine_queue[engine][:SLOT_LIMITS[engine]])
                    engine_queue[engine] = engine_queue[engine][SLOT_LIMITS[engine]:]
            compiled_instructions.append(cycle_instructions)
            
            # signal dependent instructions
            for finished in finished_list:
                # print(finished.instruction)
                dependents = []
                for dst in finished.dst:
                    dependents.extend(self.RAW_graph[(dst, self.dst_version[(finished.graph_id, dst)])])
                for adj_id in dependents:
                    self.indegree[adj_id] -= 1
                    if self.indegree[adj_id] == 0:
                        job_queue.append(adj_id)
        
        return compiled_instructions
    
    def clear(self):
        self.RAW_graph = defaultdict(list)          # (cache addr, version) (int, int) -> instruction ids (list[int])
        self.WAR_graph = defaultdict(list)          # (cache addr, version) (int, int) -> instruction ids (list[int])
        self.indegree = []                          # instruction id (int) -> indegree (int)
        self.instruction_list = []                  # instruction id (int) -> instruction
        self.dst_version = {}                       # (instruction id, cache addr) (int, int) -> dst cache version (int)
        self.dep_version = {}                       # (instruction id, cache addr) (int, int) -> dep cache version (int)
        self.WAR_indegree = defaultdict(int)
        self.cache_versions = [0] * 1536
        self.instruction_count = 0
