import json
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

class MotifType(str, Enum):
    BOUNDARY = "Boundary"     # Start/End of the program
    SEQUENCE = "Sequence"     # Linear execution
    CONDITION = "Condition"   # Branching based on boolean
    LOOP = "Loop"             # Iteration
    STATE = "State"           # Read/Write to memory
    MESSAGE = "Message"       # External function call / Math

class EdgeType(int, Enum):
    EXECUTION = 0             # Control flow (the execution pointer moves here)
    DATA = 1                  # Data flow (passing a value between nodes)

class Edge(BaseModel):
    source_node: int
    target_node: int
    edge_type: EdgeType
    input_index: int          # e.g., 0 for True path, 1 for False path; or Arg 0, Arg 1

class Node(BaseModel):
    node_id: int
    motif: MotifType
    # This is the bridge to the Legislative Branch!
    # The DiT leaves this None. The LLM fills it with an integer index.
    literal_pointer: Optional[int] = None 

class ExecutionGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    # The Constant Pool managed by the LLM (e.g., {0: "storage", 1: "empty?", 2: 42})
    literal_pool: Dict[int, Union[str, int, float, bool]]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
        
    @classmethod
    def from_json(cls, json_data: str) -> "ExecutionGraph":
        return cls.model_validate_json(json_data)

# MVP Adjacency Matrix Representation
# For the DiT, this will be represented as a multi-channel tensor [N, N, C]
# Channel 0: Presence (1 or 0)
# Channel 1: Edge Type (0 or 1)
# Channel 2: Input Index (0, 1, 2...)
