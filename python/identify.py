from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Tuple, Any, List, Optional

import python.kalman as kalman
import python.resnet as resnet

MAX_ENTITY_MEMORY = 8 # how many past state vectors and embeddings to keep

@dataclass
class Entity:
    id: int
    bbox: Tuple[float, float, float, float]

    state_vector_history: Deque[Tuple[float, ...]] = field(default_factory=lambda: deque(maxlen=MAX_ENTITY_MEMORY))
    resnet_embedding_history: Deque[Any] = field(default_factory=lambda: deque(maxlen=MAX_ENTITY_MEMORY))

    last_seen: int = 0

    def set_entity(self, id: int, bbox: Tuple[float, float, float, float, float], state_vector, resnet_embedding):
        self.id = id
        self.bbox = bbox
        self.last_seen = 0
        self.add_state_vector(state_vector)
        self.add_embedding(resnet_embedding)

    def add_state_vector(self, sv: Tuple[float, ...]) -> None:
        self.state_vector_history.append(sv)

    def add_embedding(self, emb: Any) -> None:
        self.resnet_embedding_history.append(emb)

    def get_state_vector_history(self) -> List[Tuple[float, ...]]:
        return list(self.state_vector_history)

    def get_embedding_history(self) -> List[Any]:
        return list(self.resnet_embedding_history)

@dataclass
class Memory:
    curr_entities: Optional[List[Entity]] = None

memory = Memory()

def identify(frame, curr_bboxes, next_bboxes):
    global memory

    # if no memory yet, initialize memory with current bboxes

    if memory == None or memory.curr_entities is None:
        memory.curr_entities = []
        for i, bbox in enumerate(curr_bboxes):
            e = Entity(
                id=i,
                bbox=bbox,
                state_vector=kalman.create_state_vector(bbox),
                resnet_embedding=resnet.get_resnet_embedding(frame, bbox)
            )
            memory.curr_entities.append(e)

    identified_bbox_ids = list(range(len(curr_bboxes)))
    
    return identified_bbox_ids