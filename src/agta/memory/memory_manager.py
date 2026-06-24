from mesa_llm.memory.memory import Memory
from agta.memory.procedural import ProceduralMemory
from agta.memory.working import WorkingMemory
from agta.memory.semantic import SemanticMemory
from agta.memory.episodic import EpisodicMemory
from agta.retrieval.hybrid import HybridRetriever

class MemoryManager(Memory):
    def __init__(self, agent, display=True):
        super().__init__(agent=agent, display=display)
        self.working = WorkingMemory()
        self.semantic = SemanticMemory()
        rc = getattr(agent.model, 'retrieval_config', {})
        retriever = HybridRetriever(
            dense_weight=rc.get('dense_weight', 0.4),
            sparse_weight=rc.get('sparse_weight', 0.3),
            recency_weight=rc.get('recency_weight', 0.3),
        )
        self.episodic = EpisodicMemory(retriever=retriever)
        self.procedural = ProceduralMemory()

    def get_prompt_ready(self) -> str:
        lines = []
        lines.append(f"Car: {self.working.car_location}")
        lines.append(f"Bicycle: {self.working.bicycle_location}")
        lines.append(f"You are at: {self.working.current_location}")
        if self.working.trips_today:
            lines.append("Today's trips:")
            for t in self.working.trips_today:
                lines.append(f"  {t.time} {t.from_activity} -> {t.to_activity}: {t.mode}")
        if self.working.bicycle_location != self.working.current_location:
            lines.append(f"Note: Your bicycle is at {self.working.bicycle_location}, not with you.")
        if self.working.car_location != self.working.current_location:
            lines.append(f"Note: Your car is at {self.working.car_location}, not with you.")
        semantic_str = self.semantic.to_prompt_string(max_beliefs=5)
        if semantic_str:
            lines.append("Your transport attitudes and beliefs:")
            lines.append(semantic_str)
        procedural_str = self.procedural.to_prompt_string()
        if procedural_str:
            lines.append("Your decision rules:")
            lines.append(procedural_str)
        return "\n".join(lines)

    def get_episodic_context(self, to_activity: str, k: int = 3) -> str:
        current_day = self.agent.model.current_day
        return self.episodic.to_prompt_string(to_activity, current_day, k)

    def get_communication_history(self) -> str:
        return ""

    def process_step(self, pre_step=False):
        pass