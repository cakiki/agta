from mesa_llm.memory.memory import Memory
from agta.memory.working import WorkingMemory
from agta.memory.semantic import SemanticMemory
from agta.memory.episodic import EpisodicMemory

class MemoryManager(Memory):
    def __init__(self, agent, display=True):
        super().__init__(agent=agent, display=display)
        self.working = WorkingMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()

    def get_prompt_ready(self) -> str:
        lines = []
        lines.append(f"Car: {self.working.car_location}")
        lines.append(f"Bicycle: {self.working.bicycle_location}")
        lines.append(f"You are at: {self.working.current_location}")
        if self.working.trips_today:
            lines.append("Today's trips:")
            for t in self.working.trips_today:
                lines.append(f"  {t.time} {t.from_activity} -> {t.to_activity}: {t.mode}")
        semantic_str = self.semantic.to_prompt_string()
        if semantic_str:
            lines.append("Your transport attitudes and beliefs:")
            lines.append(semantic_str)
        return "\n".join(lines)

    def get_episodic_context(self, to_activity: str, k: int = 3) -> str:
        current_day = self.agent.model.current_day
        return self.episodic.to_prompt_string(to_activity, current_day, k)

    def get_communication_history(self) -> str:
        return ""

    def process_step(self, pre_step=False):
        pass