from mesa_llm.memory.memory import Memory
from agta.memory.working import WorkingMemory


class MemoryManager(Memory):
    def __init__(self, agent, display=True):
        super().__init__(agent=agent, display=display)
        self.working = WorkingMemory()

    def get_prompt_ready(self) -> str:
        lines = []
        lines.append(f"Car: {self.working.car_location}")
        lines.append(f"Bicycle: {self.working.bicycle_location}")
        lines.append(f"You are at: {self.working.current_location}")
        if self.working.trips_today:
            lines.append("Today's trips:")
            for t in self.working.trips_today:
                lines.append(f"  {t.time} {t.from_activity} -> {t.to_activity}: {t.mode}")
        return "\n".join(lines)

    def get_communication_history(self) -> str:
        return ""

    def process_step(self, pre_step=False):
        pass