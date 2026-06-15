from dataclasses import dataclass, field


@dataclass
class SemanticMemory:
    attitudes: dict[str, int] = field(default_factory=dict)
    beliefs: list[str] = field(default_factory=list)

    def add_attitude(self, key: str, value: int):
        self.attitudes[key] = value

    def add_belief(self, belief: str):
        if belief not in self.beliefs:
            self.beliefs.append(belief)

    def to_prompt_string(self) -> str:
        lines = []
        if self.attitudes:
            for key, value in self.attitudes.items():
                lines.append(f"  {key}: {value}")
        if self.beliefs:
            for belief in self.beliefs:
                lines.append(f"  {belief}")
        return "\n".join(lines)