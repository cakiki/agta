from dataclasses import dataclass, field


@dataclass
class SemanticMemory:
    attitudes: dict[str, int] = field(default_factory=dict)
    beliefs: list[str] = field(default_factory=list)

    def add_attitude(self, key: str, value: int):
        self.attitudes[key] = value

    def add_belief(self, belief: str, threshold: float = 0.85):
        if not self.beliefs:
            self.beliefs.append(belief)
            return
        from agta.retrieval.dense import embed
        new_emb = embed([belief])
        existing_embs = embed(self.beliefs)
        sims = (existing_embs @ new_emb.T).flatten()
        if sims.max() < threshold:
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