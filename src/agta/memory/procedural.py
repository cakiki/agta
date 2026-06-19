from dataclasses import dataclass, field


class ProceduralMemory:
    def __init__(self):
        self.rules: list[str] = []

    def add_rule(self, rule: str, threshold: float = 0.85):
        if not self.rules:
            self.rules.append(rule)
            return
        from agta.retrieval.dense import embed
        new_emb = embed([rule])
        existing_embs = embed(self.rules)
        sims = (existing_embs @ new_emb.T).flatten()
        if sims.max() < threshold:
            self.rules.append(rule)

    def to_prompt_string(self) -> str:
        if not self.rules:
            return ""
        return "\n".join(f"  {r}" for r in self.rules)