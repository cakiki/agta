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
    
    def consolidate_beliefs(self):
        if len(self.memory.semantic.beliefs) < 5:
            return
        prompt = f"Here are transport beliefs about a person:\n"
        for b in self.memory.semantic.beliefs:
            prompt += f"  - {b}\n"
        prompt += "\nMerge duplicates and remove redundancies. Return a consolidated list.\n"
        prompt += 'Return JSON: {"beliefs": ["belief 1", "belief 2", ...]}'
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)
        self.memory.semantic.beliefs = parsed.get("beliefs", self.memory.semantic.beliefs)

    def to_prompt_string(self) -> str:
        lines = []
        if self.attitudes:
            for key, value in self.attitudes.items():
                lines.append(f"  {key}: {value}")
        if self.beliefs:
            for belief in self.beliefs:
                lines.append(f"  {belief}")
        return "\n".join(lines)