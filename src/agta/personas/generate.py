import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader
from litellm import completion
from tqdm import tqdm

TEMPLATE_DIR = Path(__file__).parent.parent / "prompt" / "templates"


def load_data(config):
    data_cfg = config["data"]
    if "inline" in data_cfg:
        return pd.DataFrame(data_cfg["inline"])
    df = pd.read_parquet(data_cfg["path"])
    if "filter" in data_cfg:
        col, val = data_cfg["filter"]["column"], data_cfg["filter"]["value"]
        df = df[df[col] == val]
    return df


def build_prompt(row, config, template):
    demo_map = config["demographics"]
    demographics = {label: row[col] for col, label in demo_map.items() if col in row and pd.notna(row[col])}
    att_map = config.get("attitudes", {})
    attitudes = {label: int(row[col]) for col, label in att_map.items() if col in row and pd.notna(row.get(col))}
    return template.render(
        city=config["city"],
        voice=config.get("voice", "first"),
        name=row.get("name"),
        demographics=demographics,
        attitudes=attitudes or None,
    )


def generate_persona(prompt, config):
    llm_cfg = config["llm"]
    response = completion(
        model=llm_cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=llm_cfg.get("temperature", 0.1),
        api_base=llm_cfg.get("base_url"),
    )
    print(response.model)
    return response.choices[0].message.content.strip()


def main(config_path="configs/persona_generation.yaml", dry_run=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("persona_generation.jinja")
    df = load_data(config)
    id_col = config["data"].get("id_column", "person_id")

    if dry_run:
        for _, row in df.iterrows():
            prompt = build_prompt(row, config, template)
            print(f"=== {row[id_col]} ===")
            print(prompt)
            print()
        return

    personas = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating personas"):
        prompt = build_prompt(row, config, template)
        personas[str(row[id_col])] = generate_persona(prompt, config)

    out_path = Path(config["output"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    print(f"{len(personas)} personas written to {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    config_path = args[0] if args else "configs/persona_generation.yaml"
    main(config_path, dry_run=dry_run)