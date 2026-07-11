import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader
from litellm import completion
from tqdm import tqdm

TEMPLATE_DIR = Path(__file__).parent.parent / "prompt" / "templates"

ZUF_COLS = ["P_ZUF_RAD", "P_ZUF_AUTO", "P_ZUF_OPNV", "P_ZUF_FUSS"]
EINVM_COLS = ["P_EINVM_RAD", "P_EINVM_AUTO", "P_EINVM_OPNV", "P_EINVM_FUSS"]
ZUF_LABELS = {"P_ZUF_RAD": "cycling satisfaction", "P_ZUF_AUTO": "car satisfaction", "P_ZUF_OPNV": "public transport satisfaction", "P_ZUF_FUSS": "walking satisfaction"}
EINVM_LABELS = {"P_EINVM_RAD": "cycling enjoyment", "P_EINVM_AUTO": "car enjoyment", "P_EINVM_OPNV": "public transport enjoyment", "P_EINVM_FUSS": "walking enjoyment"}


def load_mid(config):
    data_cfg = config["data"]
    p = pd.read_csv(data_cfg["persons"], sep=";", decimal=",")
    h = pd.read_csv(data_cfg["households"], sep=";", decimal=",", usecols=["H_ID", "hheink_imp"])
    w = pd.read_csv(data_cfg["trips"], sep=";", decimal=",", usecols=["HP_ID"])

    p = p.merge(h, on="H_ID", how="left")
    trip_ids = set(w["HP_ID"].unique())
    valid_zuf = (p[ZUF_COLS] >= 1) & (p[ZUF_COLS] <= 6)
    valid_einvm = (p[EINVM_COLS] >= 1) & (p[EINVM_COLS] <= 4)
    mask = (valid_zuf.any(axis=1) | valid_einvm.any(axis=1)) & p["HP_ID"].isin(trip_ids)
    return p[mask].reset_index(drop=True)


def load_inline(config):
    return pd.DataFrame(config["data"]["inline"])


def build_city(row, config):
    bland = int(row["BLAND"])
    if bland in config.get("city_states", {}):
        return config["city_states"][bland]
    bland_name = config["bland"].get(bland, f"Bundesland {bland}")
    gemtyp = config["gemtyp"].get(int(row["GEMTYP"]), "")
    return f"a {gemtyp} in {bland_name}" if gemtyp else bland_name


def build_demographics(row, config):
    demographics = {}
    demographics["age"] = int(row["HP_ALTER"])
    demographics["gender"] = config["sex"].get(int(row["HP_SEX"]), "unknown")
    bil = int(row["P_BIL"])
    if bil in config["education"]:
        demographics["education"] = config["education"][bil]
    beruf = int(row["HP_BERUF"])
    if beruf in config["employed"]:
        demographics["employed"] = config["employed"][beruf]
    hhtyp = int(row.get("hhtyp", 0))
    if hhtyp in config["hhtyp"]:
        demographics["household type"] = config["hhtyp"][hhtyp]
    eink = row.get("hheink_imp")
    if pd.notna(eink) and eink > 0:
        if eink >= 9000:
            demographics["household income"] = "9000+ EUR/month"
        else:
            demographics["household income"] = f"{int(eink)} EUR/month"
    return demographics


def build_attitudes(row):
    attitudes = {}
    for col, label in ZUF_LABELS.items():
        val = row.get(col)
        if pd.notna(val) and 1 <= val <= 6:
            attitudes[label] = int(val)
    for col, label in EINVM_LABELS.items():
        val = row.get(col)
        if pd.notna(val) and 1 <= val <= 4:
            attitudes[label] = int(val)
    return attitudes or None


def build_prompt(row, config, template, city):
    demographics = build_demographics(row, config)
    attitudes = build_attitudes(row)
    return template.render(
        city=city,
        voice=config.get("voice", "first"),
        name=row.get("name"),
        demographics=demographics,
        attitudes=attitudes or None,
        language=config.get("language", "English"),
    )


def generate_persona(prompt, config):
    llm_cfg = config["llm"]
    extra = {}
    if llm_cfg.get("disable_fallbacks"):
        extra["extra_body"] = {"disable_fallbacks": True}
    response = completion(
        model=llm_cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=llm_cfg.get("temperature", 0.1),
        api_base=llm_cfg.get("base_url"),
        **extra,
    )
    return response.choices[0].message.content.strip(), response.model


def main(config_path="configs/persona_generation.yaml", dry_run=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for key in ["city_states", "bland", "gemtyp", "sex", "education", "employed", "hhtyp"]:
        if key in config:
            config[key] = {int(k): v for k, v in config[key].items()}

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("persona_generation.jinja")

    if "inline" in config.get("data", {}):
        df = load_inline(config)
    else:
        df = load_mid(config)

    limit = config.get("limit")
    if limit:
        df = df.head(limit)

    if dry_run:
        for _, row in df.iterrows():
            city = build_city(row, config) if "BLAND" in row else config.get("city", "Germany")
            prompt = build_prompt(row, config, template, city)
            print(f"=== {int(row['HP_ID'])} ===")
            print(prompt)
            print()
        return

    model_name = config["llm"]["model"].split("/")[-1]
    voice = config.get("voice", "first")
    lang = config.get("language", "en")[:2].lower()
    temp = config["llm"].get("temperature", 0.1)
    n = limit or len(df)
    out_path = Path(config["output"]["dir"]) / f"{model_name}_{voice}_{lang}_{temp}_n{n}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path) as f:
            personas = json.load(f)
        print(f"Resuming: {len(personas)} already generated")
    else:
        personas = {}

    checkpoint_every = config.get("checkpoint_every", 100)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating personas")):
        pid = str(int(row["HP_ID"]))
        if pid in personas:
            continue
        city = build_city(row, config) if "BLAND" in row else config.get("city", "Germany")
        prompt = build_prompt(row, config, template, city)
        demographics = build_demographics(row, config)
        attitudes = build_attitudes(row)
        text, model = generate_persona(prompt, config)
        personas[pid] = {
            "persona": text,
            "demographics": demographics,
            "attitudes": attitudes,
            "city": city,
            "model": model,
        }
        if (i + 1) % checkpoint_every == 0:
            with open(out_path, "w") as f:
                json.dump(personas, f, indent=2, ensure_ascii=False)

    with open(out_path, "w") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    print(f"{len(personas)} personas written to {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    config_path = args[0] if args else "configs/persona_generation.yaml"
    main(config_path, dry_run=dry_run)