import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import StringIO

# === CONFIG ===
INPUT_DIR     = r".\Netlogo outputs"
IMPORTED_DIR  = r".\Netlogo outputs\imported"
GRAPH_DIR     = r".\graphs"

# === UTILITIES ===
def extract_metadata(df):
    replacements = {
        "max-belief-factor": "MBF",
        "credulity-factor": "CF",
        "slander-ratio": "SR",
        "reputation-spread": "RS",
        "number-of-trees": "NoT"
    }

    start_idx = df.columns.get_loc("[run number]") + 1
    end_idx = df.columns.get_loc("[step]")
    meta = df.columns[start_idx:end_idx]
    values = df.iloc[0, start_idx:end_idx]

    output = []
    for k, v in zip(meta, values):
        k_clean = k.strip("[]")
        if any(skip in k_clean for skip in ["print-enabled", "initial-number-honest-agents", "initial-number-deceptive-agents"]):
            continue
        label = replacements.get(k_clean, k_clean)
        output.append(f"{label}: {v}")

    return ', '.join(output)

def load_netlogo_csv(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_index = None
    for i, line in enumerate(lines):
        if "[step]" in line and "[run number]" in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError("Geen geldige kolomheader gevonden in bestand.")

    data_str = ''.join(lines[header_index:])
    df = pd.read_csv(StringIO(data_str), sep=",", quotechar='"')
    return df

def file_creation_timestamp(path):
    created = os.path.getctime(path)
    return datetime.fromtimestamp(created).strftime("%Y.%m.%d-%H.%M.%S")

def plot_graph(df, filename, metadata_str):
    df_grouped = df.groupby("[step]").agg({
        "count honest-agents": ['mean', 'std'],
        "count deceptive-agents": ['mean', 'std']
    })

    steps = df_grouped.index
    honest_mean = df_grouped[("count honest-agents", "mean")]
    honest_std = df_grouped[("count honest-agents", "std")]
    deceptive_mean = df_grouped[("count deceptive-agents", "mean")]
    deceptive_std = df_grouped[("count deceptive-agents", "std")]

    plt.figure(figsize=(10, 6))
    plt.title(f"{filename}\n{metadata_str}", fontsize=12)
    sns.lineplot(x=steps, y=honest_mean, label="Honest Agents", color="blue")
    plt.fill_between(steps, honest_mean - honest_std, honest_mean + honest_std, color="blue", alpha=0.2)
    sns.lineplot(x=steps, y=deceptive_mean, label="Deceptive Agents", color="red")
    plt.fill_between(steps, deceptive_mean - deceptive_std, deceptive_mean + deceptive_std, color="red", alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Agent Count")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(GRAPH_DIR, f"{filename}.png")
    plt.savefig(output_path)
    plt.close()

# === MAIN RUN ===
if __name__ == "__main__":
    # Folders maken als ze nog niet bestaan
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(IMPORTED_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    print(f"[▶] Start verwerking van {len(files)} bestand(en) uit: {INPUT_DIR}")

    for file in files:
        try:
            full_path = os.path.join(INPUT_DIR, file)
            print(f"[•] Verwerken: {file}")

            df = load_netlogo_csv(full_path)
            metadata = extract_metadata(df)
            timestamp = file_creation_timestamp(full_path)
            new_name = f"{timestamp} - {file}"
            plot_graph(df, os.path.splitext(new_name)[0], metadata)

            imported_path = os.path.join(IMPORTED_DIR, new_name)
            shutil.move(full_path, imported_path)
            print(f"[✓] Klaar. Grafiek opgeslagen, bestand verplaatst naar 'imported'.\n")

        except Exception as e:
            print(f"[!] Fout bij verwerken van {file}: {e}\n")

    print("[✔] Alle bestanden zijn verwerkt. Script beëindigd.")
