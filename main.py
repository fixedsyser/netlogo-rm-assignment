########################################################################################################################
# INSTRUCTIES
# Run één keer om de folders aan te maken. Daarna moet je zorgen dat je netlogo outputs in de "Netlogo outputs"
# terecht komen en dan kun je het programma weer draaien. Grafieken komen in "graphs" folder terecht.
#
# Afkortingen in de grafiektitel zijn:
# - MBF = Max belief factor
# - CF  = Credulity factor
# - SR  = Slander ratio
# - RS  = Reputation spread
# - DI  = Deception intensity
# - WRH = Win ratio honest/deceptive/(draw - indien Nash equilibrium na max stappen)
########################################################################################################################


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
        "number-of-trees": "NoT",
        "deception-intensity": "DI"
    }

    start_idx = df.columns.get_loc("[run number]") + 1
    end_idx = df.columns.get_loc("[step]")
    meta = df.columns[start_idx:end_idx]
    values = df.iloc[0, start_idx:end_idx]

    output = []
    for k, v in zip(meta, values):
        k_clean = k.strip("[]")
        if any(skip in k_clean for skip in ["print-enabled", "initial-number-honest-agents", "initial-number-deceptive-agents", "number-of-trees"]):
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
    # === Winstverdeling berekenen ===
    eindstap_per_run = df.groupby("[run number]").last()
    honest = eindstap_per_run["count honest-agents"]
    deceptive = eindstap_per_run["count deceptive-agents"]

    honest_wins = (honest > 0) & (deceptive == 0)
    deceptive_wins = (deceptive > 0) & (honest == 0)
    draws = ~(honest_wins | deceptive_wins)

    h_count = honest_wins.sum()
    d_count = deceptive_wins.sum()
    draw_count = draws.sum()
    total = len(eindstap_per_run)

    # Statistiek-string opbouwen
    if draw_count > 0:
        wrh_str = f"WRH: {round(h_count/total*100)}%/{round(d_count/total*100)}%/{round(draw_count/total*100)}% ({h_count}/{d_count}/{draw_count})"
    else:
        wrh_str = f"WRH: {round(h_count/total*100)}%/{round(d_count/total*100)}% ({h_count}/{d_count})"

    # Subtitel aanpassen
    subtitle = f"{metadata_str} | {wrh_str}"

    # === Grafiek tekenen ===
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
    plt.title(f"{filename}\n{subtitle}", fontsize=10)
    sns.lineplot(x=steps, y=honest_mean, label="Honest Agents", color="blue")
    plt.fill_between(steps, honest_mean - honest_std, honest_mean + honest_std, color="blue", alpha=0.2)
    sns.lineplot(x=steps, y=deceptive_mean, label="Deceptive Agents", color="red")
    plt.fill_between(steps, deceptive_mean - deceptive_std, deceptive_mean + deceptive_std, color="red", alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Agent Count")
    plt.legend()

    output_path = os.path.join(GRAPH_DIR, f"{filename}.png")
    plt.savefig(output_path)
    plt.close()


# === MAIN RUN ===
if __name__ == "__main__":
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
            timestamp = file_creation_timestamp(full_path)

            # Bepaal kolommen voor configuratie (tussen [run number] en [step])
            start_idx = df.columns.get_loc("[run number]") + 1
            end_idx = df.columns.get_loc("[step]")
            config_cols = df.columns[start_idx:end_idx]

            # Groepeer per unieke configuratie
            grouped = df.groupby(list(config_cols))
            for config_vals, subdf in grouped:
                config_list = [str(v) for v in config_vals]
                config_str = '-'.join(config_list)
                metadata = extract_metadata(subdf)
                base_filename = os.path.splitext(file)[0]
                graph_name = f"{timestamp} - {base_filename} [{config_str}]"
                plot_graph(subdf, graph_name, metadata)
                print(f"[✓] Grafiek gegenereerd: {graph_name}.png")

            # Verplaats originele CSV
            new_name = f"{timestamp} - {file}"
            imported_path = os.path.join(IMPORTED_DIR, new_name)
            shutil.move(full_path, imported_path)
            print(f"[→] Bestand verplaatst naar 'imported'.\n")

        except Exception as e:
            print(f"[!] Fout bij verwerken van {file}: {e}\n")

    print("[✔] Alle bestanden zijn verwerkt. Script beëindigd.")