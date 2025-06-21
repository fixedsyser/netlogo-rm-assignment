########################################################################################################################
# INSTRUCTIES
# Run één keer om de folders aan te maken. Daarna moet je zorgen dat je netlogo outputs in de "Netlogo outputs"
# terecht komen en dan kun je het programma weer draaien. Grafieken komen in "graphs" folder terecht.
#
# HEATMAP: Alleen gegenereerd als er PRECIES twee parameters zijn die variëren.
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
    # === Runs aanvullen tot max step ===
    run_col = "[run number]"
    step_col = "[step]"
    agent_cols = ["count honest-agents", "count deceptive-agents"]

    max_step = df[step_col].max()

    padded_rows = []
    for run, run_df in df.groupby(run_col):
        run_max = run_df[step_col].max()
        if run_max < max_step:
            missing_steps = list(range(run_max + 1, max_step + 1))
            last_row = run_df.loc[run_df[step_col] == run_max].iloc[0]
            for step in missing_steps:
                pad_row = last_row.copy()
                pad_row[step_col] = step
                padded_rows.append(pad_row)

    if padded_rows:
        df = pd.concat([df, pd.DataFrame(padded_rows)], ignore_index=True)

    # === Winstverdeling berekenen ===
    eindstap_per_run = df.groupby(run_col).last()
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

    config_cols = [] # Lege lijst voor het geval er geen files zijn en we alsnog de heatmap proberen te maken.

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

    # === Extra HEATMAP als er precies twee varierende parameters zijn ===

    # Stap 1: Bepaal welke configuratieparameters meer dan 1 unieke waarde hebben
    varied_cols = [col for col in config_cols if df[col].nunique() > 1]

    # Alleen doorgaan als er exact twee parameters variëren (anders geen 2D heatmap mogelijk)
    if len(varied_cols) == 2:
        print(f"[🎨] Heatmap gegenereerd op basis van parameters: {varied_cols[0]} en {varied_cols[1]}")
        heatmap_data = []

        # Stap 2: groepeer de data per unieke configuratieset
        grouped = df.groupby(list(config_cols))
        for config_vals, subdf in grouped:
            param_dict = dict(zip(config_cols, config_vals))  # zet configuratie in dictionaryvorm

            # Stap 3: pak per run de laatste rij (laatste timestep) van de simulatie
            eind = subdf.groupby("[run number]").last()
            honest = eind["count honest-agents"]
            deceptive = eind["count deceptive-agents"]

            # Stap 4: bepaal welke runs door honest agents gewonnen zijn
            honest_wins = (honest > 0) & (deceptive == 0)
            # Bepaal het percentage van honest wins
            h_ratio = honest_wins.sum() / len(honest_wins)

            # Voeg data toe aan de lijst voor de heatmap
            heatmap_data.append({
                varied_cols[0]: param_dict[varied_cols[0]],
                varied_cols[1]: param_dict[varied_cols[1]],
                "Honest Win %": round(h_ratio * 100)
            })

        # Stap 5: zet lijst om naar dataframe en vorm pivot-tabel
        df_heat = pd.DataFrame(heatmap_data)
        heatmap_pivot = df_heat.pivot(index=varied_cols[1], columns=varied_cols[0], values="Honest Win %")

        # Stap 6: genereer en save de heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_pivot,
            annot=True,  # Toon waardes in de cellen
            cmap="RdBu",  # Kleuren: rood = 100% honest, blauw = 0%
            center=50,  # Wit bij 50% (neutral)
            fmt=".0f",  # Afronden op hele getallen
            cbar_kws={"label": "Honest Win %"}  # Label voor de colorbar
        )
        plt.title(f"Winrates honest agents\n({varied_cols[0]} vs {varied_cols[1]})")
        plt.xlabel(varied_cols[0])
        plt.ylabel(varied_cols[1])

        # Zorg dat de subfolder bestaat en sla het bestand op
        os.makedirs(os.path.join(GRAPH_DIR, "heatmaps"), exist_ok=True)
        heatmap_file = os.path.join(GRAPH_DIR, "heatmaps", f"heatmap_{varied_cols[0]}_vs_{varied_cols[1]}.png")
        plt.savefig(heatmap_file)
        plt.close()
        print(f"[✔] Heatmap opgeslagen als: {heatmap_file}\n")

    # Sluit af als alle bestanden zijn verwerkt
    print("[✔] Alle bestanden zijn verwerkt. Script beëindigd.")