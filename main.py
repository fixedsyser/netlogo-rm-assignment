########################################################################################################################
# INSTRUCTIES
# Run één keer om de folders aan te maken. Daarna moet je zorgen dat je netlogo outputs in de "Netlogo outputs"
# terecht komen en dan kun je het programma weer draaien. Grafieken komen in "graphs" folder terecht.
#
# HEATMAP:   Alleen gegenereerd als er PRECIES twee parameters zijn die variëren.
# BARCHARTS: Alleen gegenereerd als er PRECIES één parameter is die varieert.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import StringIO
from statsmodels.stats.proportion import proportion_confint
from matplotlib import colormaps
from scipy.stats import binomtest

# === CONFIG ===
INPUT_DIR     = r".\Netlogo outputs"
IMPORTED_DIR  = r".\Netlogo outputs\imported"
GRAPH_DIR     = r".\graphs"

# === UTILITIES ===
def extract_metadata(df):
    """
    Geeft een string met alle metadata tussen [run number] en [step], met afkortingen, maar slaat kolommen over die
    variëren (d.w.z. de variabele param(s) in de run, of in de ignore-lijst staan.
    """
    replacements = {
        "max-belief-factor": "MBF",
        "credulity-factor": "CF",
        "slander-ratio": "SR",
        "reputation-spread": "RS",
        "number-of-trees": "NoT",
        "deception-intensity": "DI",
        "initial-number-honest-agents": "INIT-HA",
        "initial-number-deceptive-agents": "INIT-DA"
    }

    # Kolommen die we altijd willen negeren, ongeacht of ze variëren
    ignore_cols = [
        #"initial-number-honest-agents",
        #"initial-number-deceptive-agents",
        "print-enabled?"
    ]

    # 1) Bepaal de kolommen die metadata bevatten
    start = df.columns.get_loc("[run number]") + 1
    end   = df.columns.get_loc("[step]")
    meta_cols = list(df.columns[start:end])

    # 2) Filter de variërende kolommen
    varied = [col for col in meta_cols if df[col].nunique() > 1]

    # 3) Bouw metadata-string op uit vaste kolommen, zonder de blacklisted of variërende
    values = df.iloc[0, start:end]
    parts = []
    for col, val in zip(meta_cols, values):
        if col in varied or col in ignore_cols:
            continue
        key = col.strip("[]")
        label = replacements.get(key, key)
        parts.append(f"{label}={val}")

    return "  ".join(parts)


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

def get_cohen_h(p_baseline, p_current):
    h_value = 2 * (np.arcsin(np.sqrt(p_baseline)) - np.arcsin(np.sqrt(p_current)))
    print(f"Baseline honest {p_baseline}, actual honest: {p_current}, h: {h_value}, abs h: {abs(h_value)}")
    return abs(h_value)

def plot_graph(df, filename, metadata_str):
    # === Runs aanvullen tot max step ===
    run_col = "[run number]"
    step_col = "[step]"

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

################################################################################################################
# ============================================== MAIN RUN =====================================================#
################################################################################################################
if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(IMPORTED_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Gebruik exact dezelfde colormap als de heatmap voor de bar graphs
    cmap = colormaps["RdBu_r"]

    # Pak de kleuren voor 100% honest (blauw) en 100% deceptive (rood)
    honest_color = cmap(0.1)  # links van het spectrum = blauw
    deceptive_color = cmap(0.9)  # rechts van het spectrum = rood

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    print(f"[▶] Start verwerking van {len(files)} bestand(en) uit: {INPUT_DIR}")

    config_cols = [] # Lege lijst voor het geval er geen files zijn.

    #
    # Waarschuwing: heeeele lange for loop waarin we één file afhandelen.
    #

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

                # George: aangezien we voor nu niet meer de 'lijn' grafieken doen, heb ik die code eruit gecommentarieerd.
                # Kan altijd weer aangezet worden, maar het kostte hier te veel extra tijd om die steeds te genereren.

                # plot_graph(subdf, graph_name, metadata)
                # print(f"[✓] Grafiek gegenereerd: {graph_name}.png")

            # Bepaal hoeveel parameters we variëren we in deze run / file (om te bepalen of we barcharts of heatmap maken)
            varied_cols = [col for col in config_cols if df[col].nunique() > 1]
            nr_of_changing_params = len(varied_cols)

            # Aantal runs van de eerste config tellen om de runcount string op te bouwen
            runs_per_config = df.groupby(list(config_cols))["[run number]"].nunique().iloc[0]
            runcount_str = f"{runs_per_config} runs"

            ################################################################################################################
            # Bar graphs: alleen als er exact één parameter varieert (anders is het een multidimensionale run en is onduidelijk
            # waar je nou precies de effecten van wilt zien of in welke volgorde je de bars zet)
            ################################################################################################################

            if nr_of_changing_params == 1:
                print(f"[📊] Bar charts genereren voor parameter: {varied_cols[0]}")
                os.makedirs(os.path.join(GRAPH_DIR, "bar charts"), exist_ok=True)

                bar_data = []
                grouped = df.groupby(list(config_cols))

                # Haal de eerste en laatste bar op uit grouped
                grouped_list = list(grouped)
                first_config_vals, first_subdf = grouped_list[0]
                last_config_vals, last_subdf = grouped_list[-1]

                # Bereken % honest wins voor baseline (eerste bar)
                eind_baseline = first_subdf.groupby("[run number]").last()
                honest_baseline = eind_baseline["count honest-agents"]
                deceptive_baseline = eind_baseline["count deceptive-agents"]
                honest_wins_baseline = (honest_baseline > 0) & (deceptive_baseline == 0)
                h_baseline = honest_wins_baseline.sum() / len(eind_baseline)

                # Bereken % honest wins voor laatste bar
                eind_last = last_subdf.groupby("[run number]").last()
                honest_last = eind_last["count honest-agents"]
                deceptive_last = eind_last["count deceptive-agents"]
                honest_wins_last = (honest_last > 0) & (deceptive_last == 0)
                h_last = honest_wins_last.sum() / len(eind_last)

                # Bepaal toetsrichting; bij gelijke waarden geen voorkeur, dus two-sided
                if h_last > h_baseline:
                    alt = 'greater'
                elif h_last < h_baseline:
                    alt = 'less'
                else:
                    alt ='two-sided'


                for config_vals, subdf in grouped:
                    param_dict = dict(zip(config_cols, config_vals))
                    eind = subdf.groupby("[run number]").last()
                    honest = eind["count honest-agents"]
                    deceptive = eind["count deceptive-agents"]

                    honest_wins = (honest > 0) & (deceptive == 0)
                    deceptive_wins = (deceptive > 0) & (honest == 0)

                    h_count = honest_wins.sum()
                    d_count = deceptive_wins.sum()
                    total = len(eind)

                    h_val = h_count / total
                    h_pct = h_val * 100
                    d_pct = (d_count / total) * 100

                    h_ci_low, h_ci_upp = proportion_confint(h_count, total, method="wilson")
                    d_ci_low, d_ci_upp = proportion_confint(d_count, total, method="wilson")

                    p_val = binomtest(h_count, total, p=h_baseline, alternative=alt).pvalue
                    h_val = get_cohen_h(h_baseline, h_val)

                    bar_data.append({
                        varied_cols[0]: param_dict[varied_cols[0]],
                        "Honest Win %": h_pct,
                        "Deceptive Win %": d_pct,
                        "h_ci_low": h_ci_low * 100,
                        "h_ci_upp": h_ci_upp * 100,
                        "d_ci_low": d_ci_low * 100,
                        "d_ci_upp": d_ci_upp * 100,
                        "p_value": p_val,
                        "h_value": h_val
                    })

                df_bar = pd.DataFrame(bar_data)
                df_bar_sorted = df_bar.sort_values(by=varied_cols[0])

                ################################################################################################################
                # Stacked bar graphs
                ################################################################################################################
                bar_width = 0.5  # 50% breedte = witruimte tussen bars

                x = range(len(df_bar_sorted))
                labels = df_bar_sorted[varied_cols[0]]

                plt.figure(figsize=(10, 6))
                plt.ylim(0, 107)
                plt.bar(x, df_bar_sorted["Honest Win %"], width=bar_width, label="Honest", color=honest_color)
                plt.bar(x, df_bar_sorted["Deceptive Win %"],
                        bottom=df_bar_sorted["Honest Win %"], width=bar_width, label="Deceptive", color=deceptive_color)

                for i, row in df_bar_sorted.iterrows():
                    # Eerste bar is baseline overslaan
                    if i >= 1:
                        p = row["p_value"]
                        if p < 0.001:
                            label_p = "p < 0.001"
                        else:
                            label_p = f"p = {p:.3f}"

                        h =  row["h_value"]
                        label_h = f"h = {h:.2f}"

                        plt.text(i, 104, label_p, ha='center', fontsize=8)
                        plt.text(i, 101, label_h, ha='center', fontsize=8)

                plt.xticks(ticks=x, labels=labels)
                plt.xlabel(varied_cols[0])
                plt.ylabel("Win Percentage")

                metadata_str = extract_metadata(df)
                plt.title(f"Stacked winrates for varying {varied_cols[0]}\n{metadata_str}")
                plt.legend()

                stacked_file = os.path.join(GRAPH_DIR, "bar charts",
                                            f"{timestamp} - {runcount_str} - barchart_stacked_{varied_cols[0]}.png")
                plt.savefig(stacked_file)
                plt.close()
                print(f"[✔] Stacked bar chart opgeslagen als: {stacked_file}")

                ################################################################################################################
                # Side-by-side bar graphs met confidence intervals per bar
                ################################################################################################################
                plt.figure(figsize=(10, 6))
                x = range(len(df_bar_sorted))
                bar_width = 0.3

                plt.bar([i - bar_width/2 for i in x], df_bar_sorted["Honest Win %"],
                        yerr=[df_bar_sorted["Honest Win %"] - df_bar_sorted["h_ci_low"],
                              df_bar_sorted["h_ci_upp"] - df_bar_sorted["Honest Win %"]],
                        width=bar_width, capsize=5, label="Honest", color=honest_color)

                plt.bar([i + bar_width/2 for i in x], df_bar_sorted["Deceptive Win %"],
                        yerr=[df_bar_sorted["Deceptive Win %"] - df_bar_sorted["d_ci_low"],
                              df_bar_sorted["d_ci_upp"] - df_bar_sorted["Deceptive Win %"]],
                        width=bar_width, capsize=5, label="Deceptive", color=deceptive_color)

                plt.xticks(ticks=x, labels=labels)
                plt.xlabel(varied_cols[0])
                plt.ylabel("Win Percentage")
                plt.title(f"Winrates with Wilson CI for varying {varied_cols[0]}\n{metadata_str}")
                plt.legend()

                grouped_file = os.path.join(GRAPH_DIR, "bar charts",
                                            f"{timestamp} - {runcount_str} - barchart_grouped_{varied_cols[0]}.png")
                plt.savefig(grouped_file)
                plt.close()
                print(f"[✔] Grouped bar chart opgeslagen als: {grouped_file}")


            ################################################################################################################
            # Heatmap: alleen als er exact twee parameters variëren (anders geen 2D heatmap mogelijk)
            ################################################################################################################

            if nr_of_changing_params == 2:
                print(f"[🎨] Heatmap genereren op basis van parameters: {varied_cols[0]} en {varied_cols[1]}")
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
                metadata_str = extract_metadata(df)
                plt.title(f"Winrates honest agents\n({varied_cols[0]} vs {varied_cols[1]})\n{metadata_str}")
                plt.xlabel(varied_cols[0])
                plt.ylabel(varied_cols[1])

                # Y-as inverteren zodat waarden van klein naar groot lopen
                plt.gca().invert_yaxis()

                # Zorg dat de subfolder bestaat en sla het bestand op
                os.makedirs(os.path.join(GRAPH_DIR, "heatmaps"), exist_ok=True)
                heatmap_file = os.path.join(GRAPH_DIR, "heatmaps", f"{timestamp} - {runcount_str} - heatmap_{varied_cols[0]}_vs_{varied_cols[1]}.png")
                plt.savefig(heatmap_file)
                plt.close()
                print(f"[✔] Heatmap opgeslagen als: {heatmap_file}\n")

            # Verplaats originele CSV
            new_name = f"{timestamp} - {file}"
            imported_path = os.path.join(IMPORTED_DIR, new_name)
            shutil.move(full_path, imported_path)
            print(f"[→] Bestand verplaatst naar 'imported'.\n")

        except Exception as e:
            print(f"[!] Fout bij verwerken van {file}: {e}\n")

    # Sluit af als alle bestanden zijn verwerkt
    print("[✔] Alle bestanden zijn verwerkt. Script beëindigd.")