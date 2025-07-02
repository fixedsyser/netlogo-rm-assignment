import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def analyze_agent_simulation(df):
    # Load and clean data
    df.columns = [col.strip('[]"')
                  .replace('-', '_')
                  .replace('?', '')
                  .replace('average', 'avg')
                  .replace('reputation', 'rep')
                  .replace('_agents', '')
                  for col in df.columns]
    
    df = df.drop(columns=['print_enabled'])
    
    # Get the last row (final timestep) for each run
    eind = df.groupby("run number").last()
    end_step = eind["step"]
    # Replace run numbers with their corresponding last step numbers
    df["run number"] = df["run number"].map(end_step)

    # Rename the column
    df = df.rename(columns={"run number": "run_length"})
    
    # Convert to numeric (skip first column)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get numeric data
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    varying_cols = []
    
    for col in numeric_df.columns:
        if numeric_df[col].nunique() > 1:  # More than 1 unique value
            varying_cols.append(col)
    # Use only varying columns
    numeric_df = numeric_df[varying_cols]
    
    print(f"Analyzing {numeric_df.shape[0]} rows, {numeric_df.shape[1]} variables")
    
    # 1. Correlation heatmap
    
    corr_matrix = numeric_df.corr(min_periods=2, numeric_only=True)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()

    print("✅ YES - positive correlation (>0.3) suggests higher col1 → more col2")
    print("❌ NO - negative correlation (<0.3) suggests higher col1 → fewer col2")
    print("🤷 UNCLEAR - Otherwise: weak correlation, no clear relationship")
    
    # 2. Find strongest correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.6:  # Strong correlation
                strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if strong_corrs:
        print(f"\n📊 STRONGEST CORRELATIONS:")
        for var1, var2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)[:5]:
            print(f"   {var1} ↔ {var2}: {corr:.3f}")
    
    return plt