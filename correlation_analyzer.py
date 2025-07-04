import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(df):
    df.columns = [col.strip('[]"')
                  .replace('-', '_')
                  .replace('?', '')
                  .replace('average', 'avg')
                  .replace('reputation', 'rep')
                  .replace('_agents', '')
                  for col in df.columns]
    
    df = df.drop(columns=['print_enabled'])
    
    # Get the last row (final timestep) for each run
    eind = df.groupby("run number").max()
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
    
    return numeric_df

def create_correlation_matrix(numeric_df, save_path):
    """Create and optionally save correlation matrix with alphabetical labels"""
    print(f"Analyzing {numeric_df.shape[0]} rows, {numeric_df.shape[1]} variables")
    
    # Sort columns alphabetically
    sorted_cols = sorted(numeric_df.columns)
    numeric_df_sorted = numeric_df[sorted_cols]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df_sorted.corr(min_periods=2, numeric_only=True)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "correlation_matrix.png"))
    plt.close()
    print(f"[✔] Correlation Matrix opgeslagen als: {save_path}\n")    
    return corr_matrix

def create_scatterplots(numeric_df, save_path, corr_matrix=None, mode='strong', correlation_threshold=0.05):
    """
    Create scatterplots for variable pairs
    
    Args:
        numeric_df: DataFrame with numeric data
        corr_matrix: Pre-computed correlation matrix (optional)
        mode: 'all' for all pairs, 'strong' for pairs above threshold
        correlation_threshold: Minimum absolute correlation to include (default 0.05)
        save_path: Path to save the plot
    """
    if corr_matrix is None:
        # Sort columns alphabetically
        sorted_cols = sorted(numeric_df.columns)
        numeric_df_sorted = numeric_df[sorted_cols]
        corr_matrix = numeric_df_sorted.corr(min_periods=2, numeric_only=True)
    else:
        # Use sorted dataframe
        sorted_cols = sorted(numeric_df.columns)
        numeric_df_sorted = numeric_df[sorted_cols]
    
    # Find pairs to plot
    pairs_to_plot = []

    # Plot all variable pairs
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            match mode:
                case 'all': # Plot all variable pairs
                    pairs_to_plot.append((var1, var2, corr_val))
                case 'strong': # Plot only pairs above correlation threshold
                    if abs(corr_val) > correlation_threshold:
                        pairs_to_plot.append((var1, var2, corr_val))
                case 'run_length': # Only include pairs where var1 is run_length
                    if var1 == 'run_length' or var2 == 'run_length':
                        pairs_to_plot.append((var1, var2, corr_val))
                case _:
                    print(f"Invalid scatterplot mode: {mode}")
                    return None

    if not pairs_to_plot:
        print(f"No variable pairs found with correlation > {correlation_threshold}")
        return None
    
    print(f"\n🔍 Creating scatterplots for {len(pairs_to_plot)} variable pairs...")
    
    num_plots = len(pairs_to_plot)
    cols = min(4, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    # Handle single plot case
    if num_plots == 1:
        axes = [axes]
    
    # Create scatterplots
    for idx, (var1, var2, corr) in enumerate(pairs_to_plot):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.scatter(numeric_df_sorted[var1], numeric_df_sorted[var2], alpha=0.6, s=30)
        
        # Add trend line
        try:
            z = np.polyfit(numeric_df_sorted[var1], numeric_df_sorted[var2], 1)
            p = np.poly1d(z)
            ax.plot(numeric_df_sorted[var1], p(numeric_df_sorted[var1]), "r--", alpha=0.8)
        except:
            pass  # Skip trend line if fitting fails
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(f'{var1} vs {var2}\nCorr: {corr:.3f}')
        ax.grid(True, alpha=0.3)
        
        if (idx + 1) % 10 == 0:
            print(f"Progress: {(idx + 1)}/{num_plots} plotted")
    
    # Hide empty subplots
    for idx in range(num_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
        
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{mode}_scatterplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✔] Scatterplots opgeslagen als: {save_path}")
    return fig

def analyze_agent_simulation(df, save_path):
    """Main analysis function - now focused on correlation matrix only"""
    numeric_df = prepare_data(df)
    
    corr_matrix = create_correlation_matrix(numeric_df, save_path)
    
    print("✅ YES - positive correlation (>0.3) suggests higher col1 → more col2")
    print("❌ NO - negative correlation (<0.3) suggests higher col1 → fewer col2")
    print("🤷 UNCLEAR - Otherwise: weak correlation, no clear relationship")
    
    # Find strongest correlations
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
    

    return numeric_df, corr_matrix

def plot_run_length_vs_trees(df, save_path):
    # Check if number_of_trees exists and varies
    if 'number_of_trees' not in df.columns:
        print("[ℹ] number_of_trees not found or not varying, skipping run length analysis")
        return
    
    print(f"[📈] Generating run length vs trees analysis")
    
    # Group by number_of_trees and calculate average run_length
    avg_data = df.groupby('number_of_trees')['run_length'].agg(['mean']).reset_index()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(avg_data['number_of_trees'], avg_data['mean'], 
                marker='o', linewidth=2, markersize=6)
    
    plt.xlabel("Number of Trees")
    plt.ylabel("Average Run Length (Steps)")
    plt.title("Average Run Length vs Number of Trees")
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(save_path, "run_length_vs_trees.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✔] Run length vs trees saved")


def analyze_with_all_scatterplots(df, corr_save_path, scatter_save_path):
    """Analyze with correlation matrix + all scatterplots"""
    numeric_df, corr_matrix = analyze_agent_simulation(df, corr_save_path)
    create_scatterplots(numeric_df, scatter_save_path, corr_matrix, mode='all')

def analyze_with_strong_scatterplots(df, corr_save_path, scatter_save_path, threshold=0.05):
    """Analyze with correlation matrix + scatterplots above threshold"""
    numeric_df, corr_matrix = analyze_agent_simulation(df, corr_save_path)
    create_scatterplots(numeric_df, scatter_save_path, corr_matrix, mode='strong', 
                       correlation_threshold=threshold)

def analyze_with_run_length_scatterplots(df, corr_save_path, scatter_save_path):
    """Analyze with correlation matrix + scatterplots above threshold"""
    numeric_df, corr_matrix = analyze_agent_simulation(df, corr_save_path)
    create_scatterplots(numeric_df, scatter_save_path, corr_matrix, mode='run_length')