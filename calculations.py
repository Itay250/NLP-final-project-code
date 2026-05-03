import json
import spacy
import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
import seaborn as sns

# --- Visualization Setup ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

# Load NLP model for POS filtering
nlp = spacy.load("en_core_web_sm")

# --- Lexicon Definition (Table 1 in your paper) ---
BOOSTERS = {"certainly", "absolutely", "definitely", "fact", "proven", "undoubtedly"}
HEDGES = {"maybe", "perhaps", "possibly", "likely", "unsure", "unclear", "guess"}

def calculate_intrinsic_confidence(tokens_data):
    """
    Calculates C_int: The mean log-probability of factual tokens (excluding ADJ/Fillers).
    Formula: C_int = (1/|T_fact|) * sum(log P(x_t))
    """
    factual_log_probs = []
    
    # Process full string for POS tagging
    full_text = "".join([t['token'] for t in tokens_data])
    doc = nlp(full_text)
    
    # We only care about factual carriers: NOUN, VERB, NUM, PROPN
    # We exclude: ADJ, ADV, DET, PRON, PUNCT
    fact_categories = {"NOUN", "VERB", "NUM", "PROPN"}
    
    for i, token in enumerate(doc):
        if token.pos_ in fact_categories and i < len(tokens_data):
            factual_log_probs.append(tokens_data[i]['log_prob'])
    
    if not factual_log_probs:
        return -10.0 
    
    return np.mean(factual_log_probs)

def calculate_decisiveness_coefficient(text):
    """
    Calculates D_coeff: Linguistic decisiveness based on lexical frequency.
    Formula: D_coeff = (Count(Boosters) - Count(Hedges)) / Total_Words
    """
    words = text.lower().split()
    if not words: return 0.0
    
    b_count = sum(1 for w in words if w in BOOSTERS)
    h_count = sum(1 for w in words if w in HEDGES)
    
    return (b_count - h_count) / len(words)

def calculate_unified_confidence(c_int, d_coeff, w1=0.7, w2=0.3):
    """
    Calculates S_total: The unified confidence score.
    Maps C_int (log space) to [0,1] via exp(), then weights with D_coeff.
    """
    # Convert log-prob to linear probability
    prob_score = exp(c_int) 
    
    # Combined weighted score passed through a sigmoid for calibration
    z = (w1 * prob_score) + (w2 * d_coeff)
    return 1 / (1 + exp(-z))

def graduated_accuracy_parser(generated_text, truth_tokens, safety_tokens):
    """
    Implements the 3-tier Accuracy Parser from Section 4.2.
    - True Answer: 1.0
    - Safety/Refusal: 0.5
    - Hallucination: 0.0
    """
    text = generated_text.lower()
    
    # Tier 1: Ground Truth
    if any(t.lower() in text for t in truth_tokens):
        return 1.0
    
    # Tier 2: Safety/Uncertainty
    if any(s.lower() in text for s in safety_tokens):
        return 0.5
    
    # Tier 3: Hallucination
    return 0.0

def process_results(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    final_table = []
    
    for entry in data:
        c_int = calculate_intrinsic_confidence(entry['tokens'])
        d_coeff = calculate_decisiveness_coefficient(entry['full_text'])
        s_total = calculate_unified_confidence(c_int, d_coeff)
        acc = graduated_accuracy_parser(
            entry['full_text'], 
            entry['ground_truth'], 
            entry['safety_tokens']
        )
        
        final_table.append({
            "id": entry['id'],
            "condition": entry['condition'],
            "C_int": c_int,
            "D_coeff": d_coeff,
            "S_total": s_total,
            "Accuracy": acc
        })
    
    return pd.DataFrame(final_table)

def generate_research_dashboard(df, model_name="LLM"):
    """
    Generates a 4-panel dashboard to analyze Hallucinated Decisiveness.
    """
    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    fig.suptitle(f'Decisiveness Calibration Analysis: {model_name}', fontsize=16, fontweight='bold')

    # Plot 1:
    df['S_bin'] = pd.cut(df['S_total'], bins=np.linspace(0, 1, 11), labels=np.linspace(0.05, 0.95, 10))
    calibration = df.groupby('S_bin', observed=False)['Accuracy'].mean()
    
    axes[0, 0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    axes[0, 0].plot(calibration.index.astype(float), calibration.values, marker='o', color='royalblue', linewidth=2, label=model_name)
    axes[0, 0].set_title("Unified Reliability Diagram ($S_{total}$ vs Accuracy)")
    axes[0, 0].set_xlabel("Predicted Confidence ($S_{total}$)")
    axes[0, 0].set_ylabel("Empirical Accuracy")
    axes[0, 0].set_xlim(0, 1.05)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2:
    outliers_y = df['D_coeff'].abs().nlargest(4).index
    outliers_x = df['C_int'].nsmallest(4).index
    df_clean = df.drop(index=outliers_y.union(outliers_x))
    df_plot = df_clean.sort_values(by='Accuracy', ascending=False)
    noise = np.random.normal(0, 0.0015, size=len(df_plot))
    
    sns.scatterplot(
        data=df_plot, 
        x="C_int", 
        y=df_plot["D_coeff"] + noise, 
        hue="Accuracy", 
        hue_order=[1.0, 0.5, 0.0], # Force Blue -> White -> Red layering
        palette={1.0: "royalblue", 0.5: "#dcdcdc", 0.0: "crimson"}, 
        alpha=0.8, 
        s=80,
        edgecolor="white",
        linewidth=0.5,
        ax=axes[0, 1]
    )
    
    # Visual cues for the "Trust Gap"
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    axes[0, 1].set_title("Intrinsic Math vs. Extrinsic Linguistics (Cleaned Both Axes)")
    axes[0, 1].set_xlabel("Intrinsic Log-Prob ($C_{int}$)")
    axes[0, 1].set_ylabel("Linguistic Decisiveness ($D_{coeff}$)")
    
    # Plot 3: 
    sns.kdeplot(data=df, x="S_total", hue="Accuracy", fill=True, common_norm=False, palette="coolwarm_r", alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title("Confidence Density by Accuracy Tier")
    axes[1, 0].set_xlabel("Unified Confidence ($S_{total}$)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlim(0.4, 0.8)

    # Plot 4:
    sns.barplot(data=df, x="condition", y="Accuracy", hue="condition", palette="viridis", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Mean Accuracy by Prompt Condition")
    axes[1, 1].set_xlabel("Condition")
    axes[1, 1].set_ylabel("Mean Accuracy")
    axes[1, 1].set_ylim(0.8, 1)
    
    # Save the output as a high-quality vector graphic
    output_filename = f"decisiveness_dashboard_{model_name.lower().replace(' ', '_')}.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Dashboard saved locally as {output_filename}")

# =======================================================================================

if __name__ == "__main__":
    # 1. Process Qwen Results
    try:
        print("Processing Qwen data...")
        df_qwen = process_results("qwen_results.json")
        print("\n--- Qwen Summary Statistics ---")
        print(df_qwen[['C_int', 'D_coeff', 'S_total', 'Accuracy']].describe().round(3))
        generate_research_dashboard(df_qwen, model_name="Qwen-2.5")
    except FileNotFoundError:
        print("File 'qwen_results.json' not found. Please ensure the file is in the same directory.")

    # 2. Process Pythia Results
    # try:
    #     print("\nProcessing Pythia data...")
    #     df_pythia = process_results("pythia_results.json")
    #     print("\n--- Pythia Summary Statistics ---")
    #     print(df_pythia[['C_int', 'D_coeff', 'S_total', 'Accuracy']].describe().round(3))
    #     generate_research_dashboard(df_pythia, model_name="Pythia")
    # except FileNotFoundError:
    #     print("File 'pythia_results.json' not found.")