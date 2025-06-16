import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt


# How many steps per epoch
# STEPS_PER_EPOCH = 31501*2 # (realnewslike ) 31501
STEPS_PER_EPOCH = 22616*2 # (openwebtext) 22616

def plot_from_trainer_state(path_to_trainer_state_json: str | Path):
    path_to_file = Path(path_to_trainer_state_json)
    result_folder = path_to_file.parent.joinpath("plots")
    result_folder.mkdir(exist_ok=True, parents=True)

    # Load the JSON file
    with open(path_to_file, 'r') as f:
        data = json.load(f)

    log_history = data['log_history']

    # 1) -------- collect eval info --------
    eval_df = pd.DataFrame(
        [e for e in log_history if "eval_loss" in e and "step" in e]
    )
    eval_df["epoch"] = eval_df["step"] / STEPS_PER_EPOCH
    eval_df["perplexity"] = np.exp(eval_df["eval_loss"])
    eval_df = eval_df.sort_values("epoch").reset_index(drop=True)

    # improvements
    eval_df["epoch_delta"] = eval_df["epoch"].diff()
    eval_df["ppl_prev"] = eval_df["perplexity"].shift(1)
    eval_df["ppl_rel_improv_pct"] = (
        100 * (eval_df["ppl_prev"] - eval_df["perplexity"]) / eval_df["ppl_prev"]
    )
    eval_df["ppl_improv_pct_per_epoch"] = (
        eval_df["ppl_rel_improv_pct"] / eval_df["epoch_delta"]
    )

    # Prepare lists
    steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        # Training entry
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            train_losses.append(entry['loss'])
        # Eval entry
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])

    # Calculate epochs for both train and eval sequences
    train_epochs = [s / STEPS_PER_EPOCH for s in steps]
    eval_epochs = [s / STEPS_PER_EPOCH for s in eval_steps]

    # --- Plots by Step ---
    # Training loss vs step
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=steps, y=train_losses)
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Step')
    plt.tight_layout()
    for ext in ['pdf', 'svg', 'png']:
        plt.savefig(result_folder.joinpath(f'training_loss_vs_step.{ext}'), dpi=200)
    plt.close()

    # Eval loss vs step
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=eval_steps, y=eval_losses, color='orange')
    plt.xlabel('Step')
    plt.ylabel('Eval Loss')
    plt.title('Eval Loss vs Step')
    plt.tight_layout()
    for ext in ['pdf', 'svg', 'png']:
        plt.savefig(result_folder.joinpath(f'eval_loss_vs_step.{ext}'), dpi=200)
    plt.close()

    # --- Plots by Epoch ---
    # Training loss vs epoch
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=train_epochs, y=train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.tight_layout()
    for ext in ['pdf', 'svg', 'png']:
        plt.savefig(result_folder.joinpath(f'training_loss_vs_epoch.{ext}'), dpi=200)
    plt.close()

    # Eval loss vs epoch
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=eval_epochs, y=eval_losses, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Eval Loss')
    plt.title('Eval Loss vs Epoch')
    plt.tight_layout()
    for ext in ['pdf', 'svg', 'png']:
        plt.savefig(result_folder.joinpath(f'eval_loss_vs_epoch.{ext}'), dpi=200)
    plt.close()

    # 2) -------- plot perplexity --------
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=eval_df["epoch"], y=eval_df["perplexity"], color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Evaluation Perplexity vs Epoch")
    plt.tight_layout()
    for ext in ["pdf", "svg", "png"]:
        plt.savefig(result_folder / f"eval_perplexity_vs_epoch.{ext}", dpi=200)
    plt.close()

    # --- Combined train‑vs‑eval perplexity ---
    # Compute train perplexity and (optionally) smooth
    train_ppl = np.exp(train_losses)
    # train_ppl_smooth = pd.Series(train_ppl).rolling(window=1, center=True).mean()
    train_ppl_smooth = train_ppl

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=train_epochs, y=train_ppl_smooth,
                 label='Train', linewidth=1, alpha=.8)
    sns.lineplot(x=eval_df["epoch"], y=eval_df["perplexity"], label='Eval', linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Train vs Eval Perplexity")
    plt.ylim((1, 4))
    plt.legend()
    plt.tight_layout()
    for ext in ["pdf", "svg", "png"]:
        plt.savefig(result_folder / f"train_vs_eval_perplexity.{ext}", dpi=200)
    plt.close()


    # 3) -------- save table --------
    eval_df.round(4).to_csv(result_folder / "perplexity_milestones.csv", index=False)

    print("Plots saved as pdf, svg, and png with dpi=200.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_progress.py /path/to/trainer_state.json")
        sys.exit(1)
    json_path = Path(sys.argv[1])
    plot_from_trainer_state(json_path)

if __name__ == '__main__':
    main()