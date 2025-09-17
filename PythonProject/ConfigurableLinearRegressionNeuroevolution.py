import argparse
import time
import re
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection       import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model          import LinearRegression
from sklearn.metrics               import confusion_matrix

# Very useful toolkit for Neuroevolution
from deap import base, tools
# Progress bar for notifying progress, when generations start
from tqdm import tqdm

# —————————————————————————————————————————————————————————————————————————
# Argument Parsing
# —————————————————————————————————————————————————————————————————————————
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve a word-weight model for 5-star regression"
    )
    parser.add_argument("--num_parents",     type=int,   default=3,
                        help="Number of parents that will survive into the next generation")
    parser.add_argument("--num_offspring",   type=int,   default=6,
                        help="Children per generation from parents")
    parser.add_argument("--num_mutations",   type=int,   default=2,
                        help="Mutant copies of each parent and child")
    parser.add_argument("--num_generations", type=int,   default=37,
                        help="Number of generations to simulate")
    parser.add_argument("--mutpb",           type=float, default=0.2,
                        help="Probability that each gene mutates")
    parser.add_argument("--sigma",           type=float, default=0.1,
                        help="Std-dev of Gaussian mutation")
    parser.add_argument("--max_reviews",     type=int,   default=2000,
                        help="Max number of reviews to load")
    return parser.parse_args()

args = parse_args()

# —————————————————————————————————————————————————————————————————————————
# Load Data and Split
# —————————————————————————————————————————————————————————————————————————
df = pd.read_csv("tripadvisor_hotel_reviews.csv")
if len(df) > args.max_reviews:
    df = df.sample(n=args.max_reviews, random_state=42).reset_index(drop=True)

texts   = df["Review"].astype(str).tolist()
ratings = df["Rating"].astype(float).values  # values from 1.0 to 5.0

# First split off 10% for test
X_temp, X_txt_test, y_temp, y_test = train_test_split(
    texts, ratings, test_size=0.1, random_state=42
)
# Then split 10% of the remainder for validation, .111/10 = 1/9 is 10% of the original
X_txt_train, X_txt_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42
)

print(f"Data splits → train={len(X_txt_train)}, val={len(X_txt_val)}, test={len(X_txt_test)}")

# Tokenizer setup
token_re = re.compile(r"\b[a-zA-Z']+\b")
def tokenize(doc):
    return [w.lower() for w in token_re.findall(doc)]

# —————————————————————————————————————————————————————————————————————————
# Linear Regression Fitting on Training Set
# —————————————————————————————————————————————————————————————————————————
vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False)
X_counts_train = vectorizer.fit_transform(X_txt_train)  # fit on train only
vocab      = vectorizer.get_feature_names_out()
V          = len(vocab)
word2idx   = {w:i for i,w in enumerate(vocab)}

# Fit a closed form linear model
reg = LinearRegression()
reg.fit(X_counts_train, y_train)

# Bias for review before looking at words
intercept      = reg.intercept_
# Unseen word weight
baseline_coefs = reg.coef_
default_weight = np.mean(baseline_coefs)

print(f"Vocabulary size: {V}, using CountVectorizer on training data")

# Genome length = intercept + default + vocab
genome_length = 2 + V

# —————————————————————————————————————————————————————————————————————————
# Multi Parent Blending
# —————————————————————————————————————————————————————————————————————————
import random
from deap import creator

def cx_multi_parent_weighted_blending_per_gene(parents, num_tokens=10):
    """
    Multi-parent blend with per-gene weight recalculation:
    for each gene position, distribute `num_tokens` at random
    among the K parents, normalize to weights, then compute
    the weighted sum for that locus.

    parents     : list of Individuals (all same length, float genes)
    num_tokens  : int, total discrete tokens to distribute per gene
    returns     : one new Individual
    """
    k = len(parents)
    genome_len = len(parents[0])
    child_genome = []

    # For each gene draw tokens
    for i in range(genome_len):
        # 1) Allocate tokens
        counts = [0] * k
        for _ in range(num_tokens):
            counts[random.randrange(k)] += 1

        # 2) Divide count by total tokens to get weights
        weights = [c / num_tokens for c in counts]

        # 3) Add part of the weight gene from each parent to the child's weight gene based on token weights
        gene_val = 0.0
        for p_idx, parent in enumerate(parents):
            gene_val += weights[p_idx] * parent[i]

        child_genome.append(gene_val)

    # Wrap up as an individual
    return creator.Individual(child_genome)

# —————————————————————————————————————————————————————————————————————————
# DEAP Setup Custom Methods
# —————————————————————————————————————————————————————————————————————————
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def attr_gene(idx):
    base_vals = [intercept, default_weight] + baseline_coefs.tolist()
    return random.gauss(base_vals[idx], 1.0)

def make_individual():
    return creator.Individual([attr_gene(i) for i in range(genome_length)])
# Used for initial setup
toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Used for mating
toolbox.register("mate", cx_multi_parent_weighted_blending_per_gene, num_tokens=10)
toolbox.register("select",     tools.selBest)

# —————————————————————————————————————————————————————————————————————————
# Validation Prediction and Fitness
# —————————————————————————————————————————————————————————————————————————
def predict_score(genome, doc):
    score = genome[0]
    for w in tokenize(doc):
        idx = word2idx.get(w)
        score += genome[1] if idx is None else genome[2 + idx]
    return min(5.0, max(1.0, score))

def evaluate_ind(ind):
    preds = np.array([predict_score(ind, d) for d in X_txt_val])
    mse   = np.mean((preds - y_val) ** 2)
    ind.rmse = np.sqrt(mse)
    return (mse,)

toolbox.register("evaluate", evaluate_ind)

# —————————————————————————————————————————————————————————————————————————
# Image
# —————————————————————————————————————————————————————————————————————————
import os
from graphviz import Digraph

def plot_ea_flow(pop_size, outpath='image.png'):
    """
    Draw the GA flow: selection → crossover → mutation → next generation.
    """
    dot = Digraph(comment='GA Pipeline', format='png')
    dot.attr(rankdir='LR', splines='ortho', fontsize='12')

    # Nodes
    dot.node('POP', label=f'Initial Population\nsize={pop_size}', shape='box', style='filled', fillcolor='lightgrey')
    dot.node('SEL', label=f'Select {args.num_parents} parents', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.node('CXO', label=f'Create {args.num_offspring} offspring\n(via crossover)', shape='box', style='rounded,filled', fillcolor='lightgreen')
    dot.node('MUT', label=f'Create mutated copies of each individual\nx{args.num_mutations}\nmutpb={args.mutpb}, σ={args.sigma}', shape='box', style='rounded,filled', fillcolor='lightpink')
    dot.node('NEXT', label='Next Generation', shape='box', style='filled', fillcolor='lightgrey')

    # Edges
    dot.edge('POP', 'SEL', arrowhead='vee')
    dot.edge('SEL', 'CXO', arrowhead='vee')
    dot.edge('CXO', 'MUT', arrowhead='vee')
    dot.edge('MUT', 'NEXT', arrowhead='vee')

    # Save to file
    dot.render(filename=os.path.splitext(outpath)[0], cleanup=True)



# —————————————————————————————————————————————————————————————————————————
# Evolutionary Loop
# —————————————————————————————————————————————————————————————————————————
def main():
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0.0, sigma=args.sigma, indpb=args.mutpb)

    pop_size = (args.num_parents + args.num_offspring) * (1 + args.num_mutations)
    pop      = toolbox.population(n=pop_size)

    # initial evaluation on validation
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # filename prefix
    prefix = f"Parent_{args.num_parents}_Offspring_{args.num_offspring}_Mutations_{args.num_mutations}_Generation_{args.num_generations}"

    try:
        plot_ea_flow(pop_size, outpath=f"{prefix}_image.png")
        print("GA pipeline diagram saved to", f"{prefix}_image.png")
    except Exception as e:
        print("Could not render GA diagram:", e)

    # progress bar for generations
    gen_iter = tqdm(
        range(1, args.num_generations + 1),
        desc="Evolving gens",
        unit="gen",
        ascii=True,
        file=sys.stdout,
        ncols=80
    )
    prev_best = None
    times = []

    for _ in gen_iter:
        t0 = time.time()

        # 1) selection
        parents = toolbox.select(pop, args.num_parents)

        # 2) crossover
        offspring = []
        for _ in range(args.num_offspring):
            child = toolbox.mate(parents)
            del child.fitness.values
            offspring.append(child)

        # 3) mutation
        survivors = parents + offspring
        mutants   = []
        for indiv in survivors:
            for _ in range(args.num_mutations):
                m = toolbox.clone(indiv)
                toolbox.mutate(m)
                del m.fitness.values
                mutants.append(m)

        # 4) next gen and eval
        pop = survivors + mutants
        invalids = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)

        best_mse  = pop[0].fitness.values[0]
        best_rmse = getattr(pop[0], "rmse", np.sqrt(best_mse))
        delta_pct = (prev_best - best_mse) / prev_best * 100 if prev_best else 0.0
        prev_best = best_mse

        elapsed = time.time() - t0
        times.append(elapsed)
        gen_iter.set_postfix({
            "MSE":  f"{best_mse:.4f}",
            "RMSE": f"{best_rmse:.3f}",
            "Δ%":   f"{delta_pct:.2f}",
            "sec":  f"{elapsed:.2f}"
        })
        gen_iter.refresh()

    champ = tools.selBest(pop, 1)[0]
    total_time = sum(times)

    print("\nChampion MSE:", champ.fitness.values[0],
          "RMSE≈", round(getattr(champ, "rmse", np.sqrt(champ.fitness.values[0])), 3))

    # —————————————————————————————————————————————————————————————————————
    # Evaluate champion on Training, Validation and Test sets
    # —————————————————————————————————————————————————————————————————————
    # training
    preds_train = np.array([predict_score(champ, d) for d in X_txt_train])
    true_train = np.array(y_train)
    train_mse = np.mean((preds_train - true_train) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mean_diff = np.mean(preds_train - true_train)
    cm_train = confusion_matrix(true_train.astype(int),
                                np.clip(np.rint(preds_train).astype(int), 1, 5),
                                labels=[1, 2, 3, 4, 5])

    # validation
    preds_val = np.array([predict_score(champ, d) for d in X_txt_val])
    true_val = np.array(y_val)
    val_mse = np.mean((preds_val - true_val) ** 2)
    val_rmse = np.sqrt(val_mse)
    val_mean_diff = np.mean(preds_val - true_val)
    cm_val = confusion_matrix(true_val.astype(int),
                              np.clip(np.rint(preds_val).astype(int), 1, 5),
                              labels=[1, 2, 3, 4, 5])

    # test
    preds_test = np.array([predict_score(champ, d) for d in X_txt_test])
    true_test = np.array(y_test)
    test_mse = np.mean((preds_test - true_test) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mean_diff = np.mean(preds_test - true_test)
    cm_test = confusion_matrix(true_test.astype(int),
                               np.clip(np.rint(preds_test).astype(int), 1, 5),
                               labels=[1, 2, 3, 4, 5])

    # write an output file
    out_file = f"{prefix}_output.txt"
    with open(out_file, "w") as f:
        f.write("Model Parameters:\n")
        f.write(f"num_parents:   {args.num_parents}\n")
        f.write(f"num_offspring: {args.num_offspring}\n")
        f.write(f"num_mutations: {args.num_mutations}\n")
        f.write(f"num_generations: {args.num_generations}\n")
        f.write(f"mutpb:         {args.mutpb}\n")
        f.write(f"sigma:         {args.sigma}\n")
        f.write(f"max_reviews:   {args.max_reviews}\n\n")

        f.write("Data splits:\n")
        f.write(f"train: {len(X_txt_train)}, val: {len(X_txt_val)}, test: {len(X_txt_test)}\n")
        f.write(f"vocab size: {V}\n")
        f.write(f"time: {total_time}\n\n")

        # ----------------------------- TRAINING ---------------------------------
        f.write("------------------------------------------\n")
        f.write("Training Set (Champion):\n")
        f.write(f"MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, Mean Diff: {train_mean_diff:.4f}\n")
        f.write("# columns = pred 1-5, rows = true 1-5\n")
        f.write("    1   2   3   4   5\n")
        for i, row in enumerate(cm_train, start=1):
            f.write(f"{i}  " + "  ".join(f"{v:2d}" for v in row) + "\n")

        # --------------------------- VALIDATION --------------------------------
        f.write("------------------------------------------\n")
        f.write("Validation Set (Champion):\n")
        f.write(f"MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, Mean Diff: {val_mean_diff:.4f}\n")
        f.write("# columns = pred 1-5, rows = true 1-5\n")
        f.write("    1   2   3   4   5\n")
        for i, row in enumerate(cm_val, start=1):
            f.write(f"{i}  " + "  ".join(f"{v:2d}" for v in row) + "\n")

        # ---------------------------- TEST ------------------------------------
        f.write("------------------------------------------\n")
        f.write("Test Set (Champion):\n")
        f.write(f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, Mean Diff: {test_mean_diff:.4f}\n")
        f.write("# columns = pred 1-5, rows = true 1-5\n")
        f.write("    1   2   3   4   5\n")
        for i, row in enumerate(cm_test, start=1):
            f.write(f"{i}  " + "  ".join(f"{v:2d}" for v in row) + "\n")

    print(f"All results written to {out_file}")

# in case I want to import methods without running
if __name__ == "__main__":
    main()