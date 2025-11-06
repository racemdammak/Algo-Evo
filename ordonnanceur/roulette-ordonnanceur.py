# ga_roulette_ordonnancement.py
import random

# -------------------------
# Fonctions utilitaires
# -------------------------
def calculer_sum_completion(ordre, durees):
    total, cumul = 0, 0
    for t in ordre:
        cumul += durees[t]
        total += cumul
    return total

def fitness(ind, durees):
    val = calculer_sum_completion(ind, durees)
    return 1.0 / val if val > 0 else 1e9

def generer_population(n, pop_size):
    return [random.sample(range(n), n) for _ in range(pop_size)]

# -------------------------
# Croisements (permutations)
# -------------------------
def croisement_simple(p1, p2):
    """One-point order-preserving crossover (simple)"""
    n = len(p1)
    cut = random.randint(1, n-2)
    enfant = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
    return enfant

def croisement_double(p1, p2):
    """Two-point / OX style (copie segment central de p1)"""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    segment = p1[a:b+1]
    reste = [x for x in p2 if x not in segment]
    return reste[:a] + segment + reste[a:]

def croisement_uniforme(p1, p2):
    """Uniform Order Crossover: mask + fill from p2"""
    n = len(p1)
    mask = [random.choice([0,1]) for _ in range(n)]
    child = [-1]*n
    for i in range(n):
        if mask[i] == 1:
            child[i] = p1[i]
    fill = [x for x in p2 if x not in child]
    k = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[k]; k += 1
    return child

def croisement_barycentrique(p1, p2, alpha=None):
    """
    Trick pour permutations :
    - convertir parent en vecteur de positions pos[i] = index_of_task_i
    - faire mélange barycentrique: alpha*pos1 + (1-alpha)*pos2
    - retourner permutation = argsort(mélange)
    """
    n = len(p1)
    if alpha is None:
        alpha = random.random()
    # positions: pos_task[t] = position index of task t in the parent
    pos1 = [0]*n
    pos2 = [0]*n
    for idx, task in enumerate(p1):
        pos1[task] = idx
    for idx, task in enumerate(p2):
        pos2[task] = idx
    mixed = [alpha*pos1[i] + (1-alpha)*pos2[i] for i in range(n)]
    # argsort of mixed gives order of tasks
    orden = sorted(range(n), key=lambda x: mixed[x])
    return orden

# -------------------------
# Mutations
# -------------------------
def mutation_swap(ind):
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def mutation_inversion(ind):
    a, b = sorted(random.sample(range(len(ind)), 2))
    ind[a:b+1] = reversed(ind[a:b+1])
    return ind

def mutation_scramble(ind):
    a, b = sorted(random.sample(range(len(ind)), 2))
    segment = ind[a:b+1]
    random.shuffle(segment)
    ind[a:b+1] = segment
    return ind

def mutation_decalage(ind):
    # shift one element to the right by one position (example)
    i = random.randint(0, len(ind)-2)
    val = ind.pop(i)
    ind.insert(i+1, val)
    return ind

# -------------------------
# Sélection roulette
# -------------------------
def selection_roulette(pop, fitnesses):
    total = sum(fitnesses)
    if total == 0:
        return random.choice(pop)[:]
    probs = [f/total for f in fitnesses]
    r = random.random()
    cumul = 0.0
    for i,p in enumerate(probs):
        cumul += p
        if r <= cumul:
            return pop[i][:]
    return pop[-1][:]

# -------------------------
# Algorithme GA - Roulette
# -------------------------
def ga_roulette(durees,
                pop_size=60,
                generations=200,
                taux_croisement=0.8,
                taux_mutation=0.2,
                methode_croisement='simple',   # 'simple','double','uniforme','barycentrique'
                methode_mutation='swap',       # 'swap','inversion','scramble','decalage'
                elitisme=True,
                seed=None):
    if seed is not None:
        random.seed(seed)

    n = len(durees)
    pop = generer_population(n, pop_size)
    best, best_val = None, float('inf')

    print("\n=== GA (Roulette) ===")
    print(f"Croisement: {methode_croisement}, Mutation: {methode_mutation}")

    for g in range(1, generations+1):
        valeurs = [calculer_sum_completion(ind, durees) for ind in pop]
        fitnesses = [fitness(ind, durees) for ind in pop]

        # mise à jour du meilleur (avant remplacement)
        for ind, val in zip(pop, valeurs):
            if val < best_val:
                best_val = val
                best = ind[:]
                print(f"→ Génération {g} | Nouvelle meilleure solution : {best_val}")

        new_pop = []
        if elitisme:
            # garder le meilleur courant
            best_idx = valeurs.index(min(valeurs))
            new_pop.append(pop[best_idx][:])

        while len(new_pop) < pop_size:
            p1 = selection_roulette(pop, fitnesses)
            p2 = selection_roulette(pop, fitnesses)

            # crossover
            if random.random() < taux_croisement:
                if methode_croisement == 'simple':
                    enfant = croisement_simple(p1, p2)
                elif methode_croisement == 'double':
                    enfant = croisement_double(p1, p2)
                elif methode_croisement == 'uniforme':
                    enfant = croisement_uniforme(p1, p2)
                elif methode_croisement == 'barycentrique':
                    enfant = croisement_barycentrique(p1, p2)
            else:
                enfant = p1[:]

            # mutation
            if random.random() < taux_mutation:
                if methode_mutation == 'swap':
                    enfant = mutation_swap(enfant)
                elif methode_mutation == 'inversion':
                    enfant = mutation_inversion(enfant)
                elif methode_mutation == 'scramble':
                    enfant = mutation_scramble(enfant)
                elif methode_mutation == 'decalage':
                    enfant = mutation_decalage(enfant)
                else:
                    enfant = mutation_swap(enfant)

            new_pop.append(enfant)

        pop = new_pop

    print("\nRésultat final (GA Roulette) :")
    print("Meilleure tournée :", best)
    print("Somme des Cᵢ :", best_val)
    return best, best_val

# -------------------------
# Exemple
# -------------------------
if __name__ == "__main__":
    durees = [12,7,4,10,8,5,9,6,11,3]
    ga_roulette(durees,
                pop_size=80,
                generations=150,
                taux_croisement=0.85,
                taux_mutation=0.25,
                methode_croisement='uniforme',
                methode_mutation='scramble',
                elitisme=True,
                seed=42)