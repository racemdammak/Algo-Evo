# ga_rang_ordonnancement.py
import random

# -------------------------
# Utilitaires (idem)
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
# Même croisement & mutation que pour roulette
# -------------------------
def croisement_simple(p1, p2):
    n = len(p1)
    cut = random.randint(1, n-2)
    enfant = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
    return enfant

def croisement_double(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    segment = p1[a:b+1]
    reste = [x for x in p2 if x not in segment]
    return reste[:a] + segment + reste[a:]

def croisement_uniforme(p1, p2):
    n = len(p1)
    mask = [random.choice([0,1]) for _ in range(n)]
    child = [-1]*n
    for i in range(n):
        if mask[i]==1:
            child[i] = p1[i]
    fill = [x for x in p2 if x not in child]
    k=0
    for i in range(n):
        if child[i]==-1:
            child[i] = fill[k]; k+=1
    return child

def croisement_barycentrique(p1, p2, alpha=None):
    n = len(p1)
    if alpha is None: alpha = random.random()
    pos1 = [0]*n; pos2=[0]*n
    for idx, task in enumerate(p1): pos1[task]=idx
    for idx, task in enumerate(p2): pos2[task]=idx
    mixed = [alpha*pos1[i] + (1-alpha)*pos2[i] for i in range(n)]
    orden = sorted(range(n), key=lambda x: mixed[x])
    return orden

def mutation_swap(ind):
    i,j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def mutation_inversion(ind):
    a,b = sorted(random.sample(range(len(ind)), 2))
    ind[a:b+1] = reversed(ind[a:b+1])
    return ind

def mutation_scramble(ind):
    a,b = sorted(random.sample(range(len(ind)), 2))
    seg = ind[a:b+1]
    random.shuffle(seg)
    ind[a:b+1] = seg
    return ind

def mutation_decalage(ind):
    i = random.randint(0, len(ind)-2)
    val = ind.pop(i)
    ind.insert(i+1, val)
    return ind

# -------------------------
# Sélection par rang
# -------------------------
def selection_rang(pop, fitnesses):
    # assigner rangs : 1..n (1 = pire, n = meilleur)
    idx_sorted = sorted(range(len(pop)), key=lambda i: fitnesses[i])
    n = len(pop)
    rangs = [0]*n
    for rank, idx in enumerate(idx_sorted, start=1):
        rangs[idx] = rank
    total = sum(rangs)
    probs = [r/total for r in rangs]
    r = random.random(); cumul=0.0
    for i,p in enumerate(probs):
        cumul += p
        if r <= cumul:
            return pop[i][:]
    return pop[-1][:]

# -------------------------
# Algorithme GA - Rang
# -------------------------
def ga_rang(durees,
            pop_size=60,
            generations=200,
            taux_croisement=0.8,
            taux_mutation=0.2,
            methode_croisement='double',
            methode_mutation='inversion',
            elitisme=True,
            seed=None):
    if seed is not None:
        random.seed(seed)

    n = len(durees)
    pop = generer_population(n, pop_size)
    best, best_val = None, float('inf')

    print("\n=== GA (Rang) ===")
    print(f"Croisement: {methode_croisement}, Mutation: {methode_mutation}")

    for g in range(1, generations+1):
        valeurs = [calculer_sum_completion(ind, durees) for ind in pop]
        fitnesses = [fitness(ind, durees) for ind in pop]

        # update best
        for ind, val in zip(pop, valeurs):
            if val < best_val:
                best_val = val
                best = ind[:]
                print(f"→ Génération {g} | Nouvelle meilleure solution : {best_val}")

        new_pop = []
        if elitisme:
            best_idx = valeurs.index(min(valeurs))
            new_pop.append(pop[best_idx][:])

        while len(new_pop) < pop_size:
            p1 = selection_rang(pop, fitnesses)
            p2 = selection_rang(pop, fitnesses)

            # crossover
            if random.random() < taux_croisement:
                if methode_croisement == 'simple':
                    child = croisement_simple(p1, p2)
                elif methode_croisement == 'double':
                    child = croisement_double(p1, p2)
                elif methode_croisement == 'uniforme':
                    child = croisement_uniforme(p1, p2)
                elif methode_croisement == 'barycentrique':
                    child = croisement_barycentrique(p1, p2)
            else:
                child = p1[:]

            # mutation
            if random.random() < taux_mutation:
                if methode_mutation == 'swap':
                    child = mutation_swap(child)
                elif methode_mutation == 'inversion':
                    child = mutation_inversion(child)
                elif methode_mutation == 'scramble':
                    child = mutation_scramble(child)
                elif methode_mutation == 'decalage':
                    child = mutation_decalage(child)
                else:
                    child = mutation_inversion(child)

            new_pop.append(child)

        pop = new_pop

    print("\nRésultat final (GA Rang) :")
    print("Meilleure tournée :", best)
    print("Somme des Cᵢ :", best_val)
    return best, best_val

# -------------------------
# Exemple
# -------------------------
if __name__ == "__main__":
    durees = [12,7,4,10,8,5,9,6,11,3]
    ga_rang(durees,
            pop_size=80,
            generations=150,
            taux_croisement=0.85,
            taux_mutation=0.25,
            methode_croisement='barycentrique',
            methode_mutation='inversion',
            elitisme=True,
            seed=42)
