import streamlit as st
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

# ====================================================
# Fonctions communes pour Ordonnanceur
# ====================================================

def calculer_sum_completion(ordre, durees):
    total, cumul = 0, 0
    for t in ordre:
        cumul += durees[t]
        total += cumul
    return total

def fitness_ord(ind, durees):
    val = calculer_sum_completion(ind, durees)
    return 1.0 / val if val > 0 else 1e9

def generer_population(n, pop_size):
    return [random.sample(range(n), n) for _ in range(pop_size)]

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

# ====================================================
# GA Rang pour Ordonnanceur
# ====================================================

def selection_rang(pop, fitnesses):
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

def ga_rang_ord(durees, pop_size=60, generations=200, taux_croisement=0.8, taux_mutation=0.2,
                methode_croisement='double', methode_mutation='inversion', elitisme=True, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(durees)
    pop = generer_population(n, pop_size)
    best, best_val = None, float('inf')
    history = []
    iterations_data = []
    for g in range(1, generations+1):
        valeurs = [calculer_sum_completion(ind, durees) for ind in pop]
        fitnesses = [fitness_ord(ind, durees) for ind in pop]
        for ind, val in zip(pop, valeurs):
            if val < best_val:
                best_val = val
                best = ind[:]
        history.append(best_val)
        iterations_data.append({'Génération': g, 'Meilleure valeur': best_val, 'Meilleur ordre': best[:]})
        new_pop = []
        if elitisme:
            best_idx = valeurs.index(min(valeurs))
            new_pop.append(pop[best_idx][:])
        while len(new_pop) < pop_size:
            p1 = selection_rang(pop, fitnesses)
            p2 = selection_rang(pop, fitnesses)
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
    return best, best_val, history, iterations_data

# ====================================================
# Recuit Simulé pour Ordonnanceur
# ====================================================

def generer_voisin_ord(ordre):
    i, j = random.sample(range(len(ordre)), 2)
    voisin = ordre[:]
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

def recuit_simule_ord(durees, T0=1000, Tmin=1e-3, alpha=0.95, iter_par_T=100):
    n = len(durees)
    ordre = list(range(n))
    random.shuffle(ordre)
    cout = calculer_sum_completion(ordre, durees)
    meilleur, meilleur_cout = ordre[:], cout
    T = T0
    iteration = 0
    history = [cout]
    while T > Tmin:
        for _ in range(iter_par_T):
            iteration += 1
            voisin = generer_voisin_ord(ordre)
            cout_voisin = calculer_sum_completion(voisin, durees)
            delta = cout_voisin - cout
            if delta < 0 or random.random() < math.exp(-delta / T):
                ordre, cout = voisin, cout_voisin
                if cout < meilleur_cout:
                    meilleur, meilleur_cout = ordre[:], cout
            history.append(cout)
        T *= alpha
    return meilleur, meilleur_cout, history

# ====================================================
# GA Roulette pour Ordonnanceur
# ====================================================

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

def ga_roulette_ord(durees, pop_size=60, generations=200, taux_croisement=0.8, taux_mutation=0.2,
                    methode_croisement='simple', methode_mutation='swap', elitisme=True, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(durees)
    pop = generer_population(n, pop_size)
    best, best_val = None, float('inf')
    history = []
    for g in range(1, generations+1):
        valeurs = [calculer_sum_completion(ind, durees) for ind in pop]
        fitnesses = [fitness_ord(ind, durees) for ind in pop]
        for ind, val in zip(pop, valeurs):
            if val < best_val:
                best_val = val
                best = ind[:]
        history.append(best_val)
        new_pop = []
        if elitisme:
            best_idx = valeurs.index(min(valeurs))
            new_pop.append(pop[best_idx][:])
        while len(new_pop) < pop_size:
            p1 = selection_roulette(pop, fitnesses)
            p2 = selection_roulette(pop, fitnesses)
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
    return best, best_val, history

# ====================================================
# Tabu pour Ordonnanceur
# ====================================================

def generer_voisins_ord(ordre):
    voisins = []
    for i in range(len(ordre)):
        for j in range(i + 1, len(ordre)):
            v = ordre[:]
            v[i], v[j] = v[j], v[i]
            voisins.append((v, (i, j)))
    return voisins

def recherche_tabou_ord(durees, max_iter=300, tenure=10):
    n = len(durees)
    courant = list(range(n))
    random.shuffle(courant)
    meilleur = courant[:]
    meilleur_cout = calculer_sum_completion(meilleur, durees)
    courant_cout = meilleur_cout
    tabou = {}
    history = [meilleur_cout]
    for it in range(1, max_iter + 1):
        voisins = generer_voisins_ord(courant)
        meilleur_voisin, meilleur_voisin_cout, move = None, float("inf"), None
        for v, m in voisins:
            val = calculer_sum_completion(v, durees)
            if (m not in tabou or val < meilleur_cout) and val < meilleur_voisin_cout:
                meilleur_voisin, meilleur_voisin_cout, move = v, val, m
        courant, courant_cout = meilleur_voisin, meilleur_voisin_cout
        tabou[move] = tenure
        for m in list(tabou.keys()):
            tabou[m] -= 1
            if tabou[m] <= 0:
                del tabou[m]
        if courant_cout < meilleur_cout:
            meilleur, meilleur_cout = courant[:], courant_cout
        history.append(meilleur_cout)
    return meilleur, meilleur_cout, history

# ====================================================
# Fonctions communes pour Voyageur (TSP)
# ====================================================

def generer_villes(nb_villes, largeur=100, hauteur=100):
    return [(random.uniform(0, largeur), random.uniform(0, hauteur)) for _ in range(nb_villes)]

def distance(v1, v2):
    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def construire_matrice_distances(villes):
    n = len(villes)
    return [[distance(villes[i], villes[j]) for j in range(n)] for i in range(n)]

def calculer_distance_totale(solution, matrice_distances):
    dist = 0
    for i in range(len(solution) - 1):
        dist += matrice_distances[solution[i]][solution[i + 1]]
    dist += matrice_distances[solution[-1]][solution[0]]
    return dist

def calculer_distance_totale_villes(solution, villes):
    dist = 0
    for i in range(len(solution) - 1):
        dist += distance(villes[solution[i]], villes[solution[i + 1]])
    dist += distance(villes[solution[-1]], villes[solution[0]])
    return dist

def creer_population_initiale(taille, n):
    population = []
    for _ in range(taille):
        ind = list(range(n))
        random.shuffle(ind)
        population.append(ind)
    return population

def fitness_tsp(ind, matrice_distances):
    d = calculer_distance_totale(ind, matrice_distances)
    return 1 / d if d > 0 else 0

# ====================================================
# GA Rang pour Voyageur
# ====================================================

def selection_rang_tsp(pop, fitnesses):
    ranked = sorted(zip(pop, fitnesses), key=lambda x: x[1])
    n = len(pop)
    probs = [(i + 1) / sum(range(1, n + 1)) for i in range(n)]
    r = random.random()
    cumul = 0
    for i, p in enumerate(probs):
        cumul += p
        if r <= cumul:
            return ranked[i][0][:]
    return ranked[-1][0][:]

def croisement_double_tsp(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    segment = p1[a:b]
    reste = [x for x in p2 if x not in segment]
    return reste[:a] + segment + reste[a:]

def mutation_inversion_tsp(ind):
    a, b = sorted(random.sample(range(len(ind)), 2))
    ind[a:b] = reversed(ind[a:b])
    return ind

def algo_genetique_rang_tsp(matrice_distances, taille_pop=100, generations=300, taux_croisement=0.8, taux_mutation=0.2):
    n = len(matrice_distances)
    population = creer_population_initiale(taille_pop, n)
    meilleur = None
    meilleure_distance = float('inf')
    history = []
    for g in range(generations):
        fitnesses = [fitness_tsp(ind, matrice_distances) for ind in population]
        nouvelle_population = []
        for _ in range(taille_pop // 2):
            p1 = selection_rang_tsp(population, fitnesses)
            p2 = selection_rang_tsp(population, fitnesses)
            if random.random() < taux_croisement:
                enfant = croisement_double_tsp(p1, p2)
            else:
                enfant = p1[:]
            if random.random() < taux_mutation:
                enfant = mutation_inversion_tsp(enfant)
            nouvelle_population.append(enfant)
        population = nouvelle_population
        for ind in population:
            d = calculer_distance_totale(ind, matrice_distances)
            if d < meilleure_distance:
                meilleure_distance = d
                meilleur = ind[:]
        history.append(meilleure_distance)
    return meilleur, meilleure_distance, history

# ====================================================
# Recuit Simulé pour Voyageur
# ====================================================

def generer_voisin_tsp(solution):
    voisin = solution[:]
    i, j = random.sample(range(len(solution)), 2)
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

def recuit_simule_tsp(villes, T0=1000, Tmin=1e-3, alpha=0.995, iterations=500):
    n = len(villes)
    solution = list(range(n))
    random.shuffle(solution)
    meilleure_solution = solution[:]
    meilleure_distance = calculer_distance_totale_villes(solution, villes)
    distance_actuelle = meilleure_distance
    T = T0
    history = [meilleure_distance]
    while T > Tmin:
        for _ in range(iterations):
            voisin = generer_voisin_tsp(solution)
            distance_voisin = calculer_distance_totale_villes(voisin, villes)
            delta = distance_voisin - distance_actuelle
            if delta < 0 or random.random() < math.exp(-delta / T):
                solution = voisin
                distance_actuelle = distance_voisin
                if distance_actuelle < meilleure_distance:
                    meilleure_solution = solution[:]
                    meilleure_distance = distance_actuelle
            history.append(meilleure_distance)
        T *= alpha
    return meilleure_solution, meilleure_distance, history

# ====================================================
# GA Roulette pour Voyageur
# ====================================================

def selection_roulette_tsp(pop, fitnesses):
    total = sum(fitnesses)
    probs = [f / total for f in fitnesses]
    r = random.random()
    cumul = 0
    for i, p in enumerate(probs):
        cumul += p
        if r <= cumul:
            return pop[i][:]
    return pop[-1][:]

def algo_genetique_roulette_tsp(matrice_distances, taille_pop=100, generations=300, taux_croisement=0.8, taux_mutation=0.2):
    n = len(matrice_distances)
    population = creer_population_initiale(taille_pop, n)
    meilleur = None
    meilleure_distance = float('inf')
    history = []
    for g in range(generations):
        fitnesses = [fitness_tsp(ind, matrice_distances) for ind in population]
        nouvelle_population = []
        for _ in range(taille_pop // 2):
            p1 = selection_roulette_tsp(population, fitnesses)
            p2 = selection_roulette_tsp(population, fitnesses)
            if random.random() < taux_croisement:
                enfant = croisement_double_tsp(p1, p2)
            else:
                enfant = p1[:]
            if random.random() < taux_mutation:
                enfant = mutation_inversion_tsp(enfant)
            nouvelle_population.append(enfant)
        population = nouvelle_population
        for ind in population:
            d = calculer_distance_totale(ind, matrice_distances)
            if d < meilleure_distance:
                meilleure_distance = d
                meilleur = ind[:]
        history.append(meilleure_distance)
    return meilleur, meilleure_distance, history

# ====================================================
# Tabu pour Voyageur
# ====================================================

def generer_voisin_swap_tsp(solution, i, j):
    voisin = solution[:]
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

def enumerer_voisins_swap_tsp(solution):
    n = len(solution)
    for i in range(n-1):
        for j in range(i+1, n):
            yield (i, j, generer_voisin_swap_tsp(solution, i, j))

def tabu_search_tsp(matrice_distances, max_iter=1000, tabu_tenure=15, nb_voisins_sample=200, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(matrice_distances)
    current = list(range(n))
    random.shuffle(current)
    current_dist = calculer_distance_totale(current, matrice_distances)
    best = current[:]
    best_dist = current_dist
    tabu_list = {}
    history = [best_dist]
    for it in range(1, max_iter + 1):
        candidats = []
        if nb_voisins_sample is None:
            for i in range(n-1):
                for j in range(i+1, n):
                    voisin = generer_voisin_swap_tsp(current, i, j)
                    d = calculer_distance_totale(voisin, matrice_distances)
                    candidats.append((i, j, voisin, d))
        else:
            tried = set()
            sample_size = min(nb_voisins_sample, n*(n-1)//2)
            while len(tried) < sample_size:
                i, j = sorted(random.sample(range(n), 2))
                if (i,j) in tried:
                    continue
                tried.add((i,j))
                voisin = generer_voisin_swap_tsp(current, i, j)
                d = calculer_distance_totale(voisin, matrice_distances)
                candidats.append((i, j, voisin, d))
        candidats.sort(key=lambda x: x[3])
        chosen = None
        for (i, j, voisin, d) in candidats:
            move = (i, j)
            is_tabou = move in tabu_list
            aspiration = (d < best_dist)
            if (not is_tabou) or aspiration:
                chosen = (i, j, voisin, d)
                break
        if chosen is None:
            chosen = candidats[0]
        i_chosen, j_chosen, next_sol, next_dist = chosen
        current = next_sol
        current_dist = next_dist
        key = (min(i_chosen, j_chosen), max(i_chosen, j_chosen))
        tabu_list[key] = tabu_tenure
        expired = []
        for k in list(tabu_list.keys()):
            tabu_list[k] -= 1
            if tabu_list[k] <= 0:
                expired.append(k)
        for k in expired:
            del tabu_list[k]
        if current_dist < best_dist:
            best = current[:]
            best_dist = current_dist
        history.append(best_dist)
    return best, best_dist, history

# ====================================================
# Visualisations
# ====================================================

def plot_gantt(ordre, durees):
    fig, ax = plt.subplots()
    cumul = 0
    for i, task in enumerate(ordre):
        ax.barh(i, durees[task], left=cumul, height=0.5, label=f'Tâche {task}')
        cumul += durees[task]
    ax.set_xlabel('Temps')
    ax.set_ylabel('Ordre')
    ax.set_title('Diagramme de Gantt')
    ax.legend()
    return fig

def plot_tsp(villes, solution):
    fig, ax = plt.subplots()
    x = [villes[i][0] for i in solution] + [villes[solution[0]][0]]
    y = [villes[i][1] for i in solution] + [villes[solution[0]][1]]
    ax.plot(x, y, 'o-')
    for i, (xi, yi) in enumerate(villes):
        ax.text(xi, yi, str(i))
    ax.set_title('Tournée TSP')
    return fig

def plot_convergence(history, title):
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel('Itération')
    ax.set_ylabel('Valeur')
    ax.set_title(title)
    return fig

# ====================================================
# Interface Streamlit
# ====================================================

st.title("Test d'Algorithmes d'Optimisation")

category = st.sidebar.selectbox("Catégorie", ["Ordonnanceur", "Voyageur"])

if category == "Ordonnanceur":
    st.header("Ordonnancement de Tâches")
    durees_input = st.text_input("Durées des tâches (séparées par des virgules)", "12,7,4,10,8,5,9,6,11,3")
    durees = [int(x) for x in durees_input.split(',')]
    algo = st.sidebar.selectbox("Algorithme", ["GA Rang", "GA Roulette", "Recuit Simulé", "Tabu"])
    if algo in ["GA Rang", "GA Roulette"]:
        pop_size = st.sidebar.slider("Taille population", 20, 200, 60)
        generations = st.sidebar.slider("Générations", 50, 500, 200)
        taux_croisement = st.sidebar.slider("Taux croisement", 0.0, 1.0, 0.8)
        taux_mutation = st.sidebar.slider("Taux mutation", 0.0, 1.0, 0.2)
        methode_croisement = st.sidebar.selectbox("Méthode croisement", ["simple", "double", "uniforme", "barycentrique"])
        methode_mutation = st.sidebar.selectbox("Méthode mutation", ["swap", "inversion", "scramble", "decalage"])
        elitisme = st.sidebar.checkbox("Élitisme", True)
        seed = st.sidebar.number_input("Seed", value=42)
    elif algo == "Recuit Simulé":
        T0 = st.sidebar.slider("Température initiale", 100, 2000, 1000)
        Tmin = st.sidebar.slider("Température min", 0.001, 1.0, 0.001)
        alpha = st.sidebar.slider("Alpha", 0.9, 0.999, 0.95)
        iter_par_T = st.sidebar.slider("Itérations par T", 10, 500, 100)
    elif algo == "Tabu":
        max_iter = st.sidebar.slider("Max itérations", 100, 1000, 300)
        tenure = st.sidebar.slider("Tenure", 5, 50, 10)

    if st.button("Lancer"):
        if algo == "GA Rang":
            best, val, hist, iterations_data = ga_rang_ord(durees, pop_size, generations, taux_croisement, taux_mutation, methode_croisement, methode_mutation, elitisme, seed)
        elif algo == "GA Roulette":
            best, val, hist = ga_roulette_ord(durees, pop_size, generations, taux_croisement, taux_mutation, methode_croisement, methode_mutation, elitisme, seed)
            iterations_data = [{'Génération': i+1, 'Meilleure valeur': h, 'Meilleur ordre': best} for i, h in enumerate(hist)]
        elif algo == "Recuit Simulé":
            best, val, hist = recuit_simule_ord(durees, T0, Tmin, alpha, iter_par_T)
            iterations_data = [{'Itération': i+1, 'Valeur': h, 'Ordre': best} for i, h in enumerate(hist)]
        elif algo == "Tabu":
            best, val, hist = recherche_tabou_ord(durees, max_iter, tenure)
            iterations_data = [{'Itération': i+1, 'Valeur': h, 'Ordre': best} for i, h in enumerate(hist)]
        st.write(f"Meilleure ordre: {best}")
        st.write(f"Somme des Cᵢ: {val}")
        st.pyplot(plot_gantt(best, durees))
        st.pyplot(plot_convergence(hist, "Convergence"))
        with st.expander("Résultats par itération"):
            st.dataframe(pd.DataFrame(iterations_data))

elif category == "Voyageur":
    st.header("Voyageur de Commerce")
    nb_villes = st.sidebar.slider("Nombre de villes", 5, 20, 10)
    villes = generer_villes(nb_villes)
    matrice = construire_matrice_distances(villes)
    algo = st.sidebar.selectbox("Algorithme", ["GA Rang", "GA Roulette", "Recuit Simulé", "Tabu"])
    if algo in ["GA Rang", "GA Roulette"]:
        taille_pop = st.sidebar.slider("Taille population", 20, 200, 100)
        generations = st.sidebar.slider("Générations", 50, 500, 300)
        taux_croisement = st.sidebar.slider("Taux croisement", 0.0, 1.0, 0.8)
        taux_mutation = st.sidebar.slider("Taux mutation", 0.0, 1.0, 0.2)
    elif algo == "Recuit Simulé":
        T0 = st.sidebar.slider("Température initiale", 100, 2000, 1000)
        Tmin = st.sidebar.slider("Température min", 0.001, 1.0, 0.001)
        alpha = st.sidebar.slider("Alpha", 0.9, 0.999, 0.995)
        iterations = st.sidebar.slider("Itérations par T", 10, 500, 500)
    elif algo == "Tabu":
        max_iter = st.sidebar.slider("Max itérations", 100, 2000, 1000)
        tabu_tenure = st.sidebar.slider("Tenure", 5, 50, 15)
        nb_voisins_sample = st.sidebar.slider("Échantillon voisins", 50, 500, 200)
        seed = st.sidebar.number_input("Seed", value=42)

    if st.button("Lancer"):
        if algo == "GA Rang":
            best, val, hist = algo_genetique_rang_tsp(matrice, taille_pop, generations, taux_croisement, taux_mutation)
        elif algo == "GA Roulette":
            best, val, hist = algo_genetique_roulette_tsp(matrice, taille_pop, generations, taux_croisement, taux_mutation)
        elif algo == "Recuit Simulé":
            best, val, hist = recuit_simule_tsp(villes, T0, Tmin, alpha, iterations)
        elif algo == "Tabu":
            best, val, hist = tabu_search_tsp(matrice, max_iter, tabu_tenure, nb_voisins_sample, seed)
        st.write(f"Meilleure tournée: {best}")
        st.write(f"Distance totale: {val:.2f}")
        st.pyplot(plot_tsp(villes, best))
        st.pyplot(plot_convergence(hist, "Convergence"))
