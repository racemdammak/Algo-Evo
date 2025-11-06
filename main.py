import streamlit as st
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import time
import numpy as np
from datetime import datetime

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
# GA Rang pour Ordonnanceur (avec suivi détaillé)
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
    
    # Statistiques supplémentaires
    avg_fitness_history = []
    diversity_history = []
    best_individual_history = []
    
    for g in range(1, generations+1):
        valeurs = [calculer_sum_completion(ind, durees) for ind in pop]
        fitnesses = [fitness_ord(ind, durees) for ind in pop]
        
        # Calcul des statistiques
        avg_fitness = np.mean(fitnesses)
        avg_fitness_history.append(avg_fitness)
        
        # Mesure de diversité (distance moyenne entre individus)
        diversity = 0
        count = 0
        for i in range(len(pop)):
            for j in range(i+1, len(pop)):
                diversity += sum(1 for a, b in zip(pop[i], pop[j]) if a != b)
                count += 1
        diversity = diversity / count if count > 0 else 0
        diversity_history.append(diversity)
        
        for ind, val in zip(pop, valeurs):
            if val < best_val:
                best_val = val
                best = ind[:]
                best_individual_history.append((g, best[:], best_val))
        
        history.append(best_val)
        
        # Données détaillées pour chaque génération
        gen_data = {
            'Génération': g, 
            'Meilleure valeur': best_val, 
            'Meilleur ordre': best[:],
            'Valeur moyenne': np.mean(valeurs),
            'Écart-type': np.std(valeurs),
            'Fitness moyenne': avg_fitness,
            'Diversité': diversity,
            'Temps cumulé optimal': best_val
        }
        iterations_data.append(gen_data)
        
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
    
    stats = {
        'avg_fitness': avg_fitness_history,
        'diversity': diversity_history,
        'best_individuals': best_individual_history
    }
    
    return best, best_val, history, iterations_data, stats

# ====================================================
# Recuit Simulé pour Ordonnanceur (avec suivi détaillé)
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
    iterations_data = []
    
    # Statistiques
    temperature_history = [T]
    acceptance_rate_history = []
    accepted_moves = 0
    
    while T > Tmin:
        for _ in range(iter_par_T):
            iteration += 1
            voisin = generer_voisin_ord(ordre)
            cout_voisin = calculer_sum_completion(voisin, durees)
            delta = cout_voisin - cout
            
            accepted = False
            if delta < 0 or random.random() < math.exp(-delta / T):
                ordre, cout = voisin, cout_voisin
                accepted = True
                accepted_moves += 1
                
                if cout < meilleur_cout:
                    meilleur, meilleur_cout = ordre[:], cout
            
            # Enregistrement des données détaillées
            iter_data = {
                'Itération': iteration,
                'Valeur': cout,
                'Meilleur valeur': meilleur_cout,
                'Température': T,
                'Delta': delta,
                'Accepté': accepted,
                'Ordre courant': ordre[:],
                'Ordre optimal': meilleur[:]
            }
            iterations_data.append(iter_data)
            history.append(cout)
        
        # Calcul du taux d'acceptation
        acceptance_rate = accepted_moves / iter_par_T
        acceptance_rate_history.append(acceptance_rate)
        accepted_moves = 0
        
        temperature_history.append(T)
        T *= alpha
    
    stats = {
        'temperature': temperature_history,
        'acceptance_rate': acceptance_rate_history
    }
    
    return meilleur, meilleur_cout, history, iterations_data, stats

# ====================================================
# GA Roulette pour Ordonnanceur (avec suivi détaillé)
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
    iterations_data = []
    
    # Statistiques supplémentaires
    avg_fitness_history = []
    diversity_history = []
    
    for g in range(1, generations+1):
        valeurs = [calculer_sum_completion(ind, durees) for ind in pop]
        fitnesses = [fitness_ord(ind, durees) for ind in pop]
        
        # Calcul des statistiques
        avg_fitness = np.mean(fitnesses)
        avg_fitness_history.append(avg_fitness)
        
        # Mesure de diversité
        diversity = 0
        count = 0
        for i in range(len(pop)):
            for j in range(i+1, len(pop)):
                diversity += sum(1 for a, b in zip(pop[i], pop[j]) if a != b)
                count += 1
        diversity = diversity / count if count > 0 else 0
        diversity_history.append(diversity)
        
        for ind, val in zip(pop, valeurs):
            if val < best_val:
                best_val = val
                best = ind[:]
        
        history.append(best_val)
        
        # Données détaillées pour chaque génération
        gen_data = {
            'Génération': g, 
            'Meilleure valeur': best_val, 
            'Meilleur ordre': best[:],
            'Valeur moyenne': np.mean(valeurs),
            'Écart-type': np.std(valeurs),
            'Fitness moyenne': avg_fitness,
            'Diversité': diversity,
            'Temps cumulé optimal': best_val
        }
        iterations_data.append(gen_data)
        
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
    
    stats = {
        'avg_fitness': avg_fitness_history,
        'diversity': diversity_history
    }
    
    return best, best_val, history, iterations_data, stats

# ====================================================
# Tabu pour Ordonnanceur (avec suivi détaillé)
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
    iterations_data = []
    
    # Statistiques
    tabu_size_history = []
    improvement_history = []
    
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
        
        improvement = meilleur_cout - courant_cout if courant_cout < meilleur_cout else 0
        improvement_history.append(improvement)
        
        if courant_cout < meilleur_cout:
            meilleur, meilleur_cout = courant[:], courant_cout
        
        history.append(meilleur_cout)
        tabu_size_history.append(len(tabou))
        
        # Données détaillées pour chaque itération
        iter_data = {
            'Itération': it,
            'Valeur courante': courant_cout,
            'Meilleur valeur': meilleur_cout,
            'Taille liste Tabu': len(tabou),
            'Amélioration': improvement,
            'Ordre courant': courant[:],
            'Ordre optimal': meilleur[:],
            'Move interdit': move
        }
        iterations_data.append(iter_data)
    
    stats = {
        'tabu_size': tabu_size_history,
        'improvement': improvement_history
    }
    
    return meilleur, meilleur_cout, history, iterations_data, stats

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
# Opérateurs génétiques pour TSP
# ====================================================

def croisement_simple_tsp(p1, p2):
    n = len(p1)
    cut = random.randint(1, n-2)
    enfant = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
    return enfant

def croisement_double_tsp(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    segment = p1[a:b+1]
    reste = [x for x in p2 if x not in segment]
    return reste[:a] + segment + reste[a:]

def croisement_uniforme_tsp(p1, p2):
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

def croisement_barycentrique_tsp(p1, p2, alpha=None):
    n = len(p1)
    if alpha is None: alpha = random.random()
    pos1 = [0]*n; pos2=[0]*n
    for idx, ville in enumerate(p1): pos1[ville]=idx
    for idx, ville in enumerate(p2): pos2[ville]=idx
    mixed = [alpha*pos1[i] + (1-alpha)*pos2[i] for i in range(n)]
    orden = sorted(range(n), key=lambda x: mixed[x])
    return orden

def mutation_swap_tsp(ind):
    i,j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def mutation_inversion_tsp(ind):
    a,b = sorted(random.sample(range(len(ind)), 2))
    ind[a:b+1] = reversed(ind[a:b+1])
    return ind

def mutation_scramble_tsp(ind):
    a,b = sorted(random.sample(range(len(ind)), 2))
    seg = ind[a:b+1]
    random.shuffle(seg)
    ind[a:b+1] = seg
    return ind

def mutation_decalage_tsp(ind):
    i = random.randint(0, len(ind)-2)
    val = ind.pop(i)
    ind.insert(i+1, val)
    return ind

# ====================================================
# GA Rang pour Voyageur (avec suivi détaillé)
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

def algo_genetique_rang_tsp(matrice_distances, taille_pop=100, generations=300, taux_croisement=0.8, 
                           taux_mutation=0.2, methode_croisement='double', methode_mutation='inversion', 
                           elitisme=True, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(matrice_distances)
    population = creer_population_initiale(taille_pop, n)
    meilleur = None
    meilleure_distance = float('inf')
    history = []
    iterations_data = []
    
    # Statistiques
    avg_distance_history = []
    diversity_history = []
    
    for g in range(generations):
        distances = [calculer_distance_totale(ind, matrice_distances) for ind in population]
        fitnesses = [fitness_tsp(ind, matrice_distances) for ind in population]
        
        # Statistiques
        avg_distance = np.mean(distances)
        avg_distance_history.append(avg_distance)
        
        # Diversité
        diversity = 0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                diversity += sum(1 for a, b in zip(population[i], population[j]) if a != b)
                count += 1
        diversity = diversity / count if count > 0 else 0
        diversity_history.append(diversity)
        
        for ind, d in zip(population, distances):
            if d < meilleure_distance:
                meilleure_distance = d
                meilleur = ind[:]
        
        history.append(meilleure_distance)
        
        # Données détaillées
        gen_data = {
            'Génération': g+1,
            'Meilleure distance': meilleure_distance,
            'Distance moyenne': avg_distance,
            'Diversité': diversity,
            'Meilleure tournée': meilleur[:]
        }
        iterations_data.append(gen_data)
        
        nouvelle_population = []
        
        # Élitisme
        if elitisme:
            best_idx = distances.index(min(distances))
            nouvelle_population.append(population[best_idx][:])
        
        while len(nouvelle_population) < taille_pop:
            p1 = selection_rang_tsp(population, fitnesses)
            p2 = selection_rang_tsp(population, fitnesses)
            
            if random.random() < taux_croisement:
                if methode_croisement == 'simple':
                    enfant = croisement_simple_tsp(p1, p2)
                elif methode_croisement == 'double':
                    enfant = croisement_double_tsp(p1, p2)
                elif methode_croisement == 'uniforme':
                    enfant = croisement_uniforme_tsp(p1, p2)
                elif methode_croisement == 'barycentrique':
                    enfant = croisement_barycentrique_tsp(p1, p2)
                else:
                    enfant = croisement_double_tsp(p1, p2)
            else:
                enfant = p1[:]
            
            if random.random() < taux_mutation:
                if methode_mutation == 'swap':
                    enfant = mutation_swap_tsp(enfant)
                elif methode_mutation == 'inversion':
                    enfant = mutation_inversion_tsp(enfant)
                elif methode_mutation == 'scramble':
                    enfant = mutation_scramble_tsp(enfant)
                elif methode_mutation == 'decalage':
                    enfant = mutation_decalage_tsp(enfant)
                else:
                    enfant = mutation_inversion_tsp(enfant)
            
            nouvelle_population.append(enfant)
        
        population = nouvelle_population
    
    stats = {
        'avg_distance': avg_distance_history,
        'diversity': diversity_history
    }
    
    return meilleur, meilleure_distance, history, iterations_data, stats

# ====================================================
# Recuit Simulé pour Voyageur (avec suivi détaillé)
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
    iterations_data = []
    
    # Statistiques
    temperature_history = [T]
    acceptance_rate_history = []
    accepted_moves = 0
    iteration_count = 0
    
    while T > Tmin:
        for _ in range(iterations):
            iteration_count += 1
            voisin = generer_voisin_tsp(solution)
            distance_voisin = calculer_distance_totale_villes(voisin, villes)
            delta = distance_voisin - distance_actuelle
            
            accepted = False
            if delta < 0 or random.random() < math.exp(-delta / T):
                solution = voisin
                distance_actuelle = distance_voisin
                accepted = True
                accepted_moves += 1
                
                if distance_actuelle < meilleure_distance:
                    meilleure_solution = solution[:]
                    meilleure_distance = distance_actuelle
            
            # Données détaillées
            iter_data = {
                'Itération': iteration_count,
                'Distance courante': distance_actuelle,
                'Meilleure distance': meilleure_distance,
                'Température': T,
                'Delta': delta,
                'Accepté': accepted,
                'Tournée courante': solution[:]
            }
            iterations_data.append(iter_data)
            history.append(meilleure_distance)
        
        # Taux d'acceptation
        acceptance_rate = accepted_moves / iterations
        acceptance_rate_history.append(acceptance_rate)
        accepted_moves = 0
        
        temperature_history.append(T)
        T *= alpha
    
    stats = {
        'temperature': temperature_history,
        'acceptance_rate': acceptance_rate_history
    }
    
    return meilleure_solution, meilleure_distance, history, iterations_data, stats

# ====================================================
# GA Roulette pour Voyageur (avec suivi détaillé)
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

def algo_genetique_roulette_tsp(matrice_distances, taille_pop=100, generations=300, taux_croisement=0.8, 
                               taux_mutation=0.2, methode_croisement='double', methode_mutation='inversion', 
                               elitisme=True, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(matrice_distances)
    population = creer_population_initiale(taille_pop, n)
    meilleur = None
    meilleure_distance = float('inf')
    history = []
    iterations_data = []
    
    # Statistiques
    avg_distance_history = []
    diversity_history = []
    
    for g in range(generations):
        distances = [calculer_distance_totale(ind, matrice_distances) for ind in population]
        fitnesses = [fitness_tsp(ind, matrice_distances) for ind in population]
        
        # Statistiques
        avg_distance = np.mean(distances)
        avg_distance_history.append(avg_distance)
        
        # Diversité
        diversity = 0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                diversity += sum(1 for a, b in zip(population[i], population[j]) if a != b)
                count += 1
        diversity = diversity / count if count > 0 else 0
        diversity_history.append(diversity)
        
        for ind, d in zip(population, distances):
            if d < meilleure_distance:
                meilleure_distance = d
                meilleur = ind[:]
        
        history.append(meilleure_distance)
        
        # Données détaillées
        gen_data = {
            'Génération': g+1,
            'Meilleure distance': meilleure_distance,
            'Distance moyenne': avg_distance,
            'Diversité': diversity,
            'Meilleure tournée': meilleur[:]
        }
        iterations_data.append(gen_data)
        
        nouvelle_population = []
        
        # Élitisme
        if elitisme:
            best_idx = distances.index(min(distances))
            nouvelle_population.append(population[best_idx][:])
        
        while len(nouvelle_population) < taille_pop:
            p1 = selection_roulette_tsp(population, fitnesses)
            p2 = selection_roulette_tsp(population, fitnesses)
            
            if random.random() < taux_croisement:
                if methode_croisement == 'simple':
                    enfant = croisement_simple_tsp(p1, p2)
                elif methode_croisement == 'double':
                    enfant = croisement_double_tsp(p1, p2)
                elif methode_croisement == 'uniforme':
                    enfant = croisement_uniforme_tsp(p1, p2)
                elif methode_croisement == 'barycentrique':
                    enfant = croisement_barycentrique_tsp(p1, p2)
                else:
                    enfant = croisement_double_tsp(p1, p2)
            else:
                enfant = p1[:]
            
            if random.random() < taux_mutation:
                if methode_mutation == 'swap':
                    enfant = mutation_swap_tsp(enfant)
                elif methode_mutation == 'inversion':
                    enfant = mutation_inversion_tsp(enfant)
                elif methode_mutation == 'scramble':
                    enfant = mutation_scramble_tsp(enfant)
                elif methode_mutation == 'decalage':
                    enfant = mutation_decalage_tsp(enfant)
                else:
                    enfant = mutation_inversion_tsp(enfant)
            
            nouvelle_population.append(enfant)
        
        population = nouvelle_population
    
    stats = {
        'avg_distance': avg_distance_history,
        'diversity': diversity_history
    }
    
    return meilleur, meilleure_distance, history, iterations_data, stats

# ====================================================
# Tabu pour Voyageur (avec suivi détaillé)
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
    iterations_data = []
    
    # Statistiques
    tabu_size_history = []
    improvement_history = []
    
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
        
        improvement = best_dist - current_dist if current_dist < best_dist else 0
        improvement_history.append(improvement)
        
        if current_dist < best_dist:
            best = current[:]
            best_dist = current_dist
        
        history.append(best_dist)
        tabu_size_history.append(len(tabu_list))
        
        # Données détaillées
        iter_data = {
            'Itération': it,
            'Distance courante': current_dist,
            'Meilleure distance': best_dist,
            'Taille liste Tabu': len(tabu_list),
            'Amélioration': improvement,
            'Move': key,
            'Tournée courante': current[:],
            'Tournée optimale': best[:]
        }
        iterations_data.append(iter_data)
    
    stats = {
        'tabu_size': tabu_size_history,
        'improvement': improvement_history
    }
    
    return best, best_dist, history, iterations_data, stats

# ====================================================
# Visualisations améliorées
# ====================================================

def plot_gantt(ordre, durees):
    fig, ax = plt.subplots(figsize=(10, 6))
    cumul = 0
    for i, task in enumerate(ordre):
        ax.barh(i, durees[task], left=cumul, height=0.5, label=f'Tâche {task}', alpha=0.7)
        ax.text(cumul + durees[task]/2, i, f'T{task}\n({durees[task]})', 
                ha='center', va='center', color='white', weight='bold')
        cumul += durees[task]
    ax.set_xlabel('Temps')
    ax.set_ylabel('Ordre d\'exécution')
    ax.set_title('Diagramme de Gantt - Ordonnancement des Tâches')
    ax.grid(axis='x', alpha=0.3)
    return fig

def plot_tsp(villes, solution):
    fig, ax = plt.subplots(figsize=(10, 8))
    x = [villes[i][0] for i in solution] + [villes[solution[0]][0]]
    y = [villes[i][1] for i in solution] + [villes[solution[0]][1]]
    
    # Plot du chemin
    ax.plot(x, y, 'o-', linewidth=2, markersize=8, alpha=0.7)
    
    # Annotation des villes
    for i, (xi, yi) in enumerate(villes):
        ax.text(xi, yi, f' {i}', fontsize=12, ha='left', va='bottom')
        ax.plot(xi, yi, 'o', markersize=10, color='red')
    
    ax.set_xlabel('Coordonnée X')
    ax.set_ylabel('Coordonnée Y')
    ax.set_title(f'Tournée TSP - {len(villes)} villes')
    ax.grid(True, alpha=0.3)
    return fig

def plot_convergence(history, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, linewidth=2)
    ax.set_xlabel('Itération')
    ax.set_ylabel('Valeur optimale')
    ax.set_title(f'Convergence - {title}')
    ax.grid(True, alpha=0.3)
    
    # Ajouter des annotations pour les points importants
    if len(history) > 0:
        ax.axhline(y=min(history), color='r', linestyle='--', alpha=0.7, label=f'Minimum: {min(history):.2f}')
        ax.axhline(y=max(history), color='g', linestyle='--', alpha=0.7, label=f'Maximum: {max(history):.2f}')
        ax.legend()
    
    return fig

def plot_task_durations(durees):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(durees)), durees, alpha=0.7, color='skyblue')
    ax.set_xlabel('Tâche')
    ax.set_ylabel('Durée')
    ax.set_title('Durées des Tâches')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_completion_times(ordre, durees):
    cumul = 0
    times = []
    tasks = []
    for t in ordre:
        cumul += durees[t]
        times.append(cumul)
        tasks.append(f'T{t}')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(tasks, times, alpha=0.7, color='lightgreen')
    ax.set_xlabel('Tâche dans l\'ordre')
    ax.set_ylabel('Temps de complétion cumulé')
    ax.set_title('Temps de Complétion Cumulés par Tâche')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    return fig

def plot_tour_distances(villes, solution):
    distances = []
    steps = []
    for i in range(len(solution) - 1):
        d = distance(villes[solution[i]], villes[solution[i + 1]])
        distances.append(d)
        steps.append(f'{solution[i]}→{solution[i+1]}')
    
    d = distance(villes[solution[-1]], villes[solution[0]])
    distances.append(d)
    steps.append(f'{solution[-1]}→{solution[0]}')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(steps, distances, alpha=0.7, color='orange')
    ax.set_xlabel('Étape du parcours')
    ax.set_ylabel('Distance')
    ax.set_title('Distances entre Villes dans la Tournée')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', rotation=45)
    
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    return fig

def plot_additional_stats(stats, algorithm_name):
    """Fonction pour afficher des statistiques supplémentaires"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Statistiques Détaillées - {algorithm_name}', fontsize=16)
    
    # Plot 1: Diversité de la population
    if 'diversity' in stats:
        axes[0,0].plot(stats['diversity'])
        axes[0,0].set_title('Diversité de la Population')
        axes[0,0].set_xlabel('Génération')
        axes[0,0].set_ylabel('Diversité')
        axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Fitness moyenne
    if 'avg_fitness' in stats:
        axes[0,1].plot(stats['avg_fitness'])
        axes[0,1].set_title('Fitness Moyenne')
        axes[0,1].set_xlabel('Génération')
        axes[0,1].set_ylabel('Fitness Moyenne')
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Taille de la liste Tabu
    if 'tabu_size' in stats:
        axes[1,0].plot(stats['tabu_size'])
        axes[1,0].set_title('Taille de la Liste Tabu')
        axes[1,0].set_xlabel('Itération')
        axes[1,0].set_ylabel('Taille')
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Taux d'acceptation
    if 'acceptance_rate' in stats:
        axes[1,1].plot(stats['acceptance_rate'])
        axes[1,1].set_title('Taux d\'Acceptation')
        axes[1,1].set_xlabel('Cycle')
        axes[1,1].set_ylabel('Taux d\'acceptation')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ====================================================
# Interface Streamlit améliorée
# ====================================================

st.set_page_config(page_title="Optimisation Metaheuristique", layout="wide")

st.title("🔬 Plateforme d'Optimisation Metaheuristique")

# Sidebar pour la configuration
st.sidebar.header("Configuration")

category = st.sidebar.selectbox("Catégorie de Problème", ["Ordonnanceur", "Voyageur"])

if category == "Ordonnanceur":
    st.header("📊 Ordonnancement de Tâches")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres des Tâches")
        durees_input = st.text_input("Durées des tâches (séparées par des virgules)", "12,7,4,10,8,5,9,6,11,3")
        durees = [int(x.strip()) for x in durees_input.split(',')]
        st.write(f"Nombre de tâches: {len(durees)}")
        st.write(f"Durée totale: {sum(durees)}")
    
    with col2:
        st.subheader("Configuration de l'Algorithme")
        algo = st.selectbox("Algorithme", ["GA Rang", "GA Roulette", "Recuit Simulé", "Tabu"])
    
    # Paramètres spécifiques aux algorithmes
    if algo in ["GA Rang", "GA Roulette"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            pop_size = st.slider("Taille population", 20, 200, 60)
            generations = st.slider("Générations", 50, 1000, 200)
        with col2:
            taux_croisement = st.slider("Taux croisement", 0.0, 1.0, 0.8)
            taux_mutation = st.slider("Taux mutation", 0.0, 1.0, 0.2)
        with col3:
            methode_croisement = st.selectbox("Méthode croisement", ["simple", "double", "uniforme", "barycentrique"])
            methode_mutation = st.selectbox("Méthode mutation", ["swap", "inversion", "scramble", "decalage"])
            elitisme = st.checkbox("Élitisme", True)
            seed = st.number_input("Seed", value=42)
    
    elif algo == "Recuit Simulé":
        col1, col2 = st.columns(2)
        with col1:
            T0 = st.slider("Température initiale", 100, 5000, 1000)
            Tmin = st.slider("Température min", 0.001, 1.0, 0.001)
        with col2:
            alpha = st.slider("Alpha", 0.9, 0.999, 0.95)
            iter_par_T = st.slider("Itérations par T", 10, 500, 100)
    
    elif algo == "Tabu":
        col1, col2 = st.columns(2)
        with col1:
            max_iter = st.slider("Max itérations", 100, 2000, 300)
        with col2:
            tenure = st.slider("Tenure", 5, 50, 10)

    if st.button("🚀 Lancer l'Optimisation", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initialisation de l'algorithme...")
        
        start_time = time.time()
        
        # Exécution de l'algorithme sélectionné
        if algo == "GA Rang":
            best, val, hist, iterations_data, stats = ga_rang_ord(durees, pop_size, generations, taux_croisement, taux_mutation, methode_croisement, methode_mutation, elitisme, seed)
        elif algo == "GA Roulette":
            best, val, hist, iterations_data, stats = ga_roulette_ord(durees, pop_size, generations, taux_croisement, taux_mutation, methode_croisement, methode_mutation, elitisme, seed)
        elif algo == "Recuit Simulé":
            best, val, hist, iterations_data, stats = recuit_simule_ord(durees, T0, Tmin, alpha, iter_par_T)
        elif algo == "Tabu":
            best, val, hist, iterations_data, stats = recherche_tabou_ord(durees, max_iter, tenure)
        
        execution_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("Optimisation terminée!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats
        st.success("✅ Optimisation terminée avec succès!")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Meilleure valeur", f"{val:.2f}")
        with col2:
            st.metric("Temps d'exécution", f"{execution_time:.2f}s")
        with col3:
            st.metric("Nombre d'itérations", len(hist))
        with col4:
            improvement = ((sum(durees) * len(durees) / 2 - val) / (sum(durees) * len(durees) / 2)) * 100
            st.metric("Amélioration", f"{improvement:.1f}%")
        
        st.write(f"**Meilleur ordre trouvé:** {best}")
        
        # Visualisations
        st.subheader("📈 Visualisations")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Gantt", "Convergence", "Statistiques", "Données Détaillées", "Résumé"])
        
        with tab1:
            st.pyplot(plot_gantt(best, durees))
        
        with tab2:
            st.pyplot(plot_convergence(hist, f"{algo} - Ordonnancement"))
        
        with tab3:
            st.pyplot(plot_task_durations(durees))
            st.pyplot(plot_completion_times(best, durees))
            if stats:
                st.pyplot(plot_additional_stats(stats, algo))
        
        with tab4:
            st.subheader("Données par Itération")
            df_iterations = pd.DataFrame(iterations_data)
            
            # Afficher un aperçu des données
            st.write(f"Nombre total d'itérations: {len(df_iterations)}")
            
            # Options d'affichage
            show_all = st.checkbox("Afficher toutes les itérations", value=False)
            iterations_to_show = len(df_iterations) if show_all else min(50, len(df_iterations))
            
            st.dataframe(df_iterations.head(iterations_to_show), use_container_width=True)
            
            # Téléchargement des données
            csv = df_iterations.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les données CSV",
                data=csv,
                file_name=f"ordonnancement_{algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab5:
            st.subheader("Résumé de l'Exécution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Paramètres de l'algorithme:**")
                st.write(f"- Algorithme: {algo}")
                if algo in ["GA Rang", "GA Roulette"]:
                    st.write(f"- Taille population: {pop_size}")
                    st.write(f"- Générations: {generations}")
                    st.write(f"- Taux croisement: {taux_croisement}")
                    st.write(f"- Taux mutation: {taux_mutation}")
                    st.write(f"- Méthode croisement: {methode_croisement}")
                    st.write(f"- Méthode mutation: {methode_mutation}")
                    st.write(f"- Élitisme: {elitisme}")
                elif algo == "Recuit Simulé":
                    st.write(f"- Température initiale: {T0}")
                    st.write(f"- Température min: {Tmin}")
                    st.write(f"- Alpha: {alpha}")
                    st.write(f"- Itérations par T: {iter_par_T}")
                elif algo == "Tabu":
                    st.write(f"- Max itérations: {max_iter}")
                    st.write(f"- Tenure: {tenure}")
            
            with col2:
                st.write("**Résultats:**")
                st.write(f"- Meilleure valeur: {val:.2f}")
                st.write(f"- Meilleur ordre: {best}")
                st.write(f"- Temps d'exécution: {execution_time:.2f}s")
                st.write(f"- Itérations totales: {len(hist)}")
                st.write(f"- Amélioration: {improvement:.1f}%")

elif category == "Voyageur":
    st.header("🗺️ Voyageur de Commerce (TSP)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration du Problème")
        nb_villes = st.slider("Nombre de villes", 5, 50, 10)
        largeur = st.slider("Largeur de la carte", 50, 200, 100)
        hauteur = st.slider("Hauteur de la carte", 50, 200, 100)
        
        if st.button("Générer de nouvelles villes"):
            st.session_state.villes = generer_villes(nb_villes, largeur, hauteur)
    
    with col2:
        st.subheader("Configuration de l'Algorithme")
        algo = st.selectbox("Algorithme", ["GA Rang", "GA Roulette", "Recuit Simulé", "Tabu"])
    
    # Génération des villes
    if 'villes' not in st.session_state:
        st.session_state.villes = generer_villes(nb_villes, largeur, hauteur)
    
    villes = st.session_state.villes
    matrice = construire_matrice_distances(villes)
    
    # Paramètres spécifiques aux algorithmes
    if algo in ["GA Rang", "GA Roulette"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            taille_pop = st.slider("Taille population", 20, 200, 100)
            generations = st.slider("Générations", 50, 1000, 300)
        with col2:
            taux_croisement = st.slider("Taux croisement", 0.0, 1.0, 0.8)
            taux_mutation = st.slider("Taux mutation", 0.0, 1.0, 0.2)
        with col3:
            methode_croisement = st.selectbox("Méthode croisement", ["simple", "double", "uniforme", "barycentrique"])
            methode_mutation = st.selectbox("Méthode mutation", ["swap", "inversion", "scramble", "decalage"])
            elitisme = st.checkbox("Élitisme", True)
            seed = st.number_input("Seed", value=42)
    
    elif algo == "Recuit Simulé":
        col1, col2 = st.columns(2)
        with col1:
            T0 = st.slider("Température initiale", 100, 5000, 1000)
            Tmin = st.slider("Température min", 0.001, 1.0, 0.001)
        with col2:
            alpha = st.slider("Alpha", 0.9, 0.999, 0.995)
            iterations = st.slider("Itérations par T", 10, 1000, 500)
    
    elif algo == "Tabu":
        col1, col2 = st.columns(2)
        with col1:
            max_iter = st.slider("Max itérations", 100, 3000, 1000)
            tabu_tenure = st.slider("Tenure", 5, 50, 15)
        with col2:
            nb_voisins_sample = st.slider("Échantillon voisins", 50, 1000, 200)
            seed = st.number_input("Seed", value=42)

    if st.button("🚀 Lancer l'Optimisation TSP", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initialisation de l'algorithme TSP...")
        
        start_time = time.time()
        
        # Exécution de l'algorithme sélectionné
        if algo == "GA Rang":
            best, val, hist, iterations_data, stats = algo_genetique_rang_tsp(
                matrice, taille_pop, generations, taux_croisement, taux_mutation, 
                methode_croisement, methode_mutation, elitisme, seed
            )
        elif algo == "GA Roulette":
            best, val, hist, iterations_data, stats = algo_genetique_roulette_tsp(
                matrice, taille_pop, generations, taux_croisement, taux_mutation,
                methode_croisement, methode_mutation, elitisme, seed
            )
        elif algo == "Recuit Simulé":
            best, val, hist, iterations_data, stats = recuit_simule_tsp(villes, T0, Tmin, alpha, iterations)
        elif algo == "Tabu":
            best, val, hist, iterations_data, stats = tabu_search_tsp(matrice, max_iter, tabu_tenure, nb_voisins_sample, seed)
        
        execution_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("Optimisation TSP terminée!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats
        st.success("✅ Optimisation TSP terminée avec succès!")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Distance optimale", f"{val:.2f}")
        with col2:
            st.metric("Temps d'exécution", f"{execution_time:.2f}s")
        with col3:
            st.metric("Nombre d'itérations", len(hist))
        with col4:
            st.metric("Nombre de villes", nb_villes)
        
        st.write(f"**Meilleure tournée trouvée:** {best}")
        
        # Visualisations
        st.subheader("📈 Visualisations TSP")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Carte", "Convergence", "Distances", "Données Détaillées", "Résumé"])
        
        with tab1:
            st.pyplot(plot_tsp(villes, best))
        
        with tab2:
            st.pyplot(plot_convergence(hist, f"{algo} - TSP"))
        
        with tab3:
            st.pyplot(plot_tour_distances(villes, best))
            if stats:
                st.pyplot(plot_additional_stats(stats, algo))
        
        with tab4:
            st.subheader("Données par Itération")
            df_iterations = pd.DataFrame(iterations_data)
            
            st.write(f"Nombre total d'itérations: {len(df_iterations)}")
            
            show_all = st.checkbox("Afficher toutes les itérations", value=False, key="tsp_show_all")
            iterations_to_show = len(df_iterations) if show_all else min(50, len(df_iterations))
            
            st.dataframe(df_iterations.head(iterations_to_show), use_container_width=True)
            
            # Téléchargement des données
            csv = df_iterations.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les données CSV",
                data=csv,
                file_name=f"tsp_{algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab5:
            st.subheader("Résumé de l'Exécution TSP")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Paramètres du problème:**")
                st.write(f"- Nombre de villes: {nb_villes}")
                st.write(f"- Dimensions de la carte: {largeur} × {hauteur}")
                st.write("**Paramètres de l'algorithme:**")
                st.write(f"- Algorithme: {algo}")
                if algo in ["GA Rang", "GA Roulette"]:
                    st.write(f"- Taille population: {taille_pop}")
                    st.write(f"- Générations: {generations}")
                    st.write(f"- Taux croisement: {taux_croisement}")
                    st.write(f"- Taux mutation: {taux_mutation}")
                    st.write(f"- Méthode croisement: {methode_croisement}")
                    st.write(f"- Méthode mutation: {methode_mutation}")
                    st.write(f"- Élitisme: {elitisme}")
                elif algo == "Recuit Simulé":
                    st.write(f"- Température initiale: {T0}")
                    st.write(f"- Température min: {Tmin}")
                    st.write(f"- Alpha: {alpha}")
                    st.write(f"- Itérations par T: {iterations}")
                elif algo == "Tabu":
                    st.write(f"- Max itérations: {max_iter}")
                    st.write(f"- Tenure: {tabu_tenure}")
                    st.write(f"- Échantillon voisins: {nb_voisins_sample}")
            
            with col2:
                st.write("**Résultats:**")
                st.write(f"- Distance optimale: {val:.2f}")
                st.write(f"- Meilleure tournée: {best}")
                st.write(f"- Temps d'exécution: {execution_time:.2f}s")
                st.write(f"- Itérations totales: {len(hist)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### À propos")
st.sidebar.info(
    "Cette application compare les performances de différents algorithmes "
    "metaheuristiques sur des problèmes d'optimisation classiques."
)