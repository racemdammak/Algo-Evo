import random
import math

# ====================================================
# 🔹 Fonctions utilitaires (identiques à la version précédente)
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

def creer_population_initiale(taille, n):
    population = []
    for _ in range(taille):
        ind = list(range(n))
        random.shuffle(ind)
        population.append(ind)
    return population

def fitness(ind, matrice_distances):
    d = calculer_distance_totale(ind, matrice_distances)
    return 1 / d if d > 0 else 0


# ====================================================
# 🔹 Sélection par rang
# ====================================================

def selection_rang(pop, fitnesses):
    ranked = sorted(zip(pop, fitnesses), key=lambda x: x[1])  # du plus faible au plus fort
    n = len(pop)
    probs = [(i + 1) / sum(range(1, n + 1)) for i in range(n)]
    r = random.random()
    cumul = 0
    for i, p in enumerate(probs):
        cumul += p
        if r <= cumul:
            return ranked[i][0][:]
    return ranked[-1][0][:]


# ====================================================
# 🔹 Croisement et mutation (identiques)
# ====================================================

def croisement_double(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    segment = p1[a:b]
    reste = [x for x in p2 if x not in segment]
    return reste[:a] + segment + reste[a:]

def mutation_inversion(ind):
    a, b = sorted(random.sample(range(len(ind)), 2))
    ind[a:b] = reversed(ind[a:b])
    return ind


# ====================================================
# 🔹 Algorithme génétique — Sélection par rang
# ====================================================

def algo_genetique_rang(matrice_distances, taille_pop=100, generations=300,
                        taux_croisement=0.8, taux_mutation=0.2):
    
    n = len(matrice_distances)
    population = creer_population_initiale(taille_pop, n)
    meilleur = None
    meilleure_distance = float('inf')

    for g in range(generations):
        fitnesses = [fitness(ind, matrice_distances) for ind in population]
        nouvelle_population = []

        for _ in range(taille_pop // 2):
            p1 = selection_rang(population, fitnesses)
            p2 = selection_rang(population, fitnesses)

            if random.random() < taux_croisement:
                enfant = croisement_double(p1, p2)
            else:
                enfant = p1[:]

            if random.random() < taux_mutation:
                enfant = mutation_inversion(enfant)

            nouvelle_population.append(enfant)

        population = nouvelle_population

        for ind in population:
            d = calculer_distance_totale(ind, matrice_distances)
            if d < meilleure_distance:
                meilleure_distance = d
                meilleur = ind[:]
                print(f"Génération {g}: Nouvelle meilleure distance = {meilleure_distance:.2f}")

    return meilleur, meilleure_distance


# ====================================================
# 🔹 Exemple réaliste : 10 villes aléatoires
# ====================================================

if __name__ == "__main__":
    villes = generer_villes(10)
    matrice = construire_matrice_distances(villes)

    meilleur, meilleure_distance = algo_genetique_rang(matrice, generations=200)
    
    print("\n=== Résultat Algorithme Génétique (Sélection par Rang) ===")
    print(f"Meilleure tournée trouvée : {meilleur}")
    print(f"Distance totale : {meilleure_distance:.2f}")