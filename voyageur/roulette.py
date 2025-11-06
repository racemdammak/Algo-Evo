import random
import math

# ====================================================
# 🔹 Fonctions utilitaires
# ====================================================

def generer_villes(nb_villes, largeur=100, hauteur=100):
    """Génère aléatoirement des villes dans un plan 2D"""
    return [(random.uniform(0, largeur), random.uniform(0, hauteur)) for _ in range(nb_villes)]

def distance(v1, v2):
    """Distance euclidienne entre deux villes"""
    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def construire_matrice_distances(villes):
    """Construit une matrice de distances à partir des coordonnées"""
    n = len(villes)
    return [[distance(villes[i], villes[j]) for j in range(n)] for i in range(n)]

def calculer_distance_totale(solution, matrice_distances):
    """Calcule la distance totale d'une tournée"""
    dist = 0
    for i in range(len(solution) - 1):
        dist += matrice_distances[solution[i]][solution[i + 1]]
    dist += matrice_distances[solution[-1]][solution[0]]
    return dist

def creer_population_initiale(taille, n):
    """Crée une population initiale aléatoire"""
    population = []
    for _ in range(taille):
        ind = list(range(n))
        random.shuffle(ind)
        population.append(ind)
    return population

def fitness(ind, matrice_distances):
    """Calcule le fitness (inverse de la distance totale)"""
    d = calculer_distance_totale(ind, matrice_distances)
    return 1 / d if d > 0 else 0


# ====================================================
# 🔹 Sélection par roulette
# ====================================================

def selection_roulette(pop, fitnesses):
    total = sum(fitnesses)
    probs = [f / total for f in fitnesses]
    r = random.random()
    cumul = 0
    for i, p in enumerate(probs):
        cumul += p
        if r <= cumul:
            return pop[i][:]
    return pop[-1][:]


# ====================================================
# 🔹 Croisement et mutations
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
# 🔹 Algorithme génétique — Sélection par roulette
# ====================================================

def algo_genetique_roulette(matrice_distances, taille_pop=100, generations=300,
                            taux_croisement=0.8, taux_mutation=0.2):
    
    n = len(matrice_distances)
    population = creer_population_initiale(taille_pop, n)
    meilleur = None
    meilleure_distance = float('inf')

    for g in range(generations):
        fitnesses = [fitness(ind, matrice_distances) for ind in population]
        nouvelle_population = []

        for _ in range(taille_pop // 2):
            p1 = selection_roulette(population, fitnesses)
            p2 = selection_roulette(population, fitnesses)

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

    meilleur, meilleure_distance = algo_genetique_roulette(matrice, generations=200)
    
    print("\n=== Résultat Algorithme Génétique (Sélection par Roulette) ===")
    print(f"Meilleure tournée trouvée : {meilleur}")
    print(f"Distance totale : {meilleure_distance:.2f}")
