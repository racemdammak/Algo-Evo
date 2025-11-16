import random
import math
import matplotlib.pyplot as plt

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
def croisement_simple(p1, p2):
    n = len(p1)
    a = random.randint(0, n - 1)

    segment1 = p1[:a]
    reste1 = [x for x in p2 if x not in segment1]
    enfant1 = segment1 + reste1

    segment2 = p2[:a]
    reste2 = [x for x in p1 if x not in segment2]
    enfant2 = segment2 + reste2

    return enfant1, enfant2


def croisement_double(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    segment1 = p1[a:b]
    reste1 = [x for x in p2 if x not in segment1]
    enfant1 = reste1[:a] + segment1 + reste1[a:]

    segment2 = p2[a:b]
    reste2 = [x for x in p1 if x not in segment2]
    enfant2 = reste2[:a] + segment2 + reste2[a:]

    return enfant1, enfant2


def croisement_uniforme(p1, p2):
    n = len(p1)
    mask = [random.randint(0, 1) for _ in range(n)]

    enfant1 = []
    enfant2 = []

    for i in range(n):
        if mask[i] == 1:
            enfant1.append(p1[i])
            enfant2.append(p2[i])
        else:
            enfant1.append(p2[i])
            enfant2.append(p1[i])

    return enfant1, enfant2


def mutation_inversion(ind):
    a, b = sorted(random.sample(range(len(ind)), 2))
    ind[a:b] = reversed(ind[a:b])
    return ind

def mutation_swap(ind):
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]
    return ind

def mutation_scramble(ind):
    a, b = sorted(random.sample(range(len(ind)), 2))
    segment = ind[a:b]
    random.shuffle(segment)
    ind[a:b] = segment
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
    historique_distances = []

    for g in range(generations):
        fitnesses = [fitness(ind, matrice_distances) for ind in population]
        nouvelle_population = []

        for _ in range(taille_pop):
            p1 = selection_roulette(population, fitnesses)
            p2 = selection_roulette(population, fitnesses)

            if random.random() < taux_croisement:
                enfant1, enfant2 = croisement_double(p1, p2)
            else:
                enfant1 = p1[:]
                enfant2 = p2[:]

            if random.random() < taux_mutation:
                enfant1 = mutation_inversion(enfant1)
                enfant2 = mutation_inversion(enfant2)

            nouvelle_population.append(enfant1)
            nouvelle_population.append(enfant2)

        population = nouvelle_population

        for ind in population:
            d = calculer_distance_totale(ind, matrice_distances)
            if d < meilleure_distance:
                meilleure_distance = d
                meilleur = ind[:]
                print(f"Génération {g}: Nouvelle meilleure distance = {meilleure_distance:.2f}")

        historique_distances.append(meilleure_distance)
    return meilleur, meilleure_distance, historique_distances


# ====================================================
# 🔹 Exécution + Visualisation
# ====================================================

if __name__ == "__main__":
    # Génération des villes
    villes = generer_villes(10)
    matrice = construire_matrice_distances(villes)
    print("=== Villes générées ===")
    for i, (x, y) in enumerate(villes):
        print(f"Ville {i}: ({x:.2f}, {y:.2f})")

    # Algorithme génétique
    meilleur, meilleure_distance, historique = algo_genetique_roulette(matrice, generations=200)
    
    print("\n=== Résultat Algorithme Génétique ===")
    print(f"Meilleure tournée trouvée : {meilleur}")
    print(f"Distance totale : {meilleure_distance:.2f}")

    # -----------------------------------------
    # Visualisation 1 : Position des villes
    # -----------------------------------------
    x = [v[0] for v in villes]
    y = [v[1] for v in villes]

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, color="blue")

    for i, (xi, yi) in enumerate(villes):
        plt.text(xi + 1, yi + 1, str(i), fontsize=9)

    plt.title("Position des villes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    # -----------------------------------------
    # Visualisation 2 : Convergence
    # -----------------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(historique)
    plt.title("Évolution de la meilleure distance")
    plt.xlabel("Générations")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

    # -----------------------------------------
    # Visualisation 3 : Tournée finale
    # -----------------------------------------
    tour_x = [villes[i][0] for i in meilleur] + [villes[meilleur[0]][0]]
    tour_y = [villes[i][1] for i in meilleur] + [villes[meilleur[0]][1]]

    plt.figure(figsize=(6,6))
    plt.plot(tour_x, tour_y, marker="o")
    plt.title("Tournée finale obtenue")
    plt.xlabel("x")
    plt.ylabel("y")

    for i in meilleur:
        plt.text(villes[i][0] + 1, villes[i][1] + 1, str(i), fontsize=9)

    plt.grid(True)
    plt.show()
