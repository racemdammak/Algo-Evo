import random
from collections import deque

def calculer_distance_totale(solution, matrice_distances):
    """Calcule la distance totale d'une tournée"""
    distance_totale = 0
    for i in range(len(solution) - 1):
        distance_totale += matrice_distances[solution[i]][solution[i + 1]]
    distance_totale += matrice_distances[solution[-1]][solution[0]]  # Retour au point de départ
    return distance_totale


def generer_voisins(solution):
    """Génère les voisins en échangeant deux villes"""
    voisins = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            voisin = solution[:]
            voisin[i], voisin[j] = voisin[j], voisin[i]
            voisins.append(voisin)
    return voisins


def tabu_search(matrice_distances, nombre_iterations, taille_tabu):
    nombre_villes = len(matrice_distances)

    # Solution initiale
    solution_actuelle = list(range(nombre_villes))
    random.shuffle(solution_actuelle)

    meilleure_solution = solution_actuelle[:]
    meilleure_distance = calculer_distance_totale(solution_actuelle, matrice_distances)

    print("Solution initiale :", solution_actuelle)
    print("Distance initiale :", meilleure_distance)
    print("-" * 50)

    # Liste tabou
    tabu_list = deque(maxlen=taille_tabu)

    for iteration in range(nombre_iterations):
        voisins = generer_voisins(solution_actuelle)
        voisins = [v for v in voisins if v not in tabu_list]

        if not voisins:
            print("Aucun voisin disponible (liste tabou trop restrictive). Stop.")
            break

        # Meilleur voisin
        solution_actuelle = min(voisins, key=lambda v: calculer_distance_totale(v, matrice_distances)) 
        distance_actuelle = calculer_distance_totale(solution_actuelle, matrice_distances)

        tabu_list.append(solution_actuelle)

        print(f"Iteration {iteration+1}/{nombre_iterations}")
        print("Solution courante :", solution_actuelle)
        print("Distance courante :", distance_actuelle)

        if distance_actuelle < meilleure_distance:
            meilleure_solution = solution_actuelle[:]
            meilleure_distance = distance_actuelle
            print("✨ Nouvelle meilleure solution trouvée !")
            print("Meilleure solution :", meilleure_solution)
            print("Distance :", meilleure_distance)
        print('itération n°', iteration + 1)
        print("-" * 50)

    return meilleure_solution, meilleure_distance


# --------------------------------------------------------
# MATRICE DE DISTANCES
# --------------------------------------------------------
matrice_distances = [
    [0, 2, 1, 12, 5, 7, 6, 5],
    [2, 0, 10, 4, 7, 13, 2, 3],
    [1, 10, 0, 2, 4, 15, 3, 7],
    [12, 4, 2, 0, 1, 6, 9, 3],
    [5, 7, 4, 1, 0, 7, 2, 10],
    [7, 13, 15, 6, 7, 0, 2, 1],
    [6, 2, 3, 9, 2, 2, 0, 15],
    [5, 3, 7, 3, 10, 1, 15, 0]
]

# --------------------------------------------------------
# PARAMÈTRES
# --------------------------------------------------------
nombre_iterations = 100
taille_tabu = 20

# --------------------------------------------------------
# EXÉCUTION
# --------------------------------------------------------
meilleure_solution, meilleure_distance = tabu_search(
    matrice_distances, nombre_iterations, taille_tabu
)

print("\n✅ Résultat final")
print("Meilleure solution trouvée :", meilleure_solution)
print("Distance minimale :", meilleure_distance)
