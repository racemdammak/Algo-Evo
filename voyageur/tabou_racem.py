import random
from collections import deque

def generer_matrice_distances(nombre_villes):
    matrice = []
    for i in range(nombre_villes):
        ligne = []
        for j in range(nombre_villes):
            if i == j:
                ligne.append(0)
            else:
                ligne.append(random.randint(1, 20))
        matrice.append(ligne)
    return matrice

def generer_voisins(solution):
    voisins = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            voisin = solution[:]
            voisin[i], voisin[j] = voisin[j], voisin[i]
            voisins.append(voisin)
    return voisins

def calculer_distance(solution, matrice_distances):
    dist = 0
    for i in range(len(solution) - 1):
        dist += matrice_distances[solution[i]][solution[i + 1]]
    dist += matrice_distances[solution[-1]][solution[0]]
    return dist

def recherche_tabou(matrice_distances, nombre_villes=10, taille_tabou=10, nombre_iterations=100):
    #initialisation
    solution_actuelle = list(range(nombre_villes))
    random.shuffle(solution_actuelle)
    
    meilleur_solution = solution_actuelle[:]
    meilleur_distance = calculer_distance(solution_actuelle, matrice_distances)

    tabu_list = deque(maxlen=taille_tabou)
    
    # 
    for iteration in range(nombre_iterations):
        voisins = generer_voisins(solution_actuelle)
        voisins = [v for v in voisins if v not in tabu_list]
        if not voisins:
            print("Aucun voisin disponible (liste tabou trop restrictive). Stop.")
            break
        # Meilleur voisin
        solution_actuelle = min(voisins, key=lambda v: calculer_distance(v, matrice_distances))
        distance_actuelle = calculer_distance(solution_actuelle, matrice_distances)
        tabu_list.append(solution_actuelle)
        if distance_actuelle < meilleur_distance:
            meilleur_solution = solution_actuelle[:]
            meilleur_distance = distance_actuelle
    print("=== RÉSULTATS DE LA RECHERCHE TABOU ===")
    print(f"Meilleure solution trouvée : {meilleur_solution}")
    print(f"Distance totale : {meilleur_distance:.2f}")


    


nombre_villes = 10
matrice_distance = generer_matrice_distances(nombre_villes)

recherche_tabou(matrice_distance, nombre_villes, taille_tabou=5, nombre_iterations=100)