import random
import math

def generer_villes(nbr_villes):
    villes = {}
    for i in range(nbr_villes):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        villes[i] = (x, y)
    return villes

def distance(ville1, ville2):
    return (ville1[0]-ville2[0])**2 + (ville1[1]-ville2[1])**2

def calculer_distance(solution, villes):
    dist = 0
    for i in range(len(solution) - 1):
       dist += distance(villes[solution[i]], villes[solution[i + 1]])
    dist += distance(villes[solution[-1]], villes[solution[0]])
    return dist

def generer_voision(solution):
    voisin = solution[:]
    i, j = random.sample(range(len(solution)), 2)
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

def recuit_simule(villes, T0=1000, Tmin=1, alpha=0.995, iterations=500):
    #initialisation
    n = len(villes)
    solution_actuelle = list(range(n))
    random.shuffle(solution_actuelle)

    meilleure_solution = solution_actuelle[:]
    meilleure_distance = calculer_distance(solution_actuelle, villes)
    distance_actuelle = meilleure_distance
    T = T0
    print("=== DÉMARRAGE DU RECUIT SIMULÉ ===")
    print(f"Nombre de villes : {n}")
    print(f"Température initiale : {T0}")
    print(f"Solution initiale : {solution_actuelle}")
    print(f"Distance initiale : {meilleure_distance:.2f}\n")
    while T > Tmin:
        for iteration in range(iterations):
            #generation voisin
            voisin = generer_voision(solution_actuelle)
            distance_voisin = calculer_distance(voisin, villes)
            delta = distance_voisin - distance_actuelle

            #critere d'acceptation
            if delta < 0 or random.random() < math.exp(-delta / T):
                solution_actuelle = voisin
                distance_actuelle = distance_voisin
                if distance_actuelle < meilleure_distance:
                    meilleure_solution = solution_actuelle[:]
                    meilleure_distance = distance_actuelle
        T *= alpha

    return meilleure_solution, meilleure_distance



nombre_villes = 10
villes = generer_villes(nombre_villes)
meilleure_solution, meilleure_distance = recuit_simule(villes, T0=1000, Tmin=1, alpha=0.995, iterations=500)
print("Meilleure solution trouvée :", meilleure_solution)
print("Distance de la meilleure solution :", meilleure_distance)