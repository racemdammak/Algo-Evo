import random
import math

# ============================
# 🔹 1. Fonctions utilitaires
# ============================

def generer_villes(nb_villes, largeur=100, hauteur=100):
    """Génère aléatoirement des villes dans un plan 2D"""
    return [(random.uniform(0, largeur), random.uniform(0, hauteur)) for _ in range(nb_villes)]

def distance(v1, v2):
    """Calcule la distance euclidienne entre deux villes"""
    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def calculer_distance_totale(solution, villes):
    """Calcule la distance totale parcourue selon l'ordre donné"""
    dist = 0
    for i in range(len(solution) - 1):
        dist += distance(villes[solution[i]], villes[solution[i + 1]])
    # retour à la ville de départ
    dist += distance(villes[solution[-1]], villes[solution[0]])
    return dist

def generer_voisin(solution):
    """Crée un voisin en échangeant deux positions aléatoires"""
    voisin = solution[:]
    i, j = random.sample(range(len(solution)), 2)
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

# ============================
# 🔹 2. Algorithme du Recuit Simulé
# ============================

def recuit_simule(villes, T0=1000, Tmin=1e-3, alpha=0.995, iterations=500):
    """
    Optimisation du TSP par recuit simulé.
    
    Paramètres :
    - villes : liste des coordonnées (x, y)
    - T0 : température initiale
    - Tmin : température minimale (critère d'arrêt)
    - alpha : taux de refroidissement
    - iterations : nombre d’itérations par palier de température
    """
    
    n = len(villes)
    
    # Solution initiale aléatoire
    solution = list(range(n))
    random.shuffle(solution)
    
    meilleure_solution = solution[:]
    meilleure_distance = calculer_distance_totale(solution, villes)
    distance_actuelle = meilleure_distance
    
    T = T0
    evolution = []  # suivi des meilleures distances pour analyse
    
    print("=== DÉMARRAGE DU RECUIT SIMULÉ ===")
    print(f"Nombre de villes : {n}")
    print(f"Température initiale : {T0}")
    print(f"Solution initiale : {solution}")
    print(f"Distance initiale : {meilleure_distance:.2f}\n")
    
    # Boucle principale du recuit simulé
    while T > Tmin:
        for _ in range(iterations):
            voisin = generer_voisin(solution)
            distance_voisin = calculer_distance_totale(voisin, villes)
            delta = distance_voisin - distance_actuelle

            # 🔸 Critère d'acceptation
            if delta < 0 or random.random() < math.exp(-delta / T):
                solution = voisin
                distance_actuelle = distance_voisin

                # 🔹 Mise à jour du meilleur
                if distance_actuelle < meilleure_distance:
                    meilleure_solution = solution[:]
                    meilleure_distance = distance_actuelle
                    print(f"🔥 Nouvelle meilleure distance : {meilleure_distance:.2f} (T={T:.3f})")
            
            evolution.append(meilleure_distance)
        
        # 🔻 Refroidissement
        T *= alpha
    
    print("\n=== FIN DU RECUIT SIMULÉ ===")
    print(f"Meilleure distance trouvée : {meilleure_distance:.2f}")
    print(f"Meilleure tournée : {meilleure_solution}")
    
    return meilleure_solution, meilleure_distance, evolution


# ============================
# 🔹 3. Exemple d'utilisation
# ============================

if __name__ == "__main__":
    villes = generer_villes(10)  # 🔹 10 villes aléatoires
    meilleure_solution, meilleure_distance, evolution = recuit_simule(villes)
