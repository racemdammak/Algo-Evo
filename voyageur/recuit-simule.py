import random
import math
import matplotlib.pyplot as plt

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
    # Échange de deux villes avec des indices aléatoires
    i, j = random.sample(range(len(solution)), 2)
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

def plot_route(villes, solution, title="Route"):
    """Affiche la route sur un graphique"""
    x = [villes[i][0] for i in solution] + [villes[solution[0]][0]]
    y = [villes[i][1] for i in solution] + [villes[solution[0]][1]]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-', markersize=8, linewidth=2, color='blue')
    plt.scatter([v[0] for v in villes], [v[1] for v in villes], color='red', s=100, zorder=5)
    for i, ville in enumerate(villes):
        plt.text(ville[0], ville[1], f'{i}', fontsize=12, ha='center', va='center', color='white', weight='bold')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_evolution(evolution):
    """Affiche l'évolution de la meilleure distance"""
    plt.figure(figsize=(10, 6))
    plt.plot(evolution, color='green', linewidth=2)
    plt.title("Évolution de la meilleure distance au cours du recuit simulé")
    plt.xlabel("Itération")
    plt.ylabel("Distance totale")
    plt.grid(True)
    plt.show()

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

    # Visualisations
    plot_route(villes, meilleure_solution, "Meilleure route trouvée")
    plot_evolution(evolution)
