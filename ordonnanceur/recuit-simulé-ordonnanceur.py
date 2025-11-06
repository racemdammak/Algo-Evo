import random
import math

def calculer_sum_completion(ordre, durees):
    total, cumul = 0, 0
    for t in ordre:
        cumul += durees[t]
        total += cumul
    return total

def generer_voisin(ordre):
    i, j = random.sample(range(len(ordre)), 2)
    voisin = ordre[:]
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

def recuit_simule_ordonnancement(durees, T0=1000, Tmin=1e-3, alpha=0.95, iter_par_T=100):
    n = len(durees)
    ordre = list(range(n))
    random.shuffle(ordre)
    cout = calculer_sum_completion(ordre, durees)
    meilleur, meilleur_cout = ordre[:], cout
    T = T0
    iteration = 0

    print("\n=== Recuit Simulé - Ordonnancement ===")
    print(f"Ordre initial : {ordre}, Coût initial : {cout}")

    while T > Tmin:
        for _ in range(iter_par_T):
            iteration += 1
            voisin = generer_voisin(ordre)
            cout_voisin = calculer_sum_completion(voisin, durees)
            delta = cout_voisin - cout
            if delta < 0 or random.random() < math.exp(-delta / T):
                ordre, cout = voisin, cout_voisin
                if cout < meilleur_cout:
                    meilleur, meilleur_cout = ordre[:], cout
                    print(f"→ Itération {iteration} | Nouvelle meilleure solution : {meilleur_cout}")
        T *= alpha

    print("\nRésultat final :")
    print("Ordre optimal :", meilleur)
    print("Somme des Cᵢ :", meilleur_cout)
    return meilleur, meilleur_cout


if __name__ == "__main__":
    # Exemple clair : 10 tâches avec durées variées
    durees = [12, 7, 4, 10, 8, 5, 9, 6, 11, 3]
    recuit_simule_ordonnancement(durees)
