import random
import math

# -------------------------
# Fonctions utilitaires
# -------------------------

def generer_villes(nb_villes, largeur=100, hauteur=100):
    """Génère des coordonnées (x,y) aléatoires pour nb_villes."""
    return [(random.uniform(0, largeur), random.uniform(0, hauteur)) for _ in range(nb_villes)]

def distance_euclidienne(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def construire_matrice_distances(villes):
    """Construit la matrice complète des distances euclidiennes."""
    n = len(villes)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = distance_euclidienne(villes[i], villes[j])
    return mat

def calculer_distance_totale_perm(sol, matrice_distances):
    """Distance totale d'une tournée (permutation de villes)."""
    d = 0.0
    n = len(sol)
    for i in range(n - 1):
        d += matrice_distances[sol[i]][sol[i+1]]
    d += matrice_distances[sol[-1]][sol[0]]
    return d

# -------------------------
# Génération de voisins
# -------------------------

def generer_voisin_swap(solution, i, j):
    """Retourne une nouvelle permutation obtenue en échangeant positions i et j."""
    voisin = solution[:]
    voisin[i], voisin[j] = voisin[j], voisin[i]
    return voisin

def enumerer_voisins_swap(solution):
    """Génère tous les voisins obtenus par swap (i<j)."""
    n = len(solution)
    for i in range(n-1):
        for j in range(i+1, n):
            yield (i, j, generer_voisin_swap(solution, i, j))

# -------------------------
# Recherche Tabou
# -------------------------

def tabu_search_tsp(matrice_distances,
                    max_iter=1000,
                    tabu_tenure=15,
                    nb_voisins_sample=200,
                    initial_solution=None,
                    seed=None):
    """
    Recherche Tabou pour le TSP.

    Paramètres :
    - matrice_distances : matrice carrée des distances
    - max_iter : nombre d'itérations de la recherche
    - tabu_tenure : durée (itérations) pendant laquelle une move est taboue
    - nb_voisins_sample : si None => on énumère tout le voisinage (O(n^2)),
                          sinon on échantillonne ce nombre de voisins aléatoires (plus rapide)
    - initial_solution : permutation initiale (si None -> aléatoire)
    - seed : graine aléatoire (optionnel pour reproductibilité)

    Retour :
    - meilleure_solution, meilleure_distance, historique_best, historique_current
    """

    if seed is not None:
        random.seed(seed)

    n = len(matrice_distances)

    # solution initiale
    if initial_solution is None:
        current = list(range(n))
        random.shuffle(current)
    else:
        current = initial_solution[:]

    current_dist = calculer_distance_totale_perm(current, matrice_distances)
    best = current[:]
    best_dist = current_dist

    # tableau/tabou : on stocke les moves taboues comme tuple (i,j) avec j>i
    # et une valeur entière (tenure restante)
    tabu_list = {}  # {(i,j): remaining_tenure}

    historique_best = []
    historique_current = []

    print("=== DÉMARRAGE TABU SEARCH ===")
    print(f"Solution initiale : {current}, distance = {current_dist:.3f}")
    print(f"max_iter={max_iter}, tabu_tenure={tabu_tenure}, nb_voisins_sample={nb_voisins_sample}\n")

    for it in range(1, max_iter + 1):
        # construire ensemble candidat de voisins
        candidats = []

        if nb_voisins_sample is None:
            # énumérer tout le voisinage (coûteux)
            for i in range(n-1):
                for j in range(i+1, n):
                    voisin = generer_voisin_swap(current, i, j)
                    d = calculer_distance_totale_perm(voisin, matrice_distances)
                    candidats.append((i, j, voisin, d))
        else:
            # échantillon aléatoire de swaps
            tried = set()
            sample_size = min(nb_voisins_sample, n*(n-1)//2)
            while len(tried) < sample_size:
                i, j = sorted(random.sample(range(n), 2))
                if (i,j) in tried:
                    continue
                tried.add((i,j))
                voisin = generer_voisin_swap(current, i, j)
                d = calculer_distance_totale_perm(voisin, matrice_distances)
                candidats.append((i, j, voisin, d))

        # trier candidats par meilleure distance (ascendant)
        candidats.sort(key=lambda x: x[3])

        # choisir le meilleur candidat non tabou ou tabou si aspiration (améliore best global)
        chosen = None
        for (i, j, voisin, d) in candidats:
            move = (i, j)
            is_tabou = move in tabu_list
            aspiration = (d < best_dist)
            if (not is_tabou) or aspiration:
                chosen = (i, j, voisin, d)
                break

        if chosen is None:
            # cas improbable : on prend le meilleur malgré tout
            chosen = candidats[0]

        i_chosen, j_chosen, next_sol, next_dist = chosen

        # appliquer le move
        current = next_sol
        current_dist = next_dist

        # mettre à jour tabou : interdire l'inverse du move (ou le move lui-même)
        # convention : on stocke (min(i,j), max(i,j))
        key = (min(i_chosen, j_chosen), max(i_chosen, j_chosen))
        tabu_list[key] = tabu_tenure

        # décrémenter les durées tabou et supprimer expirées
        expired = []
        for k in list(tabu_list.keys()):
            tabu_list[k] -= 1
            if tabu_list[k] <= 0:
                expired.append(k)
        for k in expired:
            del tabu_list[k]

        # mise à jour du meilleur global
        if current_dist < best_dist:
            best = current[:]
            best_dist = current_dist
            print(f"Iter {it}: Nouveau meilleur global = {best_dist:.3f} (move swap {i_chosen}-{j_chosen})")

        # historique
        historique_best.append(best_dist)
        historique_current.append(current_dist)
        
    return best, best_dist, historique_best, historique_current

# -------------------------
# Exemple d'utilisation
# -------------------------
if __name__ == "__main__":
    # Générer 10 villes et matrice de distances
    villes = generer_villes(10, largeur=100, hauteur=100)
    matrice = construire_matrice_distances(villes)

    # Lancer Tabu Search
    best_sol, best_d, hist_best, hist_cur = tabu_search_tsp(
        matrice_distances=matrice,
        max_iter=1000,
        tabu_tenure=20,
        nb_voisins_sample=200,  # échantillon de voisins (accélère la recherche)
        seed=42
    )

    print("\n=== RÉSULTAT FINAL TABU SEARCH ===")
    print("Meilleure tournée :", best_sol)
    print("Distance minimale :", best_d)
