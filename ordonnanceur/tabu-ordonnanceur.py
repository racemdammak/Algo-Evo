import random

def calculer_sum_completion(ordre, durees):
    total, cumul = 0, 0
    for t in ordre:
        cumul += durees[t]
        total += cumul
    return total

def generer_voisins(ordre):
    voisins = []
    for i in range(len(ordre)):
        for j in range(i + 1, len(ordre)):
            v = ordre[:]
            v[i], v[j] = v[j], v[i]
            voisins.append((v, (i, j)))
    return voisins

def recherche_tabou_ordonnancement(durees, max_iter=300, tenure=10):
    n = len(durees)
    courant = list(range(n))
    random.shuffle(courant)
    meilleur = courant[:]
    meilleur_cout = calculer_sum_completion(meilleur, durees)
    courant_cout = meilleur_cout
    tabou = {}

    print("\n=== Recherche Tabou - Ordonnancement ===")
    print(f"Ordre initial : {courant}, Coût initial : {courant_cout}")

    for it in range(1, max_iter + 1):
        voisins = generer_voisins(courant)
        meilleur_voisin, meilleur_voisin_cout, move = None, float("inf"), None

        for v, m in voisins:
            val = calculer_sum_completion(v, durees)
            if (m not in tabou or val < meilleur_cout) and val < meilleur_voisin_cout:
                meilleur_voisin, meilleur_voisin_cout, move = v, val, m

        courant, courant_cout = meilleur_voisin, meilleur_voisin_cout
        tabou[move] = tenure
        for m in list(tabou.keys()):
            tabou[m] -= 1
            if tabou[m] <= 0:
                del tabou[m]

        if courant_cout < meilleur_cout:
            meilleur, meilleur_cout = courant[:], courant_cout
            print(f"→ Itération {it} | Nouvelle meilleure solution : {meilleur_cout}")

    print("\nRésultat final :")
    print("Ordre optimal :", meilleur)
    print("Somme des Cᵢ :", meilleur_cout)
    return meilleur, meilleur_cout


if __name__ == "__main__":
    durees = [12, 7, 4, 10, 8, 5, 9, 6, 11, 3]
    recherche_tabou_ordonnancement(durees)
