import random

def selection_par_roulette(population, valeurs_fitness, nb_selection):

    somme_fitness = sum(valeurs_fitness)
    probabilites = []
    for f in valeurs_fitness:

        probabilites.append(f / somme_fitness)
    print("🏅 Probabilités de sélection :", probabilites)
    selectionnes = []
    for _ in range(nb_selection):
        r = random.random()
        cumul = 0

        for i, p in enumerate(probabilites):
            cumul += p
            if r <= cumul:
                print(f"🎯 Valeur aléatoire : {r:.4f}, sélection de l'individu : {population[i]} avec probabilité {p:.4f}")
                selectionnes.append(population[i])
                break
    return selectionnes

population = ["A", "B", "C", "D"]
valeurs_fitness = [10, 5, 1, 0.5]
resultat = selection_par_roulette(population, valeurs_fitness, 2)

print("Les deux parents sélectionnés :", resultat)
