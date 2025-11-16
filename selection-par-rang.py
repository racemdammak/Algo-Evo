import random

def selection_par_rang(population, valeurs_fitness, nb_selection):
    classes = sorted(zip(population, valeurs_fitness), key=lambda x: x[1])
    print("🏅 Classes triées par fitness :", classes)
    rangs = list(range(1, len(classes) + 1))

    somme_rangs = sum(rangs)

    probabilites = []
    for r in rangs:
        probabilites.append(r / somme_rangs)

    selectionnes = []
    for _ in range(nb_selection):
        r = random.random()
        cumul = 0
        for i, p in enumerate(probabilites):
            cumul += p
            if r <= cumul:
                selectionnes.append(classes[i][0])
                break
    return selectionnes


population = ["A","B", "C", "D"]
valeurs_fitness = [0.1, 0.3, 0.5, 0.9]

resultat = selection_par_rang(population, valeurs_fitness, 2)

print("🏅 Sélection par rang :", resultat)
