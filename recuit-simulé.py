import math
import random
import matplotlib.pyplot as plt
import numpy as np

def simulated_annealing(f, x0, T0=1000, alpha=0.95, Tmin=5, max_iter = 50):
    x = x0
    fx = f(x)
    T = T0
    best_x, best_fx = x, fx

    iteration = 0
    resultats_fx = []
    meilleurs_fx = []
    resultats_T = []

    print(f"{'Itération':<10} {'Température':<15} {'x courant':<15} {'f(x)':<15}")

    while T > Tmin:
        for _ in range(max_iter):
            iteration += 1
            x_new = x + random.uniform(-1, 1) # x_new = 0.10364
            fx_new = f(x_new)  #f(x) = 1.99195
            delta = fx_new - fx
 
            # Critère d’acceptation
            if delta < 0 or random.random() < math.exp(-delta / T):
                x, fx = x_new, fx_new
                if fx < best_fx:
                    best_x, best_fx = x, fx

            resultats_fx.append(fx)
            meilleurs_fx.append(best_fx)
            resultats_T.append(T)

            print(f"n°:{_} {T:<15.5f} {x:<15.5f} {fx:<15.5f}")

        T *= alpha

    print("\n✅ Résultat final :")
    print(f"Meilleure solution trouvée : x = {best_x:.4f}, f(x) = {best_fx:.4f}")

    # 🔹 Visualisation de l'évolution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(resultats_fx, label="f(x) courant")
    plt.plot(meilleurs_fx, label="meilleur f(x)", linestyle='--')
    plt.xlabel("Itération")
    plt.ylabel("Valeur de f(x)")
    plt.title("Évolution de la fonction objectif")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(resultats_T, color='orange')
    plt.xlabel("Itération")
    plt.ylabel("Température")
    plt.title("Refroidissement au fil des itérations")

    plt.tight_layout()
    plt.show()

    # 🔹 Visualisation de la fonction et du minimum trouvé
    x_vals = np.linspace(-2, 2, 1000)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, label=r"$f(x) = x^2 + 4\sin(5x)$", color="blue")
    plt.scatter(best_x, best_fx, color="red", s=100, zorder=5, label="Minimum trouvé")

    plt.annotate(
        f"Minimum global\nx = {best_x:.3f}\nf(x) = {best_fx:.3f}",
        xy=(best_x, best_fx),
        xytext=(best_x + 0.5, best_fx + 1),
        arrowprops=dict(facecolor="red", shrink=0.05, width=1, headwidth=8),
        fontsize=11,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    plt.title("Visualisation du minimum trouvé par le recuit simulé", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_x, best_fx

f = lambda x: x**2 + 4*np.sin(5*x)

x0 = 2

best_x, best_fx = simulated_annealing(f, x0=x0, T0=1000, alpha=0.9, Tmin=5, max_iter=50)