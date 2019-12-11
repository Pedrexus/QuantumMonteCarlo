from math import exp, sqrt
from random import uniform

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

# d H-H = 1.4 bohr

# r1 = (x1, y1, z1)
# r2 = (x2, y2, z2)

A = (0, 0, 0)
B = (1.4, 0, 0)

ZA = ZB = 1

dist = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))


def psi(r1, r2, k):
    # r1, r2 = np.array(r1), np.array(r2)
    # assert r1.shape == (3,), f"psi: r1 = {r1}"
    # assert r2.shape == (3,), f"psi: r2 = {r2}"

    r1A, r1B, r2A, r2B = dist(r1, A), dist(r1, B), dist(r2, A), dist(r2, B)
    rA, rB, r1A2B, r1B2A = r1A + r2A, r1B + r2B, r1A + r2B, r1B + r2A

    return exp(-k * rA) + exp(-k * r1A2B) + exp(-k * r1B2A) + exp(-k * rB)


def kinetic(r1, r2, k):
    x1, y1, z1 = r1
    x2, y2, z2 = r2

    xA, yA, zA = A
    xB, yB, zB = B

    return -.5 * (
            k * (exp(
        -k * sqrt((x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2)) + exp(
        -k * sqrt((x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2))) * (
                    k * (x2 - xA) ** 2 * exp(-k * sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) / (
                            (x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                            z2 - zA) ** 2) + k * (
                            x2 - xB) ** 2 * exp(-k * sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2)) / (
                            (x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2) + (x2 - xA) ** 2 * exp(
                -k * sqrt(
                    (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) / (
                            (x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                            z2 - zA) ** 2) ** (3 / 2) + (
                            x2 - xB) ** 2 * exp(-k * sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2)) / (
                            (x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2) ** (3 / 2) - exp(
                -k * sqrt((x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                        z2 - zB) ** 2)) / sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2) - exp(
                -k * sqrt((x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                        z2 - zA) ** 2)) / sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) + k * (
                    exp(-k * sqrt((x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2)) + exp(-k * sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2))) * (
                    k * (y2 - yA) ** 2 * exp(-k * sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) / (
                            (x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                            z2 - zA) ** 2) + k * (
                            y2 - yB) ** 2 * exp(-k * sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2)) / (
                            (x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2) + (y2 - yA) ** 2 * exp(
                -k * sqrt(
                    (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) / (
                            (x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                            z2 - zA) ** 2) ** (3 / 2) + (
                            y2 - yB) ** 2 * exp(-k * sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2)) / (
                            (x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2) ** (3 / 2) - exp(
                -k * sqrt((x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                        z2 - zB) ** 2)) / sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2) - exp(
                -k * sqrt((x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                        z2 - zA) ** 2)) / sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) + k * (
                    exp(-k * sqrt((x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2)) + exp(-k * sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2))) * (
                    k * (z2 - zA) ** 2 * exp(-k * sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) / (
                            (x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                            z2 - zA) ** 2) + k * (
                            z2 - zB) ** 2 * exp(-k * sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2)) / (
                            (x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2) + (z2 - zA) ** 2 * exp(
                -k * sqrt(
                    (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) / (
                            (x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                            z2 - zA) ** 2) ** (3 / 2) + (
                            z2 - zB) ** 2 * exp(-k * sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2)) / (
                            (x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2) ** (3 / 2) - exp(
                -k * sqrt((x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                        z2 - zB) ** 2)) / sqrt(
                (x2 - xB) ** 2 + (y2 - yB) ** 2 + (z2 - zB) ** 2) - exp(
                -k * sqrt((x2 - xA) ** 2 + (y2 - yA) ** 2 + (
                        z2 - zA) ** 2)) / sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2)) + k * (
                    exp(-k * sqrt((x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2)) + exp(-k * sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2))) * (
                    k * (x1 - xA) ** 2 * exp(-k * sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) / (
                            (x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                            z1 - zA) ** 2) + k * (
                            x1 - xB) ** 2 * exp(-k * sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2)) / (
                            (x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2) + (x1 - xA) ** 2 * exp(
                -k * sqrt(
                    (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) / (
                            (x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                            z1 - zA) ** 2) ** (3 / 2) + (
                            x1 - xB) ** 2 * exp(-k * sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2)) / (
                            (x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2) ** (3 / 2) - exp(
                -k * sqrt((x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                        z1 - zB) ** 2)) / sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2) - exp(
                -k * sqrt((x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                        z1 - zA) ** 2)) / sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) + k * (
                    exp(-k * sqrt((x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2)) + exp(-k * sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2))) * (
                    k * (y1 - yA) ** 2 * exp(-k * sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) / (
                            (x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                            z1 - zA) ** 2) + k * (
                            y1 - yB) ** 2 * exp(-k * sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2)) / (
                            (x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2) + (y1 - yA) ** 2 * exp(
                -k * sqrt(
                    (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) / (
                            (x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                            z1 - zA) ** 2) ** (3 / 2) + (
                            y1 - yB) ** 2 * exp(-k * sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2)) / (
                            (x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2) ** (3 / 2) - exp(
                -k * sqrt((x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                        z1 - zB) ** 2)) / sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2) - exp(
                -k * sqrt((x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                        z1 - zA) ** 2)) / sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) + k * (
                    exp(-k * sqrt((x2 - xB) ** 2 + (y2 - yB) ** 2 + (
                            z2 - zB) ** 2)) + exp(-k * sqrt(
                (x2 - xA) ** 2 + (y2 - yA) ** 2 + (z2 - zA) ** 2))) * (
                    k * (z1 - zA) ** 2 * exp(-k * sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) / (
                            (x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                            z1 - zA) ** 2) + k * (
                            z1 - zB) ** 2 * exp(-k * sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2)) / (
                            (x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2) + (z1 - zA) ** 2 * exp(
                -k * sqrt(
                    (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2)) / (
                            (x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                            z1 - zA) ** 2) ** (3 / 2) + (
                            z1 - zB) ** 2 * exp(-k * sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2)) / (
                            (x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                            z1 - zB) ** 2) ** (3 / 2) - exp(
                -k * sqrt((x1 - xB) ** 2 + (y1 - yB) ** 2 + (
                        z1 - zB) ** 2)) / sqrt(
                (x1 - xB) ** 2 + (y1 - yB) ** 2 + (z1 - zB) ** 2) - exp(
                -k * sqrt((x1 - xA) ** 2 + (y1 - yA) ** 2 + (
                        z1 - zA) ** 2)) / sqrt(
                (x1 - xA) ** 2 + (y1 - yA) ** 2 + (z1 - zA) ** 2))
    )


def potential(r1, r2):
    electrons = (r1, r2)
    nuclei = (A, B)
    charges = (ZA, ZB)

    return (
            -sum(sum(Z / dist(re, rn) for Z, rn in zip(charges, nuclei)) for re
                 in electrons) +
            .5 * sum(
        sum(1 / dist(re1, re2) for re1 in electrons if all(re1 != re2)) for re2
        in
        electrons) +
            .5 * sum(sum(
        Z1 * Z2 / dist(rn1, rn2) for Z1, rn1 in zip(charges, nuclei) if
        rn1 != rn2) for Z2, rn2 in zip(charges, nuclei))
    )


def hamiltonian(r1, r2, k):
    return kinetic(r1, r2, k) + potential(r1, r2)


def E_local(r1, r2, k):
    # r1, r2 = np.array(r1), np.array(r2)
    # assert r1.shape == (3,), f"E: r1 = {r1}"
    # assert r2.shape == (3,), f"E: r2 = {r2}"

    return hamiltonian(r1, r2, k)


def new_psi(X, Y, k, i):
    X = X[0]

    if i == 0:
        return psi(r1=Y, r2=X, k=k)
    elif i == 1:
        return psi(r1=X, r2=Y, k=k)
    else:
        raise ValueError


def monte_carlo(n_particles, n_steps, ensemble_size, alpha, delta):
    shape = (n_particles, 3, ensemble_size)  # T{2, 3, 100}
    X = np.random.rand(*shape)

    E_ave = 0
    E_squ = 0
    accept = 0

    for _ in tqdm(range(n_steps)):
        for k in range(ensemble_size):
            psiX = psi(*X[..., k], alpha)

            for i in range(n_particles):
                Y = [X[i, j, k] + delta * (uniform(0, 1) - .5) for j in range(3)]
                psiY = new_psi(np.delete(X[..., k], i, 0), Y, alpha, i)

                Ac = min(psiY ** 2 / psiX ** 2, 1)

                if Ac >= uniform(0, 1):
                    X[i, :, k] = Y
                    accept += 1 / n_particles

            Ex = E_local(*X[..., k], alpha)
            E_ave += Ex
            E_squ += Ex ** 2

    A_ratio = accept / ensemble_size / n_steps
    E_mean = E_ave / ensemble_size / n_steps
    E_sigma = sqrt(E_squ / ensemble_size / n_steps - E_mean ** 2) / sqrt(ensemble_size * n_steps - 1)

    print(f"alpha={alpha:.3f}, A={A_ratio:.3f}, E={E_mean:.3f}, sigma={E_sigma:.3f}")

    return alpha, A_ratio, E_mean, E_sigma


if __name__ == '__main__':
    n_electrons = 2  # n electrons

    print("#" * 10, f"cpu count = {cpu_count()}", "#" * 10)

    n_steps = 10000
    ensemble_size = 1000
    delta = 1e-5

    print(f"n_steps={n_steps}, ensemble_size={ensemble_size}, delta={delta}")

    result = Parallel(n_jobs=-1)(
        delayed(monte_carlo)(n_electrons, n_steps, ensemble_size, alpha, delta)
        for alpha in np.linspace(0.8, 1.6, 7)
    )

    df = pd.DataFrame(np.array(result), columns=["alpha", "A", "E", "sigma"])
    df.to_csv("data.csv", sep=";", index=False)