def proba(data):
    """
    Créer le dictionnaire de probabilités d'apparition des différents caractères
    """
    assert len(data) != 0
    d = {}
    for x in data:
        d[x] = d.get(x, 0) + (1 / len(data))
    return d


def create_int(data):
    """
    Créer le dictionnaire des intervalles des différents caractères connaissant les données
    """
    p = proba(data)
    d = {}
    n = 0.0
    for c, v in p.items():
        d[c] = (n, n + v)
        n += v
    return d


def create_int2(p):
    """
    Créer le dictionnaire des intervalles des différents caractères connaissant les probas des différents caractères
    """
    d = {}
    n = 0.0
    for c, v in p.items():
        d[c] = (n, n + v)
        n += v
    return d


def encode(data):
    """
    effectue l'encodage des données
    """
    int = create_int(data)
    value = (0.0, 1.0)
    for x in data:
        d = value[1] - value[0]
        sup = value[0] + d * int[x][1]
        inf = value[0] + d * int[x][0]
        value = (inf, sup)
    return (value[0] + value[1]) / 2


def appartient(x, int):
    """
    teste l'appartenance de x à un intervalle fermé à gauche et ouvert à droite
    """
    assert len(int) == 2
    return x >= int[0] and x < int[1]


def inverse(dic):
    """
    renvoie le dictionnaire où les clés et valeurs sont inversées
    """
    d = {}
    for c, v in dic.items():
        d[v] = c
    return d


def decode(n, p, nbr_carac):
    d = inverse(create_int2(p))
    res = []
    i = n
    while len(res) < nbr_carac:
        for c, v in d.items():
            if appartient(i, c):
                res.append(v)
                i = (i - c[0]) / (c[1] - c[0])
                break
    return res


# Examples

if __name__ == "__main__":
    print(encode("WIKI"))
    print(decode(0.171875, {"W": 0.25, "I": 0.5, "K": 0.25}, 4))
    print(encode("AAABBCCCCC"))
    print(decode(0.010783125000000005, {"A": 0.3, "B": 0.2, "C": 0.5}, 10))
    print(encode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(
        decode(
            encode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            {
                1: 0.1,
                2: 0.1,
                3: 0.1,
                4: 0.1,
                5: 0.1,
                6: 0.1,
                7: 0.1,
                8: 0.1,
                9: 0.1,
                10: 0.1,
            },
            10,
        )
    )
