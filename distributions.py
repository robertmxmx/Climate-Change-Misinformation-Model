import numpy as np

def uniform():
    return((random.random()*2) -1)

def normal(m = 0, sd = 0.25):
    while True:
        temp = np.random.normal(m, sd)
        if temp < 1 and temp > -1:
            break
    return(temp)

def two_normal(m1 = 0.5, m2 = -0.5, sd = 0.25):
    while True:
        if random.random() > 0.5:
            temp = np.random.normal(m1, sd)
        else:
            temp = np.random.normal(m2, sd)

        if temp < 1 and temp > -1:
            break

    return(temp)


def three_normal(m1 = 0.5, m2 = 0, m3 = -0.5, sd = 0.10):
    while True:
        ran = random.random()
        if ran > 2/3:
            temp = np.random.normal(m1, sd)
        elif ran > 1/3:
            temp = np.random.normal(m2, sd)
        else:
            temp = np.random.normal(m3, sd)

        if temp < 1 and temp > -1:
            break

    return(temp)