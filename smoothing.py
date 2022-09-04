import mpmath as mp
import numpy as np
from myFuncs import *


# MATLAB: Array(col,row)
# Python: Array[row][col]

def start(P0=None, P1=None, P2=None, V=None):

    AA = np.array([[P0[0], P1[0], P2[0]],
                   [P0[1], P1[1], P2[1]],
                   [0.0, 0.0, 0.0]], np.float64)

    # AA = np.random.randint(-100, 100, (3,3))

    N = np.size(AA, 1)  # <- число узлов

    V0 = 0.1 * V
    d_min = 0.1
    d_goal = 5
    lstrfruher = 0
    dti = np.zeros((N - 2,))
    fistri = np.zeros((N - 2,))
    psistri = np.zeros((N - 2,))
    Ri = np.zeros((N - 2,))
    ai = np.zeros((N - 2,))
    bi = np.zeros((N - 2,))
    tl = np.zeros((N - 2,))
    li = np.zeros((N - 2,))
    tr = np.zeros((N - 2,))
    Mi = np.zeros((N - 2, 2))
    Ci = np.zeros((N - 2, 3))
    R4lskcell = np.zeros((N - 2, 4, 4))
    d_prev = 1e-16
    pi = 3.14159265

    for i in range(N - 2):
        A1 = AA[:,i:i+1]
        A2 = AA[:,i+1:i+2]
        A3 = AA[:,i+2:i+3]
        A2A1 = A1 - A2
        A2A3 = A3 - A2

        l1 = np.sqrt(A2A1[0] ** 2 + A2A1[1] ** 2 + A2A1[2] ** 2)
        li[i] = np.float16(l1)
        l2 = np.sqrt(A2A3[0] ** 2 + A2A3[1] ** 2 + A2A3[2] ** 2)

        ort1 = A2A1 / l1
        ort3 = cross(ort1, A2A3) / np.linalg.norm(cross(ort1, A2A3))
        ort2 = cross(ort3, ort1)
        fi = np.arccos(dot(A2A1, A2A3) / (l1 * l2))
        R4loc = np.float64(np.vstack((np.hstack((ort1, ort2, ort3, A2)), [0, 0, 0, 1])))
        R4lskcell[i] = R4loc
        A2_loc = np.array([0, 0])
        A1_loc = np.dot(np.linalg.inv(R4loc), np.vstack((A1, 1)))
        A1_loc = np.array([A1_loc[0], A1_loc[1]])
        A3_loc = np.dot(np.linalg.inv(R4loc), np.vstack((A3, 1)))
        A3_loc = np.array([A3_loc[0], A3_loc[1]])

        d_exact = 1e-12  # <- точность расчёта

        fi2 = pi - fi  # <- угол поворота вектора скорости при движении по кусочно-линейной кривой в данной вершине
        a = (V - V0) * fi2 / (2 * pi)
        b = V - a

        # Vc=V-2*a;% минимальная скорость на дуге окруж ослабляется тем больше, чем больше fi,
        # причем - линейно от V до минимума V0=2 м/с при fi=pi

        psi = 0.5 * fi2

        lstrmax = min([(l1 - lstrfruher) / 2, l2 / 3])
        d_max = np.sin(psi) * lstrmax / (1 + np.cos(psi))

        if d_min < d_max:
            if d_max >= d_goal >= d_min:
                d = d_goal
            elif d_goal > d_max:
                d = d_max
            else:
                d = d_min
        else:
            d = min([d_goal, d_max])

        lstr = d * (1 + np.cos(psi)) / np.sin(psi)
        R = lstr / np.tan(psi)
        dt = R * psi / b  # <- половина интервала переменности скорости при движении по дуге

        kB = lstr / l1
        kC = lstr / l2

        xB = np.float64(A2_loc[0] + (A1_loc[0] - A2_loc[0]) * kB)
        yB = np.float64(A2_loc[1] + (A1_loc[1] - A2_loc[1]) * kB)

        xC = np.float64(A2_loc[0] + (A3_loc[0] - A2_loc[0]) * kC)
        yC = np.float64(A2_loc[1] + (A3_loc[1] - A2_loc[1]) * kC)

        B_loc = np.array([xB, yB])
        C_loc = np.array([xC, yC])

        # Находим центр окружности

        A2B_loc = B_loc - A2_loc
        A2C_loc = C_loc - A2_loc
        dn_loc = np.divide((A2B_loc + A2C_loc), (2 * np.sin(0.5 * fi2) * lstr))
        M_loc = A2_loc + dn_loc * np.sqrt(R ** 2 + lstr ** 2)
        xM = M_loc[0]
        yM = M_loc[1]

        if 1 + d_exact >= abs((xB - xM) / R) > 1:
            fi1str = np.arccos(np.sign(xB - xM))
        else:
            fi1str = np.arccos((xB - xM) / R)

        fi2str = -1.0 * fi1str

        if 1 + d_exact >= abs((yB - yM) / R) > 1:
            psi1str = np.arccos(np.sign(yB - yM))
        else:
            psi1str = np.arccos((yB - yM) / R)

        psi2str = -1.0 * fi1str

        if abs((xC - xM) / R - np.cos(2 * psi + fi1str)) < d_exact:
            fistr = fi1str
        else:
            fistr = fi2str

        if abs((yC - yM) / R - np.sin(2 * psi + psi1str)) < d_exact:
            psistr = psi1str
        else:
            psistr = psi2str

        dti[i] = dt
        fistri[i] = fistr
        psistri[i] = psistr
        Ri[i] = R
        Mi[i] = M_loc  # двумерная точка
        C = np.dot(R4loc, np.array([[xC], [yC], [0.0], [1.0]], np.float64))
        Ci[i] = np.reshape(C[0:3], (3,))  # <- трёхмерная точка
        ai[i] = a
        bi[i] = b

        if i == 0:
            tl[i] = (l1 - lstr) / V

        else:
            tl[i] = tr[i - 1] + (l1 - (lstrfruher + lstr)) / V

        tr[i] = tl[i] + 2 * dt

        lstrfruher = lstr
        # Конец цикла

    CiN = np.reshape(Ci[N - 3], (1, 3))
    AAn = np.array([AA[0][-1],AA[1][-1],AA[2][-1]])
    AAn_prev = np.array([AA[0][-2],AA[1][-2],AA[2][-2]])

    T = tr[N - 3] + strekeAB(Ci[N-3], AAn) / V
    Nt = 10 * N
    dt0 = T / (Nt - 1)
    t0 = 0
    Pmass = np.array([[0],[0],[0]])
    Vmass = 0.0
    ti = 0
    xyz = np.array([[0],[0],[0]])
    v = 0

    for j in range(Nt):
        t = t0 + j * dt0

        if tl[0] + d_prev >= t >= 0 - d_prev:
            xyz = AA[:,0:1] + ((AA[:,1:2] - AA[:,0:1]) / li[0]) * V * t
            v = V


        elif t <= tr[N - 3] + d_prev:
            for k in range(N - 2):
                if tl[k] - d_prev <= t <= tr[k] + d_prev:
                    fi_c = (ai[k] * dti[k] * np.sin(pi * (t - tl[k]) / dti[k]) / pi + bi[k] * (t - tl[k])) / Ri[k]
                    xt = Mi[k][0] + Ri[k] * np.cos(fi_c + fistri[k])
                    yt = Mi[k][1] + Ri[k] * np.sin(fi_c + psistri[k])
                    xyz = np.dot(R4lskcell[k], np.array([[xt], [yt], [0], [1]]))  # переход в глобальную трехмерную систему
                    xyz = np.array([xyz[0], xyz[1], xyz[2]])
                    v = ai[k] * np.cos(pi * (t - tl[k]) / dti[k]) + bi[k]

                    break
                elif k > 1 and tr[k - 1] - d_prev <= t <= tl[k] + d_prev:  # движение по отрезку прямой
                    # сразу в глобальной трехмерной системе
                    xyz = Ci[k - 1] + ((AA[k + 1] - AA[k]) / li[k]) * V * (t - tr[k - 1])
                    xyz = np.reshape(xyz, (3, 1))

                    break
        else:  # выполняется: t>tr(N-2) !работает исправно!
            # сразу в глобальной трехмерной системе
            xyz = Ci[N - 3] + ((AAn - AAn_prev) / strekeAB(AAn, AAn_prev)) * V * (t - tr[N - 3])
            xyz = np.reshape(xyz, (3, 1))
            v = V

        Pmass = np.vstack((Pmass, xyz))
        Vmass = np.vstack((Vmass, v))
        ti = np.vstack((ti, t))

    P = np.reshape(Pmass[3:],(np.size(ti[1:]),3))
    return P, Vmass[1:], ti[1:]
