"""
STONet: A Neural Operator for Modeling Solute Transport in Micro-Cracked Reservoirs

This code is part of the STONet repository: https://github.com/ehsanhaghighat/STONet

Citation:
@article{haghighat2024stonet,
  title={STONet: A neural operator for modeling solute transport in micro-cracked reservoirs},
  author={Haghighat, Ehsan and Adeli, Mohammad Hesan and Mousavi, S Mohammad and Juanes, Ruben},
  journal={arXiv preprint arXiv:2412.05576},
  year={2024}
}

Paper: https://arxiv.org/abs/2412.05576
"""

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle


class MinMaxScaler:
    def __init__(self):
        self._min = {}
        self._max = {}

    def fit(self, X: Dict[str, Any]):
        for key in X:
            self._min[key] = np.min(X[key], axis=0)
            self._max[key] = np.max(X[key], axis=0)
        return self

    def transform(self, X: Dict[str, Any]):
        X_norm = {}
        for key in X:
            if key in self._min:
                X_norm[key] = (X[key] - self._min[key])/(self._max[key] - self._min[key])
            else:
                X_norm[key] = X[key]
        return X_norm

    def inverse_transform(self, X: Dict[str, Any]):
        X_norm = {}
        for key in X:
            if key in self._min:
                X_norm[key] = (X[key]*(self._max[key] - self._min[key])) + self._min[key]
            else:
                X_norm[key] = X[key]
        return X_norm

    def save(self, path: str):
        with open(path, 'wb') as f:  # Overwrites any existing file.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:  # Overwrites any existing file.
            return pickle.load(f)


def zscore_normalize_features(X, _type='Standard', rtn_ms=False):
    """
    returns z-score normalized X by column
    Args:
      X : (numpy array (m,n))
    Returns
      X_norm: (numpy array (m,n)) input normalized by column
    """
    if _type == 'Standard':
      mu     = np.mean(X,axis=0)
      sigma  = np.std(X,axis=0)
      X_norm = (X - mu)/sigma
      if rtn_ms:
          return(X_norm, mu, sigma)
      else:
          return(X_norm)

    elif _type == 'MinMax':
        _max = np.max(X, axis=0)
        _min = np.min(X, axis=0)
        X_norm = (X - _min)/(_max - _min)
        if rtn_ms:
          return(X_norm, _max, _min)
        else:
          return(X_norm)

def zscore_normalize_test(X, mu, sigma, rtn_ms=True):
    if rtn_ms:   # Inputs
        X_norm = (X - mu)/sigma
    else:      # Output
        X_norm = (X*sigma) + mu
    return X_norm

def maxmin_normalize_test(X, _max, _min, rtn_ms=True):
    dim = X.shape[1]
    if rtn_ms:   # Inputs
        X_norm = (X - _min[..., :dim])/(_max[..., :dim] - _min[..., :dim])
    else:      # Output
        X_norm = (X*(_max[..., :dim] - _min[..., :dim])) + _min[..., :dim]
    return X_norm


def print_result(X, Y):
    print(f"X: {np.max(X[:,0]):0.4f}, {np.min(X[:,0]):0.4f}")
    print(f"Y: {np.max(X[:,1]):0.4f}, {np.min(X[:,1]):0.4f}")
    print(f"dp: {np.max(X[:,2]):0.4f}, {np.min(X[:,2]):0.4f}")
    print(f"K11: {np.max(X[:,3]):0.4e}, {np.min(X[:,3]):0.4e}")
    print(f"K12: {np.max(X[:,4]):0.4e}, {np.min(X[:,4]):0.4e}")
    print(f"K22: {np.max(X[:,5]):0.4e}, {np.min(X[:,5]):0.4e}")
    print(f"c: {np.max(X[:,6]):0.4e}, {np.min(X[:,6]):0.4e}")
    print(f"t: {np.max(X[:,7]):0.4e}, {np.min(X[:,7]):0.4e}")
    print(f"cdot: {np.max(Y[:,0]):0.4e}, {np.min(Y[:,0]):0.4e}\n")




def Read_Mesh(Mesh_file, Connectivity_file = None):
    Coord = np.genfromtxt(Mesh_file, dtype=float)

    if Connectivity_file != None:
        Elements = np.genfromtxt(Connectivity_file, dtype=float)
        return Coord, Elements

    else:
        return Coord


def Create_input_data(Mesh_file, dp, K11, K12, K22, c, t):
    Coord = Read_Mesh(Mesh_file)
    n = Coord[:,0].size
    X_predict = np.zeros((n,8))

    for i in range(n):
        X_predict[i, 0] = Coord[i, 0]
        X_predict[i, 1] = Coord[i, 1]
        X_predict[i, 2] = dp
        X_predict[i, 3] = K11[i, 0]
        X_predict[i, 4] = K12[i, 0]
        X_predict[i, 5] = K22[i, 0]
        X_predict[i, 6] = c[i, 0]
        X_predict[i, 7] = t

    return(X_predict)

def tecplot_file(Coords, Elements, time, c, cdot):
    NNodes = Coords[:,0].size
    NElem = Elements[:,0].size

    merged_data = np.concatenate((Coords[:,0].reshape(-1,1), Coords[:,1].reshape(-1,1), c.reshape(-1,1), cdot.reshape(-1,1)), axis = 1)

    if time == 0:
        _type = 'w'
    else:
        _type = 'a+'

    with open(f'Dist_cdott_6075_sin_8192_minmax_500_rand1500_1e-3_500sample_deep_mse_dp.plt', _type) as file:
        file.write('variables = "X"  "Y"  "c" "cdot"\n')
        file.write(f"ZONE ,N = {NNodes} , E = {NElem} , ZONETYPE = FEQUADRILATERAL , DATAPACKING=POINT, SOLUTIONTIME = {time}\n")
        np.savetxt(file, merged_data, fmt = '%.5e')
        file.write('\n\n')
        np.savetxt(file, Elements, fmt = '%.i')
    file.close()

def equivalent_permeability(L, H, Coord, Elements, orient, leng, lambda_val, aper, perm_int, Pleft, Pright, path):
    Elements = Elements.astype(int)
    noel = len(Elements[:, 0])
    nond = len(Coord[:, 0])
    ndoe = len(Elements[0, :])

    elNoX = int(L * 10)
    elNoY = int(H * 10)
    x = L / elNoX
    y = H / elNoY
    t = 1
    Kint = perm_int * np.array([[1, 0], [0, 1]])

    crNo = np.random.poisson(lambda_val, elNoX * elNoY)
    crNo[crNo == 0] = 1
    CrackNo = np.sum(crNo)

    orien = np.random.normal(orient, 15 * np.pi / 180, CrackNo)
    lengthC = np.random.lognormal(np.log(leng), leng * 1.5, CrackNo)
    aperture = np.random.lognormal(np.log(aper), abs(aper) * 1.5, CrackNo)

    X = []
    Y = []
    ii = 0
    for i in range(1, elNoY + 1):
        for j in range(1, elNoX + 1):
            ii += 1
            X.extend(np.random.uniform((j - 1) * x, j * x, crNo[ii - 1]))
            Y.extend(np.random.uniform((i - 1) * y, i * y, crNo[ii - 1]))

    crCoord = np.column_stack((X, -np.array(Y)))

    coef = 5
    keq = np.zeros((noel * 2, 2))

    for i in range(1, noel + 1):
        node_elm = Elements[i - 1, :].astype(int)
        Elem_Length = max(Coord[node_elm - 1, 0]) - min(Coord[node_elm - 1, 0])
        xmax = max(Coord[node_elm - 1, 0]) + coef * Elem_Length
        xmin = min(Coord[node_elm - 1, 0]) - coef * Elem_Length
        ymax = max(Coord[node_elm - 1, 1]) + coef * Elem_Length
        ymin = min(Coord[node_elm - 1, 1]) - coef * Elem_Length
        x_rve = Elem_Length + min(coef * Elem_Length, 0.7 - max(Coord[node_elm - 1, 0])) + min(
            coef * Elem_Length, min(Coord[node_elm - 1, 0]) - 0
        )
        y_rve = Elem_Length + min(coef * Elem_Length, 0 - max(Coord[node_elm - 1, 1])) + min(
            coef * Elem_Length, min(Coord[node_elm - 1, 1]) - (-0.5)
        )
        Vp = x_rve * y_rve * t

        Crack_elem = np.where(
            (crCoord[:, 0] <= xmax)
            & (crCoord[:, 0] >= xmin)
            & (crCoord[:, 1] <= ymax)
            & (crCoord[:, 1] >= ymin)
        )[0]

        ktensor = np.zeros((2, 2))

        for j in range(len(Crack_elem)):
            index = Crack_elem[j]
            N11 = np.array(
                [
                    [np.cos(orien[index]) ** 2, -1 / 2 * np.sin(2 * orien[index])],
                    [-1 / 2 * np.sin(2 * orien[index]), np.sin(orien[index]) ** 2],
                ]
            )
            ktensor += (1 / (12 * Vp)) * (aperture[index] ** 3) * lengthC[index] * N11

        keq[2 * i - 2 : 2 * i, :] = ktensor + Kint

    kout = np.zeros((nond, 3))
    count_node = np.zeros(nond)

    for i in range(noel):
        k = keq[2*i:2*(i+1), 0:2]
        nodeE = (Elements[i, 0:4] - 1 ).astype(int)
        kout[nodeE, :] += [k[0, 0], k[0, 1], k[1, 1]]
        count_node[nodeE] += 1

    # Avoid division by zero
    count_node[count_node == 0] = 1

    kout /= count_node[:, np.newaxis]


    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=200)

    # Plot elements
    for e in range(noel):
        Xn = Coord[Elements[e, 0:ndoe] - 1, 0]
        Yn = Coord[Elements[e, 0:ndoe] - 1, 1]

        plt.plot(Xn, Yn, color='k')
        plt.plot([Xn[0], Xn[1]], [Yn[0], Yn[1]], color='k')
        plt.plot([Xn[1], Xn[2]], [Yn[1], Yn[2]], color='k')
        plt.plot([Xn[2], Xn[3]], [Yn[2], Yn[3]], color='k')
        plt.plot([Xn[3], Xn[0]], [Yn[3], Yn[0]], color='k')

    # Plot cracks
    for i in range(CrackNo):
        Xcr = [crCoord[i, 0] - 0.5 * lengthC[i] * np.cos(orien[i]),
            crCoord[i, 0] + 0.5 * lengthC[i] * np.cos(orien[i])]

        Ycr = [crCoord[i, 1] - 0.5 * lengthC[i] * np.sin(orien[i]),
            crCoord[i, 1] + 0.5 * lengthC[i] * np.sin(orien[i])]

        plt.plot(Xcr, Ycr, linewidth=aperture[i]*10000)

    plt.figtext(0.11, 0.3, fr'$P_{{left}}$(y) = {Pleft} - 9792.34*y', color='red', backgroundcolor='white', rotation=90)
    plt.figtext(0.89, 0.3, fr'$P_{{right}}$(y) = {Pright} - 9792.34*y', color='red', backgroundcolor='white', rotation=90)
    plt.axis('off')
    plt.savefig(os.path.join(path, 'crack_dist.pdf'), bbox_inches='tight')
    plt.close()

    np.savetxt(os.path.join(path, 'k_data.txt'), kout)
    

    K11 = kout[:, 0].reshape(-1,1)
    K12 = kout[:, 1].reshape(-1,1)
    K22 = kout[:, 2].reshape(-1,1)
    
    x_t = np.asanyarray(Coord[:, 0])
    y_t = np.asanyarray(Coord[:, 1])
    k11_t = np.asanyarray(K11.flatten())
    k22_t = np.asanyarray(K22.flatten())

    fig, axs = plt.subplots(1, 2, figsize=(15, 4), dpi=200)

    contour_k11 = axs[0].tricontourf(x_t, y_t, k11_t, levels=20, cmap = 'gray')
    plt.colorbar(contour_k11, ax=axs[0])
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('$K_{xx}$')

    contour_k22 = axs[1].tricontourf(x_t, y_t, k22_t, levels=20, cmap = 'gray')
    plt.colorbar(contour_k22, ax=axs[1])
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title('$K_{yy}$')
    plt.savefig(os.path.join(path, 'K.png'), dpi=200)
    plt.close()

    
    return kout

