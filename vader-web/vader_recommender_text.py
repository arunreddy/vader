import numpy as np
from numpy import matlib as mb


def padWuseLabeledData(A, order):
  n = A.shape[0]
  nl = len(order);
  ANEW = A.copy()
  ANEW = np.hstack((ANEW, np.zeros((n, nl))))
  ANEW = np.vstack((ANEW, np.zeros((nl, n + nl))))

  ANEW[n:, :] = ANEW[order, :]
  ANEW[:, n:] = ANEW[:, order]
  ANEW[n:n + nl, n:n + nl] = ANEW[order, order]

  return ANEW


def harmonic_function(W, fl):
  l = fl.shape[0]
  n = W.shape[0]

  L = np.diag(np.sum(W, 1)) - W;

  fu = -1 * np.dot(np.linalg.inv(L[l:n, l:n]), np.dot(L[l:n, :l], fl))

  return fu


def JEDI_harmonic(Y, A, order, Ysl):
  if (-1 in np.unique(Y)):
    Y = Y * 0.5 + 0.5

  numData = len(Y);
  numClass = 2;

  # TODO - Unique last.
  u, ia = np.unique(order, return_index=True);
  uniqueIdx = np.sort(ia);

  order_unique = [order[ind] for ind in uniqueIdx]
  Ysl_unique = [Ysl[ind] for ind in uniqueIdx]

  #  pad W with labeled nodes, get Wnew
  Anew = padWuseLabeledData(A, order_unique);

  Ynew = np.hstack([Y, Y[order_unique]])

  numDataNew = len(Ynew);

  ## get labeled matrix fl (learner provide labels)
  fl = []
  for _lbl in Ysl_unique:
    if _lbl == 1:
      fl.append([1, 0])
    else:
      fl.append([0, 1])

  fl = np.array(fl)

  idx = list(range(Anew.shape[0]))
  remaining_idx = list(set(idx) - set(order_unique))
  reorder_idx = order_unique + list(remaining_idx)
  Anew = Anew[reorder_idx, :]
  Anew = Anew[:, reorder_idx]

  flu = harmonic_function(Anew, fl)

  Yu = Ynew[len(order_unique):]
  Predu = np.argmax(flu, axis=1)

  # This is not clear
  Yu_predu = (((-1 * Yu) + 1) / 2)

  # output the probability w.r.t.the index of  Dt
  prob = np.zeros((numData, numClass));
  nFlu = flu.shape[0]

  _f = flu[nFlu - len(order_unique):, :]

  prob[order_unique, :] = flu[nFlu - len(order_unique):, :];
  restIdx = list(set(range(numData)) - set(order_unique))
  prob[restIdx, :] = flu[:nFlu - len(order_unique), :]

  return prob


def JEDI(X, Y, W, step, A, beta, indices, PL, YL):
  # if (-1 in np.unique(Y)):
  #   Y = Y * 0.5 + 0.5

  n, d = X.shape
  fvalue = np.zeros(shape=(n, 1))

  _X = X[indices, :]
  _Y = Y[indices]

  tminus = len(indices)

  prob = JEDI_harmonic(Y, A, indices, YL)


def JEDI_blackbox(D, Y, wstar, step, A, beta, _order, ysl_prob, ysl):
  # if (-1 in np.unique(Y)):
  #   Y = Y * 0.5 + 0.5

  numData = len(Y);
  fvalue = np.zeros((numData, 1));

  eta = step;

  Xs = D[_order[0], :]
  for _i in range(1, len(_order)):
    Xs = np.vstack((Xs, D[_order[_i], :]))

  Ys = np.zeros((len(_order), 1))
  for _i in range(len(_order)):
    Ys[_i] = Y[_order[_i]]

  d = D.shape[1]
  tminus = len(_order);

  prob = JEDI_harmonic(Y, A, _order, ysl);

  Ps = np.zeros((len(_order), 1))

  for _i in range(len(_order)):
    ys = Ys[_i];
    if ys == 1:
      Ps[_i] = ysl_prob[_i][1];
    else:  # ys == -1
      Ps[_i] = ysl_prob[_i][0]

  coeff = np.reshape(np.power(beta, np.arange(tminus, 0, -1)), (tminus, 1))

  Ys_arr = np.reshape(np.array(Ys), (len(Ys), 1))
  Ys_Ps = np.multiply((-1 * Ys_arr), Ps)

  Xs_arr = np.array(Xs)
  Ys_Ps_Xs = np.multiply(mb.repmat(Ys_Ps, 1, d), Xs_arr)

  Bs = mb.repmat(coeff, 1, d)

  Ys_Ps_Xs_Bs = np.multiply(Ys_Ps_Xs, Bs)
  FS = np.sum(Ys_Ps_Xs_Bs, axis=0)

  for id in range(numData):
    x = D[id, :]
    y = Y[id]

    eps = 0.0001
    if y == 1:
      pt = prob[id, 1]
      pnt = 1 / prob[id, 0]
    else:
      pt = prob[id, 0]
      pnt = 1 / prob[id, 1]


    if x.shape[0] == 1:
      epsilon_0 = y * (x.dot(wstar))

      k = np.power(eta, 2) * np.power(pt, 2) * np.power(np.linalg.norm(x), 2) + 2 * np.power(eta, 2) * FS.dot(
        -1 * y * x.transpose() * pt) - 2 * eta * (np.log(pnt) - np.log(1 + np.exp(-1 * epsilon_0)))

    else:
      epsilon_0 = y * (wstar.transpose().dot(x))

      k = np.power(eta, 2) * np.power(pt, 2) * np.power(np.linalg.norm(x), 2) + 2 * np.power(eta, 2) * FS.dot(
        -1 * y * x * pt) - 2 * eta * (
                  np.log(pnt) - np.log(1 + np.exp(-1 * epsilon_0)))

    fvalue[id] = k

  return prob, fvalue


def SGD_Gradient(Dt, Yt, w, l):
  n = Dt.shape[0]
  rndIdx = np.random.randint(0, n, 1)[0]
  x = Dt[rndIdx, :]
  y = Yt[rndIdx][0]

  epsilon = y * (w.transpose().dot(x))[0];

  if epsilon <= 0:
    dw = 1 / (1 + np.exp(epsilon)) * (-1 * y * x)
  else:
    dw = np.exp(-1 * epsilon) / (1 + np.exp(-1 * epsilon)) * (-1 * y * x)

  dw = np.reshape(dw, (dw.shape[0], 1))

  return dw + l * w, rndIdx


def compute_w_star(Dt, De, Yt, Ye):
  d = Dt.shape[1]
  maxIterations = 15000
  l = 0.001
  step_init = 0.2
  w = np.multiply((-1 + np.random.rand(d, 1) * 2), np.ones((d, 1)))

  nt = Dt.shape[0]
  ne = De.shape[0]

  fval = []
  for i in range(maxIterations):
    step = step_init

    dw, rndIdx = SGD_Gradient(Dt, Yt, w, l)

    w = w - step * dw
  #
  #   fval.append(np.sum(np.log(1+np.exp(np.multiply(-1*_Dt.dot(w),Yt)))) + 0.5*np.linalg.norm(w))
  #
  # plt.plot(list(range(maxIterations)),fval)
  # plt.show()

  return w


def jedi_next(Dt, Yt, De, Ye, order, ysl_prob, ysl, A, wo_SGD, beta):
  # Parameters.
  step = 0.2
  max_iterations = 600
  d = Dt.shape[1]

  # # Weighted adjacency matrix.
  # A = cosine_similarity(Dt)
  #
  #
  # A = A + A.transpose()
  #
  # _A = sp.sparse.coo_matrix(A)
  # rows = []
  # cols = []
  # w_uvs = []
  # for i, j in zip(_A.row, _A.col):
  #   u = Dt[i, :]
  #   v = Dt[j, :]
  #   w_uv = np.exp(-1 * (1 / 0.03) * (1 - u.transpose().dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))))
  #   w_uvs.append(w_uv)
  #   rows.append(i)
  #   cols.append(j)
  #
  # A_H = sp.sparse.csc_matrix((w_uvs, (rows, cols)), shape=(_A.shape[0],_A.shape[0]))
  # print(type(A_H))
  #
  # _A_H = A_H.todense()
  #
  # # Compute the W* using gradient descent.
  # w_star = compute_w_star(Dt, De, Yt, Ye)

  selectIdx, selectProb = JEDI_blackbox(Dt, Yt, wo_SGD, step, A, beta, order, ysl_prob, ysl);

  return selectIdx, selectProb
