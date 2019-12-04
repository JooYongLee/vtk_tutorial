import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def draw_vector(v0, v1, ax=None, color='r'):
    ax1 = ax or plt.gca()

    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0, color=color)
    ax1.annotate('', v1, v0, arrowprops=arrowprops)

def pca_compare():
    # cov = [[2, 8], [8, 100]]
    # mean = [0, 2]
    # x, y = np.random.multivariate_normal(mean, cov, 1000).T
    # pts = np.stack([x, y], axis=1)
    pts = read_data()
    print("---->\n", pts[:5])



    fig = plt.figure()
    ax = fig.add_subplot(111)


    pca = PCA()
    pca.fit(pts)
    print(pca.components_)

    m = pts.shape[1]
    # ax.plot(x, y, 'x')

    aligned = pts-pts.mean(axis=0)
    print(pts.mean(axis=0), "mean-----------------")
    print("alinged----->\n", aligned)
    # s = np.linalg.svd(aligned)
    u, s, vh = np.linalg.svd(aligned.T, full_matrices=False)
    #np.linalg.svd()
    print("svd signular values===========>\n", s)
    print("svd=======>\n", u)

    v = np.square(s)
    print(v[0]/(v[0]+v[1]))

    cov = np.cov(aligned, rowvar = False)

    myconv = np.dot(aligned.T, aligned) / (aligned.shape[0]-1)

    print("cov\n", myconv)
    myeig = [np.dot(np.dot(u[i], myconv), u[i]) for i in range(s.shape[0])]
    print("my eigen value-------\n", myeig)
    print(np.isclose(cov, myconv).all())

    # for i in range(m):
    #     v0 = pca.mean_
    #     v1 = pca.mean_ + pca.components_[i] * pca.explained_variance_[i] * 0.2
    #     print(i, v0, v1, pca.explained_variance_[i])
    #     draw_vector(v0, v1, color='r')

    # tpt = pca.transform(pts) / pca.explained_variance_
    tpt = np.dot(u, aligned.T).T / s
    print("pca api-------->\n", pca.components_)
    print("pca api-------->\n", pca.explained_variance_, pca.explained_variance_ratio_.sum())

    print(tpt.shape, "s", s / np.sum(s), pca.explained_variance_ratio_)
# ax.plot(tpt[:, 0], tpt[:, 1], 'x')
#are()
# ax.axis('equal')
# plt.show()
def prepare_data():
    cov = [[2, 8], [8, 100]]
    mean = [0, 2]
    x, y = np.random.multivariate_normal(mean, cov, 1000).T

    pts = np.stack([x, y], axis=1)
    print(pts.shape)
    filename = "data.txt"
    with open(filename, "w") as f:
        f.write("{} {}\n".format(pts.shape[0], pts.shape[1]))
        for p in pts:

            for k in p:
                f.write(str(k))
                f.write(" ")
            f.write("\n")

def read_data():
    filename = "data.txt"
    pts = []
    with open(filename, "r") as f:
        lines = f.readlines()
        print(len(lines))
        # for i in range(1, len(l))
        shape = lines[0]
        for p in lines[1:]:
            # print(p.split())
            pts.append([float(v) for v in p.split()])
    # print(pts)
    # print(shape)
    pts = np.asarray(pts)

    # data = [ 1, 5,
    # 2, 3,
    # 1, 3,
    # 5, 1,
    # 4, 2]
    # data = np.asarray(data, dtype=np.float32)
    # print(data)
    # pts = data.reshape([-1, 2])
    # print(pts)
    # print(pts.shape)
    return pts
        # for l in lines:
        #     print(l)

        # f.write("{} {}\n".format(pts.shape[0], pts.shape[1]))
        # for p in pts:
        #
        #     for k in p:
        #         f.write(str(k))
        #         f.write(" ")
        #     f.write("\n")

if __name__=="__main__":
    pca_compare()
    # prepare_data()
    # read_data()