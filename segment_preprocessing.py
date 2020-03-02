import vtk
import numpy as np
from sklearn.decomposition import PCA
from vtk.util import numpy_support
from sklearn.neighbors import NearestNeighbors


class myTransform(vtk.vtkTransform):
    def __init__(self, ndarray=None):
        super(myTransform, self).__init__()

        if isinstance(ndarray, np.ndarray) and ndarray.shape == (4, 4):
            self.set_from_numpy_mat(ndarray)

    def getRigidTransform(self):
        out = myTransform()
        out.Translate(*self.GetPosition())
        out.RotateWXYZ(*self.GetOrientationWXYZ())
        return out

    def convert_np_mat(self):
        mat = self.GetMatrix()
        np_mat = np.zeros([4, 4], dtype=np.float64)
        for i in range(4):
            for j in range(4):
                np_mat[i, j] = mat.GetElement(i, j)
        return np_mat

    def GetInverse(self, vtkMatrix4x4=None):
        # inverse_t = super(myTransform, self).GetInverse()
        mat4x4 = self.convert_np_mat()
        t = myTransform()
        t.set_from_numpy_mat(np.linalg.inv(mat4x4))
        return t

    def set_from_numpy_mat(self, np_mat):
        mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                mat.SetElement(i, j, np_mat[i, j])
        self.SetMatrix(mat)


def compute_curvature(polydata):
    curvaturesFilter = vtk.vtkCurvatures()
    curvaturesFilter.SetInputData(polydata)
    curvaturesFilter.SetCurvatureTypeToMinimum()
    curvaturesFilter.SetCurvatureTypeToMaximum()
    curvaturesFilter.SetCurvatureTypeToGaussian()
    curvaturesFilter.SetCurvatureTypeToMean()
    curvaturesFilter.Update()
    return curvaturesFilter.GetOutput()


def compute_transform(basis, origin):
    T = np.eye(4)
    T[:3, :3] = basis
    T[:3, 3] =  -np.dot(basis, origin)

    return T


def transoformPolydata(polydata, transform):
    vtkfilter = vtk.vtkTransformFilter()
    vtkfilter.SetInputData(polydata)
    vtkfilter.SetTransform(transform)
    vtkfilter.Update()
    return vtkfilter.GetOutput()


def polydata2actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def visualize(actors):
    ren = vtk.vtkRenderer()
    for act in actors:
        ren.AddActor(act)
    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    iren.Initialize()
    iren.Start()


def get_sample():
    # FIXME
    path = "C:/Users/jurag/PycharmProjects/pyqt_ex/sample/67998_171129150550 (3) (4)/UpperJawScan.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def main():
    polydata = get_sample()

    numpy_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

    pca = PCA()
    pca.fit(numpy_points)
    T = compute_transform(pca.components_, pca.mean_)

    # teeth alignemnt
    t = myTransform()
    t.set_from_numpy_mat(np.linalg.inv(T))
    aligned_teeth = transoformPolydata(polydata, t)
    #
    points = numpy_support.vtk_to_numpy(aligned_teeth.GetPoints().GetData())
    bounds = np.array(aligned_teeth.GetBounds())
    x_min, x_max, y_min, y_max, z_min, z_max = [i[0] for i in np.split(bounds, 6, 0)]

    # print(point_min, point_max, aligned_teeth.GetBounds())
    H = z_max - z_min
    L = x_max - x_min
    W = y_max - y_min
    xc = (x_max + x_min)/2
    yc = (y_max + y_min)/2

    Eu1 = (points[:, 2] - z_min)/H
    # E2 = curvature_thresh
    ctr = np.array([xc, yc])
    den = np.array([0.5*L, 0.5*W])
    Eu3 = np.linalg.norm((points[:, :2] - ctr) / den, axis=1)

    # simply nearest-neighbor not geodesic
    thresh = 2.5
    transform_teeth_curvature = compute_curvature(aligned_teeth)
    curvature = numpy_support.vtk_to_numpy(transform_teeth_curvature.GetPointData().GetScalars())
    sharpness_inds, = np.where(curvature > thresh)
    curvature_thresh = sharpness_inds.astype(np.float32)
    sharp_points = points[sharpness_inds]
    neighbor = NearestNeighbors()
    neighbor.fit(sharp_points)
    dist, inds = neighbor.kneighbors(points, n_neighbors=5)

    gdmax = dist[:, 0].max()
    Eu2 = 1 - dist[:, 0] / gdmax
    # reference parameter
    a1, a2, a3 = 0.4, 0.5, 0.1
    E1 = a1*Eu1 + a2*Eu2 + a3*Eu3

    # experiment threshold value
    std_thresh = np.array([-1.0, -0.1])
    e1mean = E1.mean()
    e1std = E1.std()
    e_thresh = e1mean + e1std * std_thresh
    t1, t2 = tuple(e_thresh)

    gingiva_candidate = E1 < t1
    boundary_candidate = np.logical_and(t1 <= E1, E1 < t2)
    teeth_candidate = E1 >= t2

    # set scalar value to visualize
    scalars = np.zeros([points.shape[0]])
    scalars[boundary_candidate] = 0.5
    scalars[teeth_candidate] = 1.0

    # set scalars
    aligned_teeth.GetPointData().SetScalars(numpy_support.numpy_to_vtk(scalars))
    act_teeth = polydata2actor(aligned_teeth)

    t2 = myTransform()
    # t2.Concatenate(t)
    t2.Scale(10, 10, 10)
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(t2)

    visualize([act_teeth, axes])


if __name__=="__main__":
    main()