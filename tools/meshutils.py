import json
from typing import List

import vtk
try:
    import pyvista as pv
    import pyacvd
except Exception as e:
    import commons
    logger = commons.get_runtime_logger()
    logger.error(e)
    logger.error("cannot use remesh(pyacvd)")

from . import vtk_utils
numpy_support = vtk_utils.numpy_support
# from vtk.util import numpy_support
import numpy as np
from sklearn.neighbors import NearestNeighbors
import igl

# from autoplanning import vtk_utils

from commons import get_runtime_logger, timefn2
# from common import timefn2

show = vtk_utils.show_actors
recon = vtk_utils.reconstruct_polydata
pts2act = vtk_utils.create_points_actor
p2a = vtk_utils.polydata2actor
to_spheres = vtk_utils.create_sphere


def remesh_avcd(indata: vtk.vtkPolyData, ratio=.5, divide=False):
    v, f = vtk_2_vf(indata)
    pdata = pv.PolyData(v, f)
    clus = pyacvd.Clustering(pdata)
    # mesh is not dense enough for uniform remeshing
    if ratio > 1.0 or divide:
        clus.subdivide(2)
    clus.cluster(int(indata.GetNumberOfPoints() * ratio))
    return clus.create_mesh()


def _actor_2_numpy(act):
    return numpy_support.vtk_to_numpy(act.GetMapper().GetInput().GetPoints().GetData())


def _actor_2_numpy_polys(act):
    return numpy_support.vtk_to_numpy(act.GetMapper().GetInput().GetPolys().GetData()).reshape([-1, 4])


def to_polydata(polydata_or_actor):
    if issubclass(type(polydata_or_actor), vtk.vtkActor):
        return polydata_or_actor.GetMapper().GetInput()
    elif issubclass(type(polydata_or_actor), vtk.vtkPolyData):
        return polydata_or_actor
    else:
        raise ValueError('not implemented')



def vtk_2_vf(polydata_or_actor):
    if issubclass(type(polydata_or_actor), vtk.vtkActor):
        pts, polys = _actor_2_vf(polydata_or_actor)
    elif issubclass(type(polydata_or_actor), vtk.vtkPolyData):
        pts = numpy_support.vtk_to_numpy(polydata_or_actor.GetPoints().GetData())
        polys = numpy_support.vtk_to_numpy(polydata_or_actor.GetPolys().GetData())
    else:
        raise ValueError
    return pts, polys.reshape([-1, 4])


def vtk_to_points(polydata_or_actor):
    if issubclass(type(polydata_or_actor), vtk.vtkActor):
        pd = polydata_or_actor.GetMapper().GetInput()
    elif issubclass(type(polydata_or_actor), vtk.vtkPolyData):
        pd = polydata_or_actor

    return numpy_support.vtk_to_numpy(pd.GetPoints().GetData())


def vtk_2_vfigl(polydata_or_actor):
    if issubclass(type(polydata_or_actor), vtk.vtkActor):
        pts, polys = _actor_2_vf(polydata_or_actor)
    elif issubclass(type(polydata_or_actor), vtk.vtkPolyData):
        pts = numpy_support.vtk_to_numpy(polydata_or_actor.GetPoints().GetData())
        polys = numpy_support.vtk_to_numpy(polydata_or_actor.GetPolys().GetData())
    else:
        raise ValueError
    polys_reshape = polys.reshape([-1, 4])
    return pts, polys_reshape[:, 1:]


def _actor_2_vf(act):
    v = _actor_2_numpy(act)
    f = _actor_2_numpy_polys(act)
    return v, f


def expand_vert_mask(polys, pts_mask, op=np.any, iter=2):
    assert op in [np.all, np.any]
    # select_pts = pts[pts_mask]

    # args = np.where(pts_mask)[0]

    reshape_polys = polys.reshape([-1, 4])
    if iter>0:
        for _ in range(iter):
            select_polys = pts_mask[reshape_polys[:, 1:]]
            valid_cell = op(select_polys, axis=-1)

            # valid_polys = np.where(valid_cell)[0]
            valid_vert_args = np.unique(reshape_polys[valid_cell, 1:])
            pts_mask = np.zeros([pts_mask.shape[0]], dtype=np.bool)
            pts_mask[valid_vert_args] = True
        return pts_mask
    else:
        return pts_mask


def extract_polydata(pts, polys, pts_mask, op=np.all, return_polys=False):
    assert op in [np.all, np.any]
    # select_pts = pts[pts_mask]

    # args = np.where(pts_mask)[0]
    reshaped_poly = polys.reshape([-1, 4])

    reshape_polys = polys.reshape([-1, 4])
    select_polys = pts_mask[reshape_polys[:, 1:]]
    valid_cell = op(select_polys, axis=-1)

    valid_polys = np.where(valid_cell)[0]
    valid_vert_args = np.unique(reshape_polys[valid_cell, 1:])
    args = valid_vert_args
    select_pts = pts[valid_vert_args]

    _2map = np.ones([pts.shape[0]], dtype='int64') * (-1)
    _2map[args] = np.arange(args.shape[0])
    polys_map = _2map[reshaped_poly[:, 1:]]
    polys_map_valid = polys_map[valid_cell]

    np.testing.assert_equal(polys_map_valid >= 0, True)

    polys_map_valid_cells = np.ones([polys_map_valid.shape[0], 4], dtype=polys_map_valid.dtype) * 3
    polys_map_valid_cells[:, 1:] = polys_map_valid

    if return_polys:
        return select_pts, polys_map_valid_cells, valid_vert_args, valid_polys
    else:
        return select_pts, polys_map_valid_cells, valid_vert_args



def decimation(polydata: vtk.vtkPolyData, deci: float):
    dec = vtk.vtkDecimatePro()
    dec.SetInputData(polydata)
    dec.SetPreserveTopology(True)
    dec.SetTargetReduction(deci)
    dec.SetBoundaryVertexDeletion(False)
    dec.Update()
    return dec.GetOutput()


def smoothing_polydata(polydata, iter=15, factor=0.6):
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(polydata)
    smoothFilter.SetNumberOfIterations(iter)
    smoothFilter.SetRelaxationFactor(factor)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    return smoothFilter.GetOutput()


def get_vert_indices(polydata: vtk.vtkPolyData, points):
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    inds = []
    for pt in points:
        i = loc.FindClosestPoint(*pt)
        inds.append(i)
    return np.asarray(inds)


def extracting_edge_mask(voxel_mask: np.ndarray):
    """
    from integer mask, generate edge-mask within non-zero-values
    :param voxel_mask:
    :return:
    """
    assert voxel_mask.ndim == 3

    ############
    mask = voxel_mask
    shifts = []
    for i in range(8):
        shifts.append([int(t) * 2 for t in format(i, '#05b')[2:]])

    # padding array, [d,h,w]---->[d+2, h+2, w+2]
    mask_ex = np.pad(mask, [[1, 1], [1, 1], [1, 1]], mode="reflect")

    d, h, w = mask.shape
    # get volume neighbor index
    neighbor_mask = []
    for shift in shifts:
        i, j, k = shift
        neighbor_mask.append(mask_ex[i:i + d, j:j + h, k:k + w])

    edge_mask = np.zeros_like(mask)

    # compare different(edge) value in center with neighbor 8 index value
    for sm in neighbor_mask:
        edge_mask = np.where(np.logical_xor(mask, sm), np.ones_like(mask), edge_mask)
    edge_mask = np.where(mask > 0, edge_mask, np.zeros_like(edge_mask))

    return edge_mask
    # show([vtk_utils.numpyvolume2vtkvolume(edge_mask * 255, 123)])


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def smoothing_edge_polydata(polydata, iteration=10, w0=0.5):
    edges = compute_boundary_pts_array(polydata)
    edges0 = edges[:-1]
    args = get_vert_indices(polydata, edges0)

    for _ in range(iteration):
        edges0 = smoothing_edge(edges0, w0)
    V0, F0 = vtk_2_vf(polydata)
    V1 = V0.copy()
    V1[args] = edges0
    return recon(V1, F0)

def smoothing_edge(loops, w0=0.5, iteration=5):

    def _smoothing_edge(loops, w0):
        num = loops.shape[0]
        ix = np.arange(num + 1) % num
        iy = np.concatenate([[-1], np.arange(num + 1)]) % num
        i0 = iy[:num]
        i1 = iy[1:num+1]
        i2 = iy[2:num+2]
        # w0 = 0.5
        cw0 = (1 - w0) / 2
        pts = loops[i1] * w0 + (loops[i0] + loops[i2]) * cw0
        return pts

    pts = loops
    for i in range(iteration):
        pts = _smoothing_edge(pts, w0)
    return pts



def compute_boundary_pts_array_test(target_polydata, return_indices=True, return_full=False):
    target_edge = vtk_utils.compute_boundary_edge(target_polydata)
    edge_pts = numpy_support.vtk_to_numpy(target_edge.GetPoints().GetData())
    edge_lines = numpy_support.vtk_to_numpy(target_edge.GetLines().GetData())
    if edge_pts.size == 0:
        logger = get_runtime_logger()
        logger.fatal("boundary points not exists")
        return np.array([])
    # show([p2a(target_edge)])
    # line check
    assert (edge_lines.reshape([-1, 3])[:, 0] == 2).all()
    reshape_edge_lines = edge_lines.reshape([-1, 3])

    i0 = reshape_edge_lines[:, 1]
    i1 = reshape_edge_lines[:, 2]
    arg_sorti0 = np.argsort(i0)
    sort_i0 = i0[arg_sorti0]
    sort_i1 = i1[arg_sorti0]

    edge_arg_inds = []
    # init
    init_ix = 0
    bound_args = [sort_i0[init_ix]]
    masks = np.ones([sort_i0.shape[0]], dtype=np.bool)

    non_continue = np.cumsum(np.diff(sort_i0, prepend=[sort_i0[0] - 1]) - 1)
    non_continue = non_continue * -1
    # pad_noncontinue = np.pad(non_continue, )
    # if edge_pts.shape[0] > sort_i0.shape[0]
    num_pad = np.maximum(edge_pts.shape[0] - sort_i0.shape[0], 0)
    pad_noncontinue = np.pad(non_continue, [0, num_pad], 'edge')
    for _ in range(len(sort_i1)):
        inds = bound_args[-1]
        comp_inds = inds + pad_noncontinue[inds]
        masks[comp_inds] = False
        ix = sort_i1[comp_inds]
        is_closed = bound_args[0] == ix
        # print(bound_args[-1], ix)
        bound_args.append(ix)
        # print()

        if is_closed:
            # add new closed loopsa
            edge_arg_inds.append(bound_args)
            init_ix = np.where(masks)[0]
            # if reamin next loop
            if init_ix.size > 0:
                # print(init_ix)
                bound_args = [sort_i0[init_ix[0]]]
        else:
            pass

    closed_loop = [ix[0] == ix[-1] for ix in edge_arg_inds]
    logger = get_runtime_logger()
    logger.info("closed loop check {}".format(closed_loop))

    line_lengts_list = []
    line_pts_list = []
    for ix in edge_arg_inds:
        line_pts = edge_pts[ix]
        diff_lines = np.diff(line_pts, axis=0)
        approximate_length = np.linalg.norm(diff_lines, axis=-1).sum()
        line_pts_list.append(line_pts)
        line_lengts_list.append(approximate_length)
    logger.info("each loop(or line) length : {}".format(line_lengts_list))
    is_show = False
    if is_show:
        curve_list = [vtk_utils.create_curve_actor(pts) for pts in line_pts_list]
        for pts, length in zip(line_pts_list, line_lengts_list):
            logger.info("length:{}".format(length))

    if return_full:
        if return_indices:
            return [edge_pts[arg] for arg in edge_arg_inds], edge_arg_inds
        else:
            return [edge_pts[arg] for arg in edge_arg_inds]
    else:
        edge_sort_inds = np.array(edge_arg_inds[0])
        edge_sort_pts = edge_pts[edge_sort_inds]
        return edge_sort_pts


@timefn2
def compute_boundary_pts_array(target_polydata, return_indices=True, return_full=False):
    if target_polydata.GetNumberOfLines() == 0:
        target_edge = vtk_utils.compute_boundary_edge(target_polydata)
    else:
        target_edge = target_polydata
    edge_pts = numpy_support.vtk_to_numpy(target_edge.GetPoints().GetData())
    edge_lines = numpy_support.vtk_to_numpy(target_edge.GetLines().GetData())
    if edge_pts.size == 0:
        logger = get_runtime_logger()
        logger.fatal("boundary points not exists")
        return np.array([])
    # edge_actor = p2a(target_edge)
    # edge_actor.GetProperty().SetLineWidth(5)
    # show([edge_actor])
    # show([p2a(target_edge)])
    # line check
    assert (edge_lines.reshape([-1, 3])[:, 0] == 2).all()
    reshape_edge_lines = edge_lines.reshape([-1, 3])

    i0 = reshape_edge_lines[:, 1]
    i1 = reshape_edge_lines[:, 2]
    arg_sorti0 = np.argsort(i0)
    sort_i0 = i0[arg_sorti0]
    sort_i1 = i1[arg_sorti0]
    sort_i0_ex = np.arange(edge_pts.shape[0])
    sort_i0_ex[np.arange(sort_i0.shape[0])] = sort_i0
    edge_arg_inds = []

    sort_i01 = np.stack([sort_i0, sort_i1], axis=-1)
    # for test
    # init
    init_ix = 0
    bound_args = [sort_i0[init_ix]]
    masks = np.ones([edge_pts.shape[0]], dtype=np.bool)
    # masks[init_ix] = True
    non_closed_args = []
    tomap = np.zeros([edge_pts.shape[0]], dtype=sort_i0.dtype)
    tomap[sort_i0] = np.arange(sort_i0.shape[0])
    masks[init_ix] = False
    for _ in range(len(sort_i1)):
        inds = bound_args[-1]

        args = tomap[inds] #masks[comp_inds] = False
        ix = sort_i1[args]
        is_closed = bound_args[0] == ix
        bound_args.append(ix)
        print(inds, ix)
        # masks[ix] = False

        if is_closed:
            # add new closed loopsa
            edge_arg_inds.append(bound_args)
            init_ix = np.where(masks)[0]
            # if reamin next loop
            if init_ix.size > 0:
                print(init_ix)
                bound_args = [sort_i0[init_ix[0]]]
                masks[init_ix[0]] = False
        # elif not masks[ix]:
        #     # alread traversed
        #     non_closed_args.append(bound_args)
        #     init_ix = np.where(masks)[0]
        #     if init_ix.size == 0:
        #         break
        #     print("new init", init_ix[0])
        #     if init_ix.size > 0 :
        #         bound_args = [sort_i0[init_ix[0]]]
        #         masks[init_ix[0]] = False
        elif sort_i0[tomap[ix]] != ix:
            # alread traversed
            non_closed_args.append(bound_args)
            init_ix = np.where(masks)[0]
            bound_args = list()
            if init_ix.size == 0:
                break
            print("new init", init_ix[0], len(bound_args))
            if init_ix.size > 0 :
                bound_args = [sort_i0[init_ix[0]]]
                masks[init_ix[0]] = False
        else:
            pass

            masks[ix] = False


    print([len(i) for i in edge_arg_inds])
    print([len(i) for i in non_closed_args])
    # vtk_utils.write_stl("clip.stl", target_polydata)
    debug = False
    if debug:
        show([vtk_utils.create_curve_actor(pts) for pts in [edge_pts[arg] for arg in edge_arg_inds]])
        show([vtk_utils.create_curve_actor(pts) for pts in [edge_pts[arg] for arg in non_closed_args]])
        show([vtk_utils.create_curve_actor(pts) for pts in [edge_pts[arg] for arg in non_closed_args]])
        show([vtk_utils.create_curve_actor(edge_pts[non_closed_args[-6]])])

    #     # print()
    #
    # non_continue = np.cumsum(np.diff(sort_i0, prepend=[sort_i0[0] - 1]) - 1)
    # non_continue = non_continue * -1
    # # pad_noncontinue = np.pad(non_continue, )
    # # if edge_pts.shape[0] > sort_i0.shape[0]
    # num_pad = np.maximum(edge_pts.shape[0] - sort_i0.shape[0], 0)
    # pad_noncontinue = np.pad(non_continue, [0, num_pad], 'edge')
    # non_closed_args = []
    #
    #
    # for _ in range(len(sort_i1)):
    #
    # # while np.any(masks):
    #     inds = bound_args[-1]
    #     # comp_inds = inds
    #     comp_inds = inds + pad_noncontinue[inds]
    #
    #     masks[comp_inds] = False
    #     ix = sort_i1[comp_inds]
    #     is_closed = bound_args[0] == ix
    #     # print(bound_args[-1], ix)
    #     bound_args.append(ix)
    #     print(not masks[ix])
    #     # print()
    #
    #     if is_closed:
    #         # add new closed loopsa
    #         edge_arg_inds.append(bound_args)
    #         init_ix = np.where(masks)[0]
    #         # if reamin next loop
    #         if init_ix.size > 0:
    #             # print(init_ix)
    #             bound_args = [sort_i0[init_ix[0]]]
    #     elif not masks[ix]:
    #         # alread traversed
    #         non_closed_args.append(bound_args)
    #         init_ix = np.where(masks)[0]
    #         print("new init", init_ix[0])
    #         if init_ix.size > 0 :
    #             bound_args = [sort_i0[init_ix[0]]]
    #     else:
    #
    #         pass
    closed_loop = [ix[0] == ix[-1] for ix in edge_arg_inds]
    logger = get_runtime_logger()
    logger.info("closed loop check {}".format(closed_loop))

    line_lengts_list = []
    line_pts_list = []
    for ix in edge_arg_inds:
        line_pts = edge_pts[ix]
        diff_lines = np.diff(line_pts, axis=0)
        approximate_length = np.linalg.norm(diff_lines, axis=-1).sum()
        line_pts_list.append(line_pts)
        line_lengts_list.append(approximate_length)
    logger.info("each loop(or line) length : {}".format(line_lengts_list))
    is_show = False
    if is_show:
        show([vtk_utils.create_curve_actor(pts) for pts in [edge_pts[arg] for arg in edge_arg_inds]])
        show([vtk_utils.create_curve_actor(pts) for pts in [edge_pts[arg] for arg in non_closed_args]])

        show(
            [vtk_utils.create_curve_actor(edge_pts[np.asarray(bound_args)])]
        )

        non_closed_lines = np.asarray(bound_args)
        show(
            [ *vtk_utils.create_sphere(edge_pts[np.asarray(bound_args)], .1, is_coloring_rainbow=True)]
        )

        show(
            [ *vtk_utils.create_sphere(edge_pts[np.unique(non_closed_lines)], .1, is_coloring_rainbow=True)]
        )

        curve_list = [vtk_utils.create_curve_actor(pts) for pts in line_pts_list]
        for pts, length in zip(line_pts_list, line_lengts_list):
            logger.info("length:{}".format(length))

    # only one loop and its length same with
    # reshape_edge_lines.shape[0] == len(edge_arg_inds[0])
    if return_full:
        if return_indices:
            return [edge_pts[arg] for arg in edge_arg_inds], edge_arg_inds
        else:
            return [edge_pts[arg] for arg in edge_arg_inds]
    else:
        edge_sort_inds = np.array(edge_arg_inds[0])
        edge_sort_pts = edge_pts[edge_sort_inds]
        return edge_sort_pts


def compute_tensor(polydata_or_actor):
    if isinstance(polydata_or_actor, vtk.vtkActor):
        polydata_or_actor = polydata_or_actor.GetMapper().GetInput()
    v, f = vtk_2_vf(polydata_or_actor)
    vtk_utils.compute_normal(polydata_or_actor)
    n = numpy_support.vtk_to_numpy(polydata_or_actor.GetPointData().GetNormals())
    return np.concatenate([v, n], axis=-1)



def unifrom_resampling_curve(curve_points, interval=0.1):
    """
    :param curve_points: N X 3
    :param interval: float
    :return:
    """
    curve_diff = np.diff(curve_points, axis=0)
    length = np.linalg.norm(curve_diff, axis=1)
    # concat_direction = curve_diff / np.linalg.norm(curve_diff, axis=1).reshape([-1, 1])
    cum_length = np.cumsum(length)
    cum_length_ex = np.concatenate([[0.], cum_length], axis=0)
    curve_diff_ex = np.concatenate([curve_diff, curve_diff[-1:]])

    # 균일하게 샘플링하고자하는 길이를 설정한다
    dx = interval
    num = int(np.ceil(cum_length[-1] / dx))
    # 누적 샘플링하는 값을 증가배열설정한다
    uniform_length = np.linspace(0, num - 1, num) * dx

    # pre-curve의 길이를 비교하여 가장 가까운 이웃 index(pre-next)를 계산한다
    diff_length = cum_length_ex - uniform_length.reshape([-1, 1])
    inds = np.argmin(np.abs(diff_length), axis=1)
    diff_value = diff_length[np.arange(uniform_length.shape[0]), inds]
    pre_inds = inds.copy()
    negat = diff_value > 0
    pre_inds[negat] = inds[negat] - 1
    next_inds = pre_inds + 1
    # protect index overflow
    next_inds_clip = np.clip(next_inds, 0, cum_length_ex.shape[0] - 1)
    # 이웃 간격 샘플링의 내분점 비율을 계산한다
    # ratio = (uniform_length - cum_length_ex[pre_inds]) / (cum_length_ex[next_inds] - cum_length_ex[pre_inds])
    # ratio2 = (uniform_length - cum_length_ex[pre_inds]) / (cum_length_ex[next_inds_clip] - cum_length_ex[pre_inds])
    dividor = np.where(next_inds >= cum_length_ex.shape[0], 1,
                       (cum_length_ex[next_inds_clip] - cum_length_ex[pre_inds]))
    # dividor = np.where(dividor == 0., )
    ratio = np.where( dividor > 0, (uniform_length - cum_length_ex[pre_inds]) / dividor, 1.0)

    # 내분점 비율을 반영하여 포인트를 다시 계산한다
    uniform_curve = curve_points[pre_inds] + curve_diff_ex[pre_inds] * ratio.reshape([-1, 1])
    return uniform_curve


def fill_hole(polydata, holesize=50):
    fillHolesFilter = vtk.vtkFillHolesFilter()
    fillHolesFilter.SetInputData(polydata)
    fillHolesFilter.SetHoleSize(holesize)
    fillHolesFilter.Update()

    filled_data = fillHolesFilter.GetOutput()

    v0, f0 = vtk_2_vf(polydata)
    v1, f1 = vtk_2_vf(filled_data)
    if f0.shape[0] == f1.shape[0]:
        return filled_data
    else:

        np.testing.assert_equal(v0, v1)
        np.testing.assert_equal(f0, f1[:f0.shape[0]])
        f2 = f1.copy()
        f2[f0.shape[0]:, 1:] = f1[f0.shape[0]:, 1:][:, ::-1]
        filled_recon = recon(v1, f2)
        return filled_recon


def reconstruct_surface(pts, normals):
    pts_actor = vtk_utils.create_points_actor(pts)
    pts_pd = pts_actor.GetMapper().GetInput()
    normals_data = numpy_support.numpy_to_vtk(normals)
    pts_pd.GetPointData().SetNormals(normals_data)

    surf = vtk.vtkSurfaceReconstructionFilter()
    surf.SetInputData(pts_pd)
    cf = vtk.vtkContourFilter()
    cf.SetInputConnection(surf.GetOutputPort())
    cf.SetValue(0, 0.0)

    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(cf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    return reverse.GetOutput()


def compute_curvature(polydata_or_actor):
    if isinstance(polydata_or_actor, vtk.vtkActor):
        polydata = polydata_or_actor.GetMapper().GetInput()
    elif isinstance(polydata_or_actor, vtk.vtkPolyData):
        polydata = polydata_or_actor
    curvaturesFilter = vtk.vtkCurvatures()
    curvaturesFilter.SetInputData(polydata)
    curvaturesFilter.SetCurvatureTypeToMinimum()
    curvaturesFilter.SetCurvatureTypeToMaximum()
    curvaturesFilter.SetCurvatureTypeToGaussian()
    # curvaturesFilter.SetCurvatureTypeToMean()
    curvaturesFilter.Update()

    return curvaturesFilter.GetOutput()


def find_intersection(p1, p2, locator: vtk.vtkCellLocator, return_id=False):
    """
    https://vtk.org/Wiki/VTK/Examples/Python/DataManipulation/LineOnMesh
    """
    # assert isinstance(p1, )
    t = vtk.mutable(0)
    subId = vtk.mutable(0)
    tol = 1e-9
    pose = [0, 0, 0]
    pcoords = [0, 0, 0]
    cellId = vtk.mutable(0)
    isexist = locator.IntersectWithLine(p1, p2, tol, t, pose, pcoords, subId, cellId)
    if return_id:
        return isexist, pose, cellId.get()
    else:
        return isexist, pose


def get_cellnormals(polydata:vtk.vtkPolyData):
    normal_gen = vtk.vtkPolyDataNormals()
    normal_gen.SetInputData(polydata)
    normal_gen.ComputePointNormalsOn()
    # if splitting is False:
    normal_gen.SplittingOff()

    normal_gen.ComputeCellNormalsOn()
    normal_gen.Update()
    normal_polydata = normal_gen.GetOutput()
    # aa = p2a(normal_polydata)
    return numpy_support.vtk_to_numpy(normal_polydata.GetCellData().GetNormals())


def get_normals(polydata:vtk.vtkPolyData):
    vtk_utils.compute_normal(polydata)
    return numpy_support.vtk_to_numpy(polydata.GetPointData().GetNormals())


def compute_signed_distances(targetpd:vtk.vtkPolyData, points:np.ndarray):
    impDist = vtk.vtkImplicitPolyDataDistance()
    impDist.SetInput(targetpd)
    distances = []
    neihgbor_pose = []
    for pt in points:
        pose = [0, 0, 0]
        dist = impDist.EvaluateFunctionAndGetClosestPoint(pt, pose)
        distances.append(dist)
        neihgbor_pose.append(pose)
    return np.array(distances), np.array(neihgbor_pose)

def compare_signed_distances(srcpd, tarpd, show=True):
    dist_list = []
    close_pose = []
    impDist = vtk.vtkImplicitPolyDataDistance()
    impDist.SetInput(tarpd)
    srcpts, _ = vtk_2_vf(srcpd)
    for upt in srcpts:
        pose = [0, 0, 0]
        dist = impDist.EvaluateFunctionAndGetClosestPoint(upt, pose)
        dist_list.append(dist)
        close_pose.append(pose)
    dist_list = np.asarray(dist_list)
    return _show_target_with_scalars(srcpd, dist_list, show=show)


def final_trimming(srcpd, tarpd):
    trim_pose = []
    dist_list = []
    impDist = vtk.vtkImplicitPolyDataDistance()
    impDist.SetInput(tarpd)
    srcpts, src_polys = vtk_2_vf(srcpd)

    for upt in srcpts:
        pose = [0, 0, 0]
        dist = impDist.EvaluateFunctionAndGetClosestPoint(upt, pose)
        dist_list.append(dist)
        trim_pose.append(pose)
    trim_pose = np.array(trim_pose)

    dist_list = np.asarray(dist_list)
    trim_pts = srcpts.copy()
    trim_pts[dist_list < 0] = trim_pose[dist_list < 0]
    return recon(trim_pts, src_polys)


def _show_target_with_scalars(pd:vtk.vtkPolyData, numpy_scalars=None, show=True, inrange=None):
    return show_target_with_scalars(pd, numpy_scalars=numpy_scalars, show=show, inrange=inrange)


def get_roi_polydata_among_polydata(polydata, margin_loops):
    loop_list, _ = compute_boundary_pts_array(polydata, return_full=True)
    neighbor = NearestNeighbors()
    neighbor.fit(margin_loops)
    min_dist = 1e10

    for i, loops in enumerate(loop_list):
        dist, _ = neighbor.kneighbors(loops, n_neighbors=1, return_distance=True)
        err = dist.mean()
        if err < min_dist:
            neighbor_loops = loops
            min_dist = err

    inds = get_vert_indices(polydata, neighbor_loops)
    v0, f0 = vtk_2_vf(polydata)
    mask = np.zeros([v0.shape[0]], dtype=np.bool)
    mask[inds] = True
    sel_mask = select_region_growing(f0, mask)
    v1, f1, _ = extract_polydata(v0, f0, sel_mask, np.all)
    return recon(v1, f1)


@timefn2
def select_region_growing(ff, seedmask):
    # show([p2a(test_clip)])

    # mask = np.zeros([v1.shape[0]], dtype=np.bool)
    # mask[inds] = True
    selmask = seedmask.copy()
    cnt = 0
    while seedmask.shape[0] / 100:
        cnt = cnt + 1
        omask = expand_vert_mask(ff, selmask, iter=1)
        if np.logical_xor(omask, selmask).sum() == 0:
            break
        selmask = omask

    return selmask


def show_target_with_scalars(pd_or_numpy:vtk.vtkPolyData, numpy_scalars=None, show=True, inrange=None, input_actors=[]):
    if isinstance(pd_or_numpy, np.ndarray):
        pts_actor = vtk_utils.create_points_actor(pd_or_numpy)
        pd = pts_actor.GetMapper().GetInput()
    else:
        # _actor_2_vf()
        pd = to_polydata(pd_or_numpy)

    copy_pd = vtk.vtkPolyData()
    copy_pd.DeepCopy(pd)
    used_cell_scalars = False
    if numpy_scalars is not None:
        if numpy_scalars.dtype == 'bool':
            numpy_scalars = numpy_scalars.astype('float')
        if numpy_scalars.shape[0] == pd.GetNumberOfPoints():
            vtk_scalars = numpy_support.numpy_to_vtk(numpy_scalars)
            copy_pd.GetPointData().SetScalars(vtk_scalars)
        elif numpy_scalars.shape[0] == pd.GetNumberOfPolys():
            vtk_scalars = numpy_support.numpy_to_vtk(numpy_scalars, array_type=vtk.VTK_FLOAT)
            copy_pd.GetCellData().SetScalars(vtk_scalars)
            used_cell_scalars = True
        else:
            raise ValueError("scalar shape {}".format(numpy_scalars.shape))

    elif copy_pd.GetPointData().GetScalars() is not None:
        vtk_scalars = copy_pd.GetPointData().GetScalars()
        assert vtk_scalars, "empty scalars and invalid scalar array"

        numpy_scalars = numpy_support.vtk_to_numpy(vtk_scalars)
    elif copy_pd.GetCellData().GetScalars() is not None:
        vtk_scalars = copy_pd.GetCellData().GetScalars()
        numpy_scalars = numpy_support.vtk_to_numpy(vtk_scalars)

    if inrange is None:
        dmin, dmax = numpy_scalars.min(), numpy_scalars.max()
    else:
        dmin, dmax = inrange

    copy_actor = vtk_utils.polydata2actor(copy_pd)
    copy_actor.GetMapper().SetScalarRange(dmin, dmax)
    lookuptable = copy_actor.GetMapper().GetLookupTable()
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(lookuptable)
    scalarBar.SetTitle("Prob")
    scalarBar.SetNumberOfLabels(5)
    if copy_pd.GetCellData().GetScalars() and used_cell_scalars:
        copy_actor.GetMapper().SetScalarModeToUseCellData()
    items = [copy_actor, scalarBar]
    if show:
        vtk_utils.show_actors([copy_actor, scalarBar, *input_actors])
    return items



def clipping_loop(target_polydata, loop_points: np.ndarray, attaching=True, selection="small",
                  smoothing_attached=0, return_selected=False):
    assert selection in ["large", "small"]
    tv, _ = vtk_2_vf(target_polydata)

    attatch_pts = loop_points
    if attaching :
        POINT2FACE = 0
        POINT2POINT = 1
        METHOD = POINT2FACE
        if METHOD == POINT2POINT:
            neighbor = NearestNeighbors()
            neighbor.fit(tv)
            dist, inds = neighbor.kneighbors(loop_points, n_neighbors=1, return_distance=True)

            # dist, attatch_pts = compute_signed_distances(target_polydata, loop_points)


            # sd, sp = compute_signed_distances(target_polydata, loop_points)
            inds = inds[:, 0]
            attatch_pts = tv[inds]
        elif METHOD == POINT2FACE:
            if smoothing_attached > 0:
                attatch_pts = loop_points
                for _ in range(smoothing_attached):
                    _, attatch_pts = compute_signed_distances(target_polydata, attatch_pts)

                    attatch_pts = smoothing_edge(attatch_pts)
                    # _, attatch_pts = compute_signed_distances(target_polydata, attatch_pts)
            # unifrom_resampling_curve()

                #
                # for _ in range(smoothing_attached):
                #     attatch_pts = smoothing_edge(attatch_pts)
                #
                # _, attatch_pts = compute_signed_distances(target_polydata, attatch_pts)
        else:
            raise ValueError

        # clip = vtk.vtkClipPolyData()
        # clip.SetInputConnection(loop.GetOutputPort())
        # clip.Update()
        pass

    # 너무 촘촘하면 clipping이 안될때가 있는것 같다. 그리서 인위적으로 subsampling 처리한다.

    dt = 1

    def _clipping(_points)->vtk.vtkSelectPolyData:
        tar_curve = vtk_utils.create_curve_actor(_points)
        # tar_curve.GetProperty().SetLineWidth(5)
        # show([p2a(target_polydata), tar_curve])
        margin_polydata = tar_curve.GetMapper().GetInput()

        loop = vtk.vtkSelectPolyData()
        loop.SetInputData(target_polydata)
        loop.SetLoop(margin_polydata.GetPoints())
        loop.GenerateSelectionScalarsOn()
        # loop.SetSelectionModeToSmallestRegion() # negative scalars inside
        if selection == "small":
            loop.SetSelectionModeToLargestRegion()
        else:
            loop.SetSelectionModeToSmallestRegion()
        loop.Update()
        return loop


    while dt < 7:

        loop = _clipping(attatch_pts[::dt])
        #
        # tar_curve = vtk_utils.create_curve_actor(attatch_pts[::dt])
        # # tar_curve.GetProperty().SetLineWidth(5)
        # # show([p2a(target_polydata), tar_curve])
        # margin_polydata = tar_curve.GetMapper().GetInput()
        #
        # loop = vtk.vtkSelectPolyData()
        # loop.SetInputData(target_polydata)
        # loop.SetLoop(margin_polydata.GetPoints())
        # loop.GenerateSelectionScalarsOn()
        # # loop.SetSelectionModeToSmallestRegion() # negative scalars inside
        # if selection == "small":
        #     loop.SetSelectionModeToLargestRegion()
        # else:
        #     loop.SetSelectionModeToSmallestRegion()
        # loop.Update()

        success = loop.GetOutput() and loop.GetOutput().GetNumberOfPoints() > 0
        if success:
            logger = get_runtime_logger()
            logger.info("clipping successfully")
            break
        dt += 1
    if not success:
        # sample = attatch_pts[::2]
        resample = unifrom_resampling_curve(attatch_pts, 0.05)
        _, resample = compute_signed_distances(target_polydata, resample)

        with open("debug.json", "w") as f:
            json.dump({
                "points": resample.tolist()
            }, f)

        vtk_utils.write_stl("debug.stl", target_polydata)
        dt = 1
        while dt < 5:

            loop = _clipping(resample[::dt])
            success = loop.GetOutput() and loop.GetOutput().GetNumberOfPoints() > 0

            if success:
                logger = get_runtime_logger()
                logger.info('successfully clipping')
                break
            dt += 1

        if not success:
            vtk_utils.show_actors([
                vtk_utils.polydata2actor(target_polydata),
                *vtk_utils.create_sphere(resample[::dt], 0.09)
            ])



    assert success, "failing clipping"


    # loop.GetErrorCode()


    clip = vtk.vtkClipPolyData()
    clip.SetInputData(loop.GetOutput())
    # clip.SetInputConnection(loop.GetOutputPort())
    clip.Update()
    if not return_selected:
        return clip.GetOutput()

    else:
        return clip.GetOutput(), loop.GetOutput()


def stiching_polydata(pd1, pd2):
    def extractPoints(source):
        """
        Return points from a polydata as a list of tuples.
        """
        points = source.GetPoints()
        indices = range(points.GetNumberOfPoints())
        pointAccessor = lambda i: points.GetPoint(i)
        return list(map(pointAccessor, indices))

    def findClosestSurfacePoint(source, point):
        # source = ensurePolyData(source)
        locator = vtk.vtkKdTreePointLocator()
        locator.SetDataSet(source)
        locator.BuildLocator()

        pId = locator.FindClosestPoint(point)
        return pId


    bounds1 = compute_boundary_pts_array(pd1)[:-1]
    temp_bounds2 = compute_boundary_pts_array(pd2)[:-1]

    # anti porcessing
    neighbor = NearestNeighbors()
    neighbor.fit(bounds1)
    inds = neighbor.kneighbors(temp_bounds2, return_distance=False)[:, 0]
    # circular roate args
    diff_inds = np.diff(inds)
    ix = np.abs(diff_inds).argmax()
    # anti direction
    if np.sign(diff_inds[ix]) > 0:
        pass
        bounds2 = temp_bounds2
    else:
        # same direction
        bounds2 = temp_bounds2[::-1]


    # show([
    #     *vtk_utils.create_sphere(bounds1, size=0.1, is_coloring_rainbow=True),
    #     *vtk_utils.create_sphere(bounds2, size=0.1, is_coloring_rainbow=True),
    # ])
    # Extract points along the edge line (in correct order).
    # The following further assumes that the polyline has the
    # same orientation (clockwise or counterclockwise).
    edge1 = vtk_utils.create_curve_actor(bounds1).GetMapper().GetInput()
    edge2 = vtk_utils.create_curve_actor(bounds2).GetMapper().GetInput()


    points1 = extractPoints(edge1)
    points2 = extractPoints(edge2)


    # points1 = bounds1
    # points2 = bounds2
    n1 = len(points1)
    n2 = len(points2)

    # Prepare result containers.
    # Variable points concatenates points1 and points2.
    # Note: all indices refer to this targert container!
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    points.SetNumberOfPoints(n1+n2)
    for i, p1 in enumerate(points1):
        points.SetPoint(i, p1)
    for i, p2 in enumerate(points2):
        points.SetPoint(i+n1, p2)

    # The following code stitches the curves edge1 with (points1) and
    # edge2 (with points2) together based on a simple growing scheme.

    # Pick a first stitch between points1[0] and its closest neighbor
    # of points2.
    i1Start = 0
    i2Start = findClosestSurfacePoint(edge2, points1[i1Start])
    i2Start += n1 # offset to reach the points2

    # Initialize
    i1 = i1Start
    i2 = i2Start
    p1 = np.asarray(points.GetPoint(i1))
    p2 = np.asarray(points.GetPoint(i2))
    mask = np.zeros(n1+n2, dtype=bool)
    count = 0
    while not np.all(mask):
        count += 1
        print(count, i1, i2)
        i1Candidate = (i1+1)%n1
        i2Candidate = (i2-1-n1)%n2+n1
        p1Candidate = np.asarray(points.GetPoint(i1Candidate))
        p2Candidate = np.asarray(points.GetPoint(i2Candidate))
        diffEdge12C = np.linalg.norm(p1-p2Candidate)
        diffEdge21C = np.linalg.norm(p2-p1Candidate)

        print(count, i1Candidate, i2Candidate, diffEdge12C, diffEdge21C, i1, i2)
        mask[i1] = True
        mask[i2] = True
        if mask[i1Candidate]:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0,i1)
            triangle.GetPointIds().SetId(1,i2)
            triangle.GetPointIds().SetId(2,i2Candidate)
            cells.InsertNextCell(triangle)
            i2 = i2Candidate
            p2 = p2Candidate
        elif mask[i2Candidate]:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0,i1)
            triangle.GetPointIds().SetId(1,i2)
            triangle.GetPointIds().SetId(2,i1Candidate)
            cells.InsertNextCell(triangle)
            i1 = i1Candidate
            p1 = p1Candidate
        elif diffEdge12C < diffEdge21C:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0,i1)
            triangle.GetPointIds().SetId(1,i2)
            triangle.GetPointIds().SetId(2,i2Candidate)
            cells.InsertNextCell(triangle)
            i2 = i2Candidate
            p2 = p2Candidate
        else:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0,i1)
            triangle.GetPointIds().SetId(1,i2)
            triangle.GetPointIds().SetId(2,i1Candidate)
            cells.InsertNextCell(triangle)
            i1 = i1Candidate
            p1 = p1Candidate

    # Add the last triangle.


    i1Candidate = (i1+1)%n1
    i2Candidate = (i2-1-n1)%n2+n1
    if (i1Candidate <= i1Start) or (i2Candidate <= i2Start):
        if i1Candidate <= i1Start:
            iC = i1Candidate
        else:
            iC = i2Candidate
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0,i1)
        triangle.GetPointIds().SetId(1,i2)
        triangle.GetPointIds().SetId(2,iC)
        cells.InsertNextCell(triangle)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    poly.BuildLinks()

    # show([p2a(poly)])
    #
    # show([p2a(vtk_utils.compute_boundary_edge(poly)),
    #       *vtk_utils.create_sphere(bounds1, 0.05, is_coloring_rainbow=True),
    #       *vtk_utils.create_sphere(bounds2, 0.05, is_coloring_rainbow=True),
    #     p2a(poly)
    #       ])
    # show([p2a(vtk_utils.compute_boundary_edge(poly)),])
    v1, f1 = vtk_2_vf(pd1)
    v2, f2 = vtk_2_vf(pd2)

    v12 = np.concatenate([v1, v2], axis=0)
    f22 = f2.copy()
    f22[:, 1:] = f2[:, 1:] + v1.shape[0]
    f12 = np.concatenate([f1, f22], axis=0)
    # show([p2a(recon(v12, f12))])

    inds1 = get_vert_indices(pd1, bounds1)
    inds2 = get_vert_indices(pd2, bounds2)
    inds2_ = inds2 + v1.shape[0]
    tomap = np.concatenate([inds1, inds2_])


    cell_array = numpy_support.vtk_to_numpy(cells.GetData())
    cell_array_reshape = cell_array.reshape([-1, 4])
    cell_array12 = np.concatenate([cell_array_reshape[:, :1], tomap[cell_array_reshape[:, 1:]]], axis=-1)

    f12_c = np.concatenate([f1, f22, cell_array12], axis=0)
    merge_pd = recon(v12, f12_c)
    return merge_pd



def get_closest_cellid(locator:vtk.vtkCellLocator, points):
    cellids = []
    dists = []
    close_pose = []
    for pt in points:
        id1 = vtk.mutable(0)
        id2 = vtk.mutable(0)
        dist = vtk.mutable(0.0)
        outPoint = [0, 0, 0]

        isexist = locator.FindClosestPoint(pt, outPoint, id1, id2, dist)
        cellids.append(id1.get())
        dists.append(dist.get())
        close_pose.append(outPoint)
    return np.asarray(cellids), np.asarray(dists), np.asarray(close_pose)


def split_mesh(polydata)->List[vtk.vtkPolyData]:
    vin, fin = vtk_2_vf(polydata)

    adj = igl.adjacency_matrix(fin[:, 1:])
    num_split, indices, _ = igl.connected_components(adj)

    split_polydatas = []
    for i in range(num_split):
        print("splting...{}".format(i))
        v_split, f_split, _= extract_polydata(vin, fin, indices == i, np.all, )
        poly_split = vtk_utils.reconstruct_polydata(v_split, f_split)
        split_polydatas.append(poly_split)
    return split_polydatas