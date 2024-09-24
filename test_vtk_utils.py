import glob

from tools import vtk_utils
from commons import get_runtime_logger
import os
import numpy as np

RESOURCE_PATH = './assets'
DCM_DATA_PATH = 'D:/dataset/CT_SCAN_RAW_DATA/김미란/김미란_2022.10/36843_221021143823/CTData'

def main():
    print('abc show me the money')



def proof():
    pass

def test():
    pass





def test_func121212121():
    pass



def abc():
    pass

def defdef():
    pass



def test_dcm_read():
    print('test_dcm_read')
    print('test_dcm_read')
    print('test_dcm_read')
    assert os.path.exists(DCM_DATA_PATH)
    # path = 'D:/dataset/TeethSegmentation/marker_set(only2class)/sample1/17/CT_김명래 ct'
    vol, sp = vtk_utils.read_dicom(DCM_DATA_PATH)
    vtk_utils.show_actors([vol])
    vtk_utils.show_actors([vol, vtk_utils.get_axes(100)])


def test_show_points_actor():
    x = np.random.randn(10, 3)
    vtk_utils.show_actors([x])


def test_compares_dicom_and_vtkfile():
    assert os.path.exists(DCM_DATA_PATH), 'emtpy dicom file'

    # dicom_path = os.path.join(DCM_DATA_PATH)
    itk_files = glob.glob(os.path.join(DCM_DATA_PATH, '../*.vtk'))
    assert len(itk_files) == 1, 'emtpyh vtk(itk exported) file'
    vol, sp = vtk_utils.read_dicom(DCM_DATA_PATH, normalize=True)

    itkfile = itk_files[0]
    seg = vtk_utils.read_vtk_itksnapped(itkfile)
    vtk_utils.show_actors([
        vol,
        seg
    ])


def test_show_target_with_scalars():
    # filename = 'D:/dataset/scanSegmentation/all/202307_complete/5. 21-26 임플란트크라운 01-15/AntagonistScan.stl'
    filename = os.path.join(RESOURCE_PATH, 'lower.stl')

    assert os.path.exists(filename), 'empty-file'
    poly = vtk_utils.read_stl(filename)
    vin, fin = vtk_utils.vtk_2_vf(poly)

    colors_table = {
        -10. : (0, 0, 1),
        5.: (0, 1, 0),
        0.: (1, 1, 0),
        3: (0, 0, 1),
        10: (0, 1, 1)
    }
    scalars = vin[:, 0]
    # test without color_table
    vtk_utils.show_target_with_scalars(poly, scalars)

    vtk_utils.show_target_with_scalars(poly, scalars, color_tables=colors_table)
    # v= vtk_utils.numpy_support.vtk_to_numpy(poly)


def test_cutter():
    # filename = 'D:/dataset/scanSegmentation/all/202307_complete/5. 21-26 임플란트크라운 01-15/AntagonistScan.stl'
    filename = os.path.join(RESOURCE_PATH, 'lower.stl')
    assert os.path.exists(filename)
    vtk = vtk_utils.vtk
    vtk.vtkObject.SetGlobalWarningDisplay(False)
    poly = vtk_utils.read_stl(filename)

    cut_lines = vtk_utils.cutter_polydata_by_plane(poly, (0, 0, 0), (0, 0, 1))

    vtk_utils.show_actors([
        cut_lines
    ])
    # change color
    vtk_utils.polydata_coloring(cut_lines, (0, 1, 0))

    vtk_utils.show_actors([
        cut_lines,
        poly
    ])


def test_point_pair_and_sphere():
    x = np.random.randn(100, 3)
    y = x + np.random.uniform(-10, 10, [x.shape[0], 3])

    vtk_utils.show_actors([
        vtk_utils.point_pair(x, y),
        *vtk_utils.create_sphere(x, 0.2, is_coloring_rainbow=False, color=(0, 1, 0)),
        *vtk_utils.create_sphere(y, 0.2, is_coloring_rainbow=False, color=(1, 1, 0)),
    ])


def test_draw_curve():
    x = np.linspace(-10, 10, 100)
    y = 2 * (x-1)**2 + 10
    points = np.stack([x, y, np.zeros_like(x)], axis=-1)
    curve = vtk_utils.create_curve_actor(points)
    vtk_utils.show_actors([curve])

    vtk_utils.show_actors([curve, vtk_utils.create_points_actor(curve)])


def test_capture_image():

    shape = (128, ) * 3
    x = np.random.uniform(0, 1, shape)
    items = [x]
    save_path = 'd:/temp/capture'
    vtk_utils.capture_image(items, save_path)


def test_split_windows():
    shape = (128, ) * 3
    x = np.random.uniform(0, 1, shape)
    y = np.random.uniform(0, 1, shape)
    items1 = [x]
    items2 = [y]
    # np.random.randint()
    # save_path = 'd:/temp/capture'
    # next_func = lambda: ([np.random.uniform(0, 1, np.random.randint(50, 150, [3]))], [np.random.uniform(0, 1, np.random.randint(50, 150, [3]))])
    def double_next():
        # make sure tuple[list, list]
        return (
            [np.random.uniform(0, 1, np.random.randint(50, 150, [3]))],
            [np.random.uniform(0, 1, np.random.randint(50, 150, [3]))]
        )

    def single_next():
        return (
            [np.random.uniform(0, 1, np.random.randint(50, 150, [3]))],
        )


    vtk_utils.split_show(items1, items2, next_func=double_next, render_colors=[(1, 0, 0), (0, 1, 0)])

    vtk_utils.show_actors(items1,
                          next_func=single_next)


def test_write_ply():
    filename = 'D:/dataset/scanSegmentation/2020/04959_171123110704/LowerJawScan.stl'
    poly = vtk_utils.read_stl(filename)
    # vtk_utils.vtk_2_vf()
    v, _ = vtk_utils.vtk_2_vf(poly)
    actors = vtk_utils.show_target_with_scalars(poly, v[:, 0], show=True)
    act = actors[0]
    vtk_utils.write_ply('test.ply', act)
    filename = ''
    # vtk_utils.write_ply()
    # vtk.vtkPLYWriter()


def all_test():
    vtk_utils.vtk.vtkObject.SetGlobalWarningDisplay(False)
    logger = get_runtime_logger()
    for k, v in globals().items():
        print(k, v, callable(v))
        if callable(v) and k.startswith('test'):
            try:
                v()
            except Exception as e:
                logger.error(e.args)



def test_float_volume():
    pass
    # vol, sp = vtk_utils.read_dicom(DCM_DATA_PATH)
    from tools import diskmanager, dicom_read_wrapper

    dirs = diskmanager.deep_search_directory('D:/dataset/CT_SCAN_RAW_DATA/김미란/김미란_2022.10', exts=['.dcm'], filter_func=lambda x: len(x) > 100)
    # vtk_utils.show_actors([vol])
    vtkorder = False
    cam_direction = (0.3, -1.0, 0.3) if vtkorder else (0.3, 1., -0.2)
    view_up = (0., 0, -1.0) if vtkorder else (0, 0, 1.)
    for _ in range(5):
        path = np.random.choice(dirs)
        vol, sp = dicom_read_wrapper.read_dicom_wrapper(path, vtkorder=vtkorder)
        vtk_utils.show_actors([vtk_utils.volume_coloring(vol, pallete='normal', threshold=0.3)],
                              image_save=True, view_up=view_up, cam_direction=cam_direction)

    # vol = np.where(vol > .3, vol, np.zeros)
    # clip_vol = np.clip(vol, 0.5, 1.0)
    # vol_int = (vol * 10).astype(np.int32)
    # vtk_utils.show_actors([vtk_utils.volume_coloring(vol_int, pallete='rainbow')])

    # vtk_utils.show_actors([vtk_utils.volume_coloring((vol > .5).astype(np.int32), pallete='rainbow')])
    # vtk_utils.show_actors([vol>.2])
    # vtk_utils.show_actors([clip_vol])
    # vtk_utils.show_actors([vtk_utils.volume_coloring(vol, pallete='normal', threshold=0.3)])

def test_integer_volume():
    pass


def test_compose_volume():
    pass


def test_split_volume():
    pass


def test_add_actors_as_dict_list():

    items1 = {
        '10': np.random.randn(10, 3),
        '11': np.random.randn(10, 3),
    }

    items2 = [
        # np.random.randn(30, 3),
        # np.random.randn(30, 3),
        np.random.randn(3),
        np.random.randn(3),
        np.random.randn(3),
        np.random.randn(3),
    ]

    vtk_utils.show_actors([
        items2
    ])



def test_point_cloud_scalars():
    x = np.random.randn(5000, 3)
    dists = np.linalg.norm(x, axis=-1)
    scalar_range = (.5, 2.0)
    delta = scalar_range[1] - scalar_range[0]
    norm_dists = (dists - dists.min()) / (dists.max() - dists.min()) * delta + scalar_range[0]
    vtk_utils.show_pc(x, norm_dists)

def save_image():
    filename = 'temp.png'
    x = np.random.randn(1000, 3)
    vtk_utils.show_actors([x], image_save=True, show=False, savename=filename)

if __name__=='__main__':
    save_image()
    # all_test()
    # test_cpature_image()
    # test_capture_image()
    # test_split_windows()
    # test_write_ply()
    # test_float_volume()
    # test_add_actors_as_dict_list()
    # test_point_cloud_scalars()
