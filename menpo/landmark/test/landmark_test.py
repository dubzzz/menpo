import numpy as np
from numpy.testing import assert_allclose

from menpo.landmark import LandmarkGroup, LandmarkManager
from menpo.shape import PointCloud
from menpo.testing import is_same_array


def test_LandmarkGroup_copy_true():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict)
    assert (not is_same_array(lgroup._pointcloud.points, points))
    assert (lgroup._labels_to_masks is not mask_dict)
    assert (lgroup._pointcloud is not pcloud)


def test_LandmarkGroup_copy_false():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)
    assert (is_same_array(lgroup._pointcloud.points, points))
    assert (lgroup._labels_to_masks is mask_dict)
    assert (lgroup._pointcloud is pcloud)


def test_LandmarkManager_set_landmarkgroup():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)

    man = LandmarkManager(target)
    man['test_set'] = lgroup
    assert (not is_same_array(man['test_set']._pointcloud.points,
                              lgroup._pointcloud.points))
    assert (man['test_set'] is not lgroup._labels_to_masks)
    assert (man['test_set']._target is target)


def test_LandmarkManager_set_pointcloud():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points, copy=True)

    man = LandmarkManager(target)
    man['test_set'] = pcloud

    lgroup = man['test_set']
    assert (lgroup._target is target)
    assert (lgroup._pointcloud is not pcloud)
    assert_allclose(lgroup._labels_to_masks['all'], np.ones(10, dtype=np.bool))


def test_landmarkgroup_copy_method():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    lgroup = LandmarkGroup(target, 'label', pcloud, mask_dict, copy=False)
    lgroup_copy = lgroup.copy()

    assert (not is_same_array(lgroup_copy._pointcloud.points,
                              lgroup._pointcloud.points))
    assert (lgroup_copy._target is lgroup._target)
    # Check the mask dictionary is deepcopied properly
    assert (lgroup._labels_to_masks is not lgroup_copy._labels_to_masks)
    masks = zip(lgroup_copy._labels_to_masks.values(),
                lgroup._labels_to_masks.values())
    for ms in masks:
        assert (ms[0] is not ms[1])


def test_LandmarkManager_copy_method():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points, copy=True)

    man = LandmarkManager(target)
    man['test_set'] = pcloud
    man_copy = man.copy()

    assert (man_copy._target is man._target)
    assert (man_copy['test_set'] is not man['test_set'])
    assert (not is_same_array(man_copy['test_set'].lms.points,
                              man['test_set'].lms.points))