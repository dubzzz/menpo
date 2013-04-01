import numpy as np
from mayavi import mlab
from collections import OrderedDict
from pybug.visualization import PointCloudViewer3d, LabelViewer3d


class FieldError(Exception):
    pass

class PointFieldError(FieldError):
    pass

class SpatialDataConstructionError(Exception):
    pass


class SpatialData(object):
    """ Abstract representation of a n-dimentional piece of spatial data.
    This could be simply be a set of vectors in an n-dimentional space,
    or a structed surface or mesh. At this level of abstraction we only
    define basic metadata that can be attached to all kinds of spatial
    data
    """
    def __init__(self):
        pass


class PointCloud(SpatialData):
    """n-dimensional point cloud. Can be coerced to a PCL Point Cloud Object
    for using their library methods (TODO). Handles the addition of spatial
    metadata (most commonly landmarks) by storing all such 'metapoints'
    (points which aren't part of the shape) and normal points together into
    a joint field (_allpoints). This is masked from the end user by the use
    of properties.
    """
    def __init__(self, points, n_metapoints=0):
        SpatialData.__init__(self)
        self.n_points, n_dims  = points.shape
        self.n_metapoints = n_metapoints
        self._allpoints = np.empty([self.n_points + self.n_metapoints, n_dims])
        self._allpoints[:self.n_points] = points
        self.pointfields = {}

    @property
    def points(self):
        return self._allpoints[:self.n_points]

    @property
    def metapoints(self):
        """Points which are solely for metadata. Are guaranteed to be
        transformed in exactly the same way that points are. Useful for
        storing explicit landmarks (landmarks that have coordinates and
        don't simply reference exisiting points).
        """
        return self._allpoints[self.n_points:]

    @property
    def n_dims(self):
        return self.points.shape[1]

    def __str__(self):
        message = str(type(self)) + ': n_points: ' + `self.n_points`  \
                + ', n_dims: ' + `self.n_dims`
        if len(self.pointfields) != 0:
            message += '\n  pointfields:'
            for k,v in self.pointfields.iteritems():
                try:
                    field_dim = v.shape[1]
                except IndexError:
                    field_dim = 1
                message += '\n    ' + str(k) + '(' + str(field_dim) + 'D)'
        return message


    def add_pointfield(self, name, field):
        """Add another set of field values (of arbitrary dimention) to each
        point.
        """
        if field.shape[0] != self.n_points:
            raise PointFieldError("Trying to add a field with " +
                    `field.shape[0]` + " values (need one field value per " +
                    "point => " + `self.n_points` + " values required")
        else:
            self.pointfields[name] = field

    def view(self):
            print 'arbitrary dimensional PointCloud rendering is not supported.'

class PointCloud3d(PointCloud):

    def __init__(self, points, n_metapoints=0):
        PointCloud.__init__(self, points, n_metapoints)
        if self.n_dims != 3:
            raise SpatialDataConstructionError(
                    'Trying to build a 3D Point Cloud with from ' +
                    str(self.n_dims) + ' data')

    def view(self):
        viewer = PointCloudViewer3d(self.points)
        return viewer.view()

    def attach_landmarks(self, landmarks_dict):
        self.landmarks = Landmarks(self, landmarks_dict)

class Landmarks(object):
    """Class for storing and manipulating Landmarks associated with a shape.
    Landmarks index into the points and metapoints of the associated 
    PointCloud. Landmarks which are expicitly given as coordinates would
    be entirely constructed from metapoints, whereas point indexed landmarks
    would be composed entirely of points. This class can handle any arbitrary
    mixture of the two.
    """
    def __init__(self, pointcloud, landmarks_dict):
        """ pointcloud - the shape whose these landmarks apply to
        landmark_dict - keys - landmark classes (e.g. 'mouth')
                        values - ordered list of landmark indices into 
                        pointcloud._allpoints
        """
        self.pc = pointcloud
        # indexes are the indexes into the points and metapoints of self.pc.
        # note that the labels are always sorted when stored.
        self.indexes = OrderedDict(sorted(landmarks_dict.iteritems()))

    def all(self, withlabels=False, indexes=False, numbered=False):
        """return all the landmark indexes. The order is always guaranteed to
        be the same for a given landmark configuration - specifically, the 
        points will be returned by sorted label, and always in the order that 
        each point the landmark was construted in.
        """
        all_lm = []
        labels = []
        for k, v in self.indexes.iteritems():
            if indexes:
                all_lm += v
            else:
                all_lm += list(self.pc._allpoints[v])
            newlabels = [k] * len(v)
            if numbered:
                newlabels = [x + '_' + str(i) for i, x in enumerate(newlabels)]
            labels += newlabels
        if withlabels:
           return np.array(all_lm), labels
        return np.array(all_lm)

    def __getitem__(self, label):
        return self.pc._allpoints[self.indexes[label]]

    def view(self):
        """ View all landmarks on the current shape, using the default
        shape view method.
        """
        lms, labels = self.all(withlabels=True, numbered=True)
        pcviewer = self.pc.view()
        pointviewer = PointCloudViewer3d(lms)
        pointviewer.view(onviewer=pcviewer)
        lmviewer = LabelViewer3d(lms, labels, offset=np.array([0,16,0]))
        lmviewer.view(onviewer=pcviewer)
        return lmviewer


    @property
    def config(self):
        """A nested tuple specifying the precise nature of the landmarks
        (labels, and n_points per label). Allows for comparison of Landmarks
        to see if they are likely describing the same shape.
        """
        return tuple((k,len(v)) for k,v in self.indexes.iteritems())
       
