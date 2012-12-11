# distutils: language = c++
# distutils: sources = ./ibugMM/mesh/mesh.cpp ./ibugMM/mesh/vertex.cpp ./ibugMM/mesh/halfedge.cpp ./ibugMM/mesh/vec3.cpp ./ibugMM/mesh/triangle.cpp


from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
from scipy.sparse import coo_matrix
cimport numpy as np
cimport cython

cdef extern from "mesh.h":
  cdef cppclass Mesh:
    Mesh(double   *coords,      unsigned n_coords,
         unsigned *coordsIndex, unsigned n_triangles) except +
    unsigned n_triangles, n_coords
    void verifyAttachements()
    void calculateLaplacianOperator()
    void calculateGradient()
    void calculateDivergence()
    double* vertexScalar
    double* vertexVec3
    double* triangleScalar
    double* triangleVec3
    double n_full_edges
    double* v_sparse
    unsigned* i_sparse
    unsigned* j_sparse
    unsigned n_full_edges


cdef class CppMesh:
  cdef Mesh* thisptr
  cdef np.ndarray vertex_scalar
  cdef np.ndarray vertex_vector
  cdef np.ndarray triangle_scalar
  cdef np.ndarray triangle_vector
  cdef public np.ndarray i_sparse
  cdef public np.ndarray j_sparse
  cdef public np.ndarray v_sparse

  def __cinit__(self, np.ndarray[double,   ndim=2, mode="c"] coords      not None, 
                      np.ndarray[unsigned, ndim=2, mode="c"] coordsIndex not None, **kwargs):
    
    self.coords = coords;
    self.coordsIndex = coordsIndex
    self.thisptr = new Mesh(     &coords[0,0],      coords.shape[0], 
                            &coordsIndex[0,0], coordsIndex.shape[0])

  def __init__(self, **kwargs):
    print 'init called now'
    self.i_sparse = np.zeros([self.thisptr.n_full_edges*2 + self.thisptr.n_coords],dtype=np.uint32)
    self.j_sparse = np.zeros([self.thisptr.n_full_edges*2 + self.thisptr.n_coords],dtype=np.uint32)
    self.v_sparse = np.zeros([self.thisptr.n_full_edges*2 + self.thisptr.n_coords])
    self._set_i_sparse(self.i_sparse)
    self._set_j_sparse(self.j_sparse)
    self._set_v_sparse(self.v_sparse)

    self.vertex_scalar = np.zeros([self.n_coords])
    self.vertex_vector = np.zeros([self.n_coords,3])
    self.triangle_scalar = np.zeros([self.n_triangles])
    self.triangle_vector = np.zeros([self.n_triangles,3])
    self._set_vertex_scalar(self.vertex_scalar)
    self._set_vertex_vector(self.vertex_vector)
    self._set_triangle_scalar(self.triangle_scalar)
    self._set_triangle_vector(self.triangle_vector)

  def __dealloc__(self):
    del self.thisptr

  @property
  def n_coords(self):
    return self.thisptr.n_coords

  @property
  def n_triangles(self):
    return self.thisptr.n_triangles
  @property
  def n_full_edges(self):
    return self.thisptr.n_full_edges

  def zero_attachments(self):
    """Sets all the attachments to 0
    """
    self.vertex_scalar[:]     = 0
    self.vertex_vector[:,:]   = 0
    self.triangle_scalar[:]   = 0
    self.triangle_scalar[:,:] = 0


  def laplacian_operator(self):
    self.thisptr.calculateLaplacianOperator()
    # we know this leaves sparse_matrix as Lc
    # and vertex_scalar as 2/3 vertex areas
    L_c = self.construct_sparse_vertex_matrix()
    A_vals = self.vertex_scalar
    A_i = np.arange(self.n_coords)
    A = coo_matrix((A_vals, (A_i,A_i)))
    return L_c, A

  def gradient(self, np.ndarray[double, ndim=1, mode="c"] s_field not None):
    """
    Return the gradient (per face) of the per vertex scalar field 

    C++ effects:
    vertex_scalar   - the scalar field value (per vertex) that we are taking
                      the gradient of.
    triangle_vector - the resulting gradient (per triangle)
    :param s_field: scalar field value per vertex
    :type s_field: ndarray[1,n_coords]
    :return: Gradient evaluated over each triangle
    :rtype: ndarray[float]
   
    """
    self._set_vertex_scalar(s_field)
    self.thisptr.calculateGradient()
    return self.triangle_vector.copy()

  def divergence(self, np.ndarray[double, ndim=2, mode="c"] v_field not None):
    """
    Return the divergence (per vertex) of the field stored in triangle_vector.

    C++ effects:
    triangle_vector - input
    vertex_scalar   - result storage

    :return: Gradient evaluated over each triangle
    :rtype: ndarray[float]
   
    """
    self._set_triangle_vector(v_field)
    self.thisptr.calculateDivergence()
    return self.vertex_scalar.copy()


  cdef construct_sparse_vertex_matrix(self):
    """Takes the sparse matrix instructions from 
       the Mesh class and returns a sparse matrix built from them.
       Note that unlike the other attachments, changing this field 
       will not trigger a change in the C++ relm.
       TODO: wipe the sparse matrix clean in c++
    """
    return coo_matrix((self.v_sparse, (self.i_sparse, self.j_sparse)))

 
  def _set_vertex_scalar(self, np.ndarray[double, ndim=1, mode="c"] 
                               vertex_scalar not None):
    if vertex_scalar.shape[0] != self.n_coords:
      raise Exception('trying to attach a vertex scalar of incorrect dimensionality')
    else:
      self.thisptr.vertexScalar = &vertex_scalar[0]

  def _set_vertex_vector(self, np.ndarray[double, ndim=2, mode="c"] 
                               vertex_vector not None):
    if vertex_vector.shape[0] != self.n_coords or vertex_vector.shape[1] != 3:
      raise Exception('trying to attach a vertex vector of incorrect dimensionality')
    else:
      self.thisptr.vertexVec3 = &vertex_vector[0,0]

  def _set_triangle_scalar(self, np.ndarray[double, ndim=1, mode="c"] 
                               triangle_scalar not None):
    if triangle_scalar.shape[0] != self.n_triangles:
      raise Exception('trying to attach a triangle scalar of incorrect dimensionality')
    else:
      self.thisptr.triangleScalar = &triangle_scalar[0]

  def _set_triangle_vector(self, np.ndarray[double, ndim=2, mode="c"] 
                               triangle_vector not None):
    if triangle_vector.shape[0] != self.n_triangles or triangle_vector.shape[1] != 3:
      raise Exception('trying to attach a triangle vector of incorrect dimensionality')
    else:
      self.thisptr.triangleVec3 = &triangle_vector[0,0]

  def _set_i_sparse(self, np.ndarray[unsigned, ndim=1, mode="c"] 
                               i_sparse not None):
    self.thisptr.i_sparse = &i_sparse[0]

  def _set_j_sparse(self, np.ndarray[unsigned, ndim=1, mode="c"] 
                               j_sparse not None):
    self.thisptr.j_sparse = &j_sparse[0]

  def _set_v_sparse(self, np.ndarray[double, ndim=1, mode="c"] 
                               v_sparse not None):
    self.thisptr.v_sparse = &v_sparse[0]

  def verify_attachments(self):
    self.thisptr.verifyAttachements()
