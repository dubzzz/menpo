import os
from setuptools import setup, find_packages
from Cython.Build import cythonize
import glob
from os.path import isfile
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
import subprocess
import versioneer

# BEGINNING of CUDA functions
# *** Necessary for CUDA compilation ***

# CUDA compilation is adapted from the source
# https://github.com/rmcgibbo/npcuda-example

def find_in_path(name, path):
    """
    Find a file in a search path
    """
    
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """
    Locate the CUDA environment on the system
    
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

# END of CUDA functions

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    install_requires = []
    ext_modules = []
    include_dirs = []
    cython_exts = []
else:
    import numpy as np

    CUDA = locate_cuda()

    # Obtain the numpy include directory. This logic works across numpy versions.
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()

    # ---- C/C++ EXTENSIONS ---- #
    cython_modules = ["menpo/shape/mesh/normals.pyx",
                      "menpo/transform/piecewiseaffine/fastpwa.pyx",
                      "menpo/image/feature/cppimagewindowiterator.pyx"]
    
    # Build extensions
    cython_exts = list()
    for module in cython_modules:
        module_path = '/'.join(module.split('/')[:-1])
        module_sources_cu = list() #glob.glob(pjoin(pjoin(module_path, "cpp"), "*.cu"))
        module_sources_cpp = glob.glob(pjoin(pjoin(module_path, "cpp"), "*.cpp"))
        
        module_ext = Extension(name=module[:-4],
                               sources=module_sources_cu + [name for name in module_sources_cpp if not name.endswith("main.cpp")] + [module], # sources = cuda files + cpp files (order seems important)
                               library_dirs=[CUDA['lib64']],
                               libraries=['cudart'],
                               language='c++',
                               runtime_library_dirs=[CUDA['lib64']],
                               extra_compile_args={'gcc': [],
                                                   'nvcc': ['-arch=sm_20', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
                               include_dirs=[numpy_include, CUDA['include'], 'src'])
        cython_exts.append(module_ext)

    #cython_exts = cythonize(cython_exts, quiet=True),
    include_dirs = [numpy_include]
    install_requires = [# Core
                        'numpy>=1.8.0',
                        'scipy>=0.14.0',
                        'Cython>=0.20.1',

                        # Image
                        'Pillow>=2.0.0',
                        'scikit-image>=0.8.2',

                        # ML
                        'scikit-learn>=0.14.1',

                        # 3D import
                        'menpo-pyvrml97==2.3.0a4',
                        'cyassimp>=0.1.3',

                        # Rasterization
                        'cyrasterize>=0.1.5',

                        # Visualization
                        'matplotlib>=1.2.1',
                        'mayavi>=4.3.0']

# Versioneer allows us to automatically generate versioning from
# our git tagging system which makes releases simpler.
versioneer.VCS = 'git'
versioneer.versionfile_source = 'menpo/_version.py'
versioneer.versionfile_build = 'menpo/_version.py'
versioneer.tag_prefix = 'v'  # tags are like v1.2.0
versioneer.parentdir_prefix = 'menpo-'  # dirname like 'menpo-v1.2.0'

setup(name='menpo',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False,
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=include_dirs,
      ext_modules=cythonize(cython_exts, quiet=True),
      packages=find_packages(),
      install_requires=install_requires,
      package_data={'menpo': ['data/*']},
      tests_require=['nose>=1.3.0', 'mock>=1.0.1']
)
