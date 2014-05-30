from setuptools import setup, find_packages
from Cython.Build import cythonize
import os
import glob
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy as np
import versioneer

# CUDA compilation is adapted from the source
# https://github.com/rmcgibbo/npcuda-example

# Versioneer allows us to automatically generate versioning from
# our git tagging system which makes releases simpler.  
versioneer.VCS = 'git'
versioneer.versionfile_source = 'menpo/_version.py'
versioneer.versionfile_build = 'menpo/_version.py'
versioneer.tag_prefix = 'v'             # tags are like v1.2.0
versioneer.parentdir_prefix = 'menpo-'  # dirname like 'menpo-v1.2.0'

# ---- C/C++ EXTENSIONS ---- #
cython_modules = ["menpo/shape/mesh/normals.pyx",
                  "menpo/transform/piecewiseaffine/fastpwa.pyx",
                  "menpo/image/feature/cppimagewindowiterator.pyx"]

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
CUDA = locate_cuda()

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# Build extensions
cython_exts = list()
for module in cython_modules:
    module_path = '/'.join(module.split('/')[:-1])
    module_sources_cpp = glob.glob(pjoin(pjoin(module_path, "cpp"), "*.cpp"))
    module_sources_cu = glob.glob(pjoin(pjoin(module_path, "cpp"), "*.cu"))
    
    module_ext = Extension(module[:-4],
        sources=module_sources_cu + [name for name in module_sources_cpp if not name.endswith("main.cpp")] + [module], # sources = cuda files + cpp files + pyx file (order seems important)
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={'gcc': [],
                            'nvcc': ['-arch=sm_20', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
        include_dirs=[numpy_include, CUDA['include'], 'src'])
    cython_exts.append(module_ext)

def customize_compiler_for_nvcc(self):
    """
    inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(name='menpo',
      version=versioneer.get_version(),
      cmdclass={'build_ext': custom_build_ext},
      zip_safe=False,
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cythonize(cython_exts, quiet=True),
      packages=find_packages(),
      install_requires=[# Core
                        'numpy>=1.8.0',
                        # we would like this to be 0.14.0, waiting for conda
                        # support though (see menpo.math.multivariate)
                        'scipy>=0.13.3',
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
                        'matplotlib>=1.2.1'],
      package_data={'menpo': ['data/*']},
      tests_require=['nose>=1.3.0'],
      extras_require={'3d': 'mayavi>=4.3.0'}
      )
