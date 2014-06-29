#ifndef __CHECK_CUDA_ERRORS_HPP__
#define __CHECK_CUDA_ERRORS_HPP__

// define printf for kernels
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher    
#endif

// raise an excpetion Python-side, does not leave C/C++ code
#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

// goto onfailure, on failure
#define cudaErrorCheck_goto(call) { if (!cudaAssert(call,__FILE__,__LINE__)) {goto onfailure;} }

bool cudaAssert(const cudaError err, const char *file, const int line);
void throwRuntimeError(const char *what);

#endif