#ifndef __CUDACHECKS_HPP__
#define __CUDACHECKS_HPP__

#ifdef __CUDACC__
    #define cudaErrorRaise(err) { fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); }
    #define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }
    void cudaAssert(const cudaError err, const char *file, const int line);
#endif

#endif
