#ifndef __HELPER_CUDA__
#define __HELPER_CUDA__

#include <cudnn.h>

#define CudnnSafeCall( err ) __cudnnSafeCall( err, __FILE__, __LINE__ )
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return;
}

inline void __cudnnSafeCall( cudnnStatus_t err, const char *file, const int line )
{
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        fprintf( stderr, "cudnnSafeCall() failed at %s:%i : %s\n",
                 file, line, cudnnGetErrorString( err ) );
        exit( -1 );
    }

    return;
}

#endif