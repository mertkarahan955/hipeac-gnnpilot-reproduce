#ifndef KG_UTILS__
#define KG_UTILS__

#include <cuda_runtime_api.h>   // cudaEvent APIs

#define kg_max(a, b) (((a) >= (b))? (a): (b))
#define kg_min(a, b) (((a) < (b))? (a): (b))

#define CUDA_CHECK_ERROR(call)\
{\
	cudaError_t _error = (cudaError_t)(call);\
	if(_error != cudaSuccess)\
	{\
		printf("*** CUDA Error *** at [%s:%d] error=%d, reason:%s \n",\
			__FILE__, __LINE__, _error, cudaGetErrorString(_error));\
	}\
}

#define CUSPARSE_CHECK(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE) ;                                                   \
    }                                                                          \
}

// Encapsule CUDA timing APIs.
// 
// Usage:
//   GpuTimer timer; // create
//   timer.start();  // when you start recording
//   timer.stop();   // when  you stop recording
//   float dur = timer.elapsed_msecs(); // duration in milliseconds

struct GpuTimer
{
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GpuTimer() 
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start()
    {
        cudaEventRecord(startEvent, 0);
    }

    void stop()
    {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
    }

    float elapsed_msecs()
    {
        float elapsed;
        cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
        return elapsed;
    }
};

#endif
