
#ifndef __CPU_TIMER_H__
#define __CPU_TIMER_H__

// TODO: Remove commented-out code
//#include <time.h>
#include <windows.h>

/**
 * @brief Timer for CPU.
 * 
 */
struct CpuTimer
{
    /* clock_t start;
    clock_t stop; */

    LARGE_INTEGER freq;
    LARGE_INTEGER start;
    LARGE_INTEGER stop;

    CpuTimer() // : start(0), stop(0)
    {
        QueryPerformanceFrequency(&freq);
    }

    void Start()
    {
        //start = clock();
        QueryPerformanceCounter(&start);
    }

    void Stop()
    {
        //stop = clock();
        QueryPerformanceCounter(&stop);
    }

    /**
     * @brief Computes the elapsed time in milliseconds.
     * 
     * @return the elapsed time in milliseconds 
     */
    double Elapsed() const
    {
        //return (double)1000 * (stop - start) / CLOCKS_PER_SEC;
        return (double)1000 * (stop.QuadPart - start.QuadPart) / freq.QuadPart;
    }
};

#endif