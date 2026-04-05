// Cross-platform time wrapper (not in PPSU paper)

#ifndef PORTABLE_TIME_H
#define PORTABLE_TIME_H

#ifdef _WIN32
// Windows-specific includes
#include <winsock2.h>
#include <windows.h>
#include <stdint.h>
#pragma comment(lib, "ws2_32.lib")  // Link Winsock for select()

// timeval structure (not in Windows by default unless winsock is included)
// Under Windows winsock2.h defines timeval, so we shouldn't redefine it.

// Emulate gettimeofday()
static inline int gettimeofday(struct timeval *tp, void *tzp) {
    FILETIME ft;
    ULARGE_INTEGER tmp;
    GetSystemTimeAsFileTime(&ft);
    tmp.LowPart = ft.dwLowDateTime;
    tmp.HighPart = ft.dwHighDateTime;

    // Convert from Windows epoch (1601) to Unix epoch (1970)
    uint64_t t = tmp.QuadPart - 116444736000000000ULL;
    tp->tv_sec = (long)(t / 10000000UL);
    tp->tv_usec = (long)((t % 10000000UL) / 10);
    return 0;
}

#else
// POSIX systems
#include <sys/time.h>
#include <unistd.h>
#include <sys/select.h>
#endif

// Wall-clock time in seconds as a double (portable)
static inline double u_wseconds(void) {
    #ifdef _WIN32
        static LARGE_INTEGER freq = {0};
        LARGE_INTEGER counter;
        if (freq.QuadPart == 0) {
            QueryPerformanceFrequency(&freq);
        }
        QueryPerformanceCounter(&counter);
        return (double)counter.QuadPart / (double)freq.QuadPart;
    #else
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return tp.tv_sec + tp.tv_usec / 1e6;
    #endif
    }
    

// Add timeval values: result = a + b
#ifndef timeradd
#define timeradd(a, b, result)                                \
    do {                                                      \
        (result)->tv_sec = (a)->tv_sec + (b)->tv_sec;         \
        (result)->tv_usec = (a)->tv_usec + (b)->tv_usec;      \
        if ((result)->tv_usec >= 1000000) {                   \
            ++(result)->tv_sec;                               \
            (result)->tv_usec -= 1000000;                     \
        }                                                     \
    } while (0)
#endif

// Subtract timeval values: result = a - b
#ifndef timersub
#define timersub(a, b, result)                                \
    do {                                                      \
        (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;         \
        (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;      \
        if ((result)->tv_usec < 0) {                          \
            --(result)->tv_sec;                               \
            (result)->tv_usec += 1000000;                     \
        }                                                     \
    } while (0)
#endif

#endif  // PORTABLE_TIME_H
