#define _GNU_SOURCE
#include <cpuid.h>
#include <sched.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    int count = (int)sysconf(_SC_NPROCESSORS_ONLN);
    unsigned int max_leaf = __get_cpuid_max(0, NULL);
    printf("max_cpuid_leaf=0x%x\n", max_leaf);
    for (int cpu = 0; cpu < count; ++cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            perror("sched_setaffinity");
            return 1;
        }
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (max_leaf >= 0x1a)
            __cpuid_count(0x1a, 0, eax, ebx, ecx, edx);
        printf("cpu=%d core_type=0x%02x native_model=0x%06x eax=0x%08x\n",
               cpu, eax >> 24, eax & 0x00ffffff, eax);
    }
    return 0;
}
