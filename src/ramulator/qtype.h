#ifndef __QTYPE
#define __QTYPE

namespace request {
enum class Type {
    R_READ,     // dram read
    R_WRITE,    // dram write
    REFRESH,
    POWERDOWN,
    SELFREFRESH,
    EXTENSION,
    MAX
};
enum class QueueType {
    TAG_ARR_READ, TAG_ARR_WRITE, CACHE_HIT, CACHE_MISS, CACHE_EVICT, REMAIN_LINE_READ,
    ON_DEMAND_CACHE_PROBE, DRAM_CACHE_LINE_READ,
    ON_DEMAND_CACHE_FILL, 
    ON_DEMAND_CACHE_META_FILL,
    PCIE_VALIDATE, PCIE_INVALIDATE, PCIE_INVALIDATE_CACHE_HIT, CACHE_HIT_RD, CACHE_HIT_WR, CACHE_MISS_RD, CACHE_MISS_WR, MAX
};
}
#endif