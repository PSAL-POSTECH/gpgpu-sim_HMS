#ifndef _HMS_REQUEST_H
#define _HMS_REQUEST_H

#include "Request.h"
#include "../gpgpu-sim/addrdec.h"

namespace ramulator {
class HMSRequestCreator {
public:
  typedef request::QueueType QTYPE;////Request::QueueType
  typedef request::Type RTYPE;//Request::Type

  HMSRequestCreator(
    const memory_config* m_config,
    unsigned int num_dram_cache_offset_bits,
    std::deque<Request> *pcie_pending,  // Pending for PCI-E invalidation
    std::deque<Request> *core_pending,  // Pending for core (SM) request
    std::deque<Request> *tag_arr_pending,    // Pending for TagArray request
    std::deque<Request> *cache_probe_pending,
    std::deque<Request> *line_read_pending,
    std::deque<Request> *pcie_invalidate_cache_hit_pending,
    std::deque<Request> *read_dram_cache_line_pending);

  typedef struct request_property {
    int rank_id;
    std::deque<Request> *pending;
  } request_property;

  Request create_hms_request(unsigned long scm_addr, unsigned long dram_addr, 
                             int num_bytes, RTYPE rtype, QTYPE qtype, 
                             unsigned long clk,
                             int evict_target_index = -1,
                             mem_fetch* mf = nullptr,
                             function<void(Request&)> callback = nullptr,
                             unsigned long long req_receive_time = 0,
                             unsigned long long req_scheduled_time = 0);//[](Request& req){}
  unsigned long get_dram_row_start_addr(mem_addr_t dram_addr);

private:
  const memory_config* m_config;

  std::deque<Request>* pcie_pending;
  //std::deque<Request>* pcie_cache_probe_pending;
  std::deque<Request>* core_pending;
  //std::deque<Request>* hm_pending;
  std::deque<Request>* tag_arr_pending;
  std::deque<Request>* cache_probe_pending;
  std::deque<Request>* line_read_pending;
  std::deque<Request> *pcie_invalidate_cache_hit_pending;
  std::deque<Request> *read_dram_cache_line_pending;

  std::map<QTYPE, request_property> request_property_map;

  std::vector<int> get_addr_vec(unsigned long mem_addr, int mem_type);

   unsigned line_size_log_bits;
   unsigned num_dram_cache_offset_bits;
};

} // end namespace
#endif
