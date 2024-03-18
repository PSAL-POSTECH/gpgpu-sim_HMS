#include "HMS.h"
#include "HMSRequest.h"

#include "../gpgpu-sim/addrdec.h"
#include "../gpgpu-sim/gpu-sim.h"

namespace ramulator {

HMSRequestCreator::HMSRequestCreator(
      const memory_config* m_config,
      unsigned int num_dram_cache_offset_bits,
      std::deque<Request> *pcie_pending,  // Pending for PCI-E invalidation
      std::deque<Request> *core_pending,  // Pending for core (SM) request
      std::deque<Request> *tag_arr_pending,    // Pending for TagArr request
      std::deque<Request> *cache_probe_pending,
      std::deque<Request> *line_read_pending,// Pending for probing dram cache
      std::deque<Request> *pcie_invalidate_cache_hit_pending,//Pending for PCI-E invalidation, dram cache hit
      std::deque<Request> *read_dram_cache_line_pending
      )
    : m_config(m_config), num_dram_cache_offset_bits(num_dram_cache_offset_bits), pcie_pending(pcie_pending), 
      core_pending(core_pending), line_read_pending(line_read_pending),
      tag_arr_pending(tag_arr_pending), cache_probe_pending(cache_probe_pending),
      pcie_invalidate_cache_hit_pending(pcie_invalidate_cache_hit_pending),
      read_dram_cache_line_pending(read_dram_cache_line_pending) {
    
    request_property_map[QTYPE::TAG_ARR_READ] = {HMS::DRAM_RANK_ID, tag_arr_pending};
    request_property_map[QTYPE::TAG_ARR_WRITE] = {HMS::DRAM_RANK_ID, tag_arr_pending};

    request_property_map[QTYPE::CACHE_HIT] = {HMS::DRAM_RANK_ID, core_pending};
    request_property_map[QTYPE::CACHE_MISS] = {HMS::SCM_RANK_ID, core_pending};
    request_property_map[QTYPE::REMAIN_LINE_READ] = {HMS::SCM_RANK_ID, line_read_pending};
    request_property_map[QTYPE::CACHE_EVICT] = {HMS::SCM_RANK_ID, nullptr};

    request_property_map[QTYPE::ON_DEMAND_CACHE_PROBE] = 
      {HMS::DRAM_RANK_ID, cache_probe_pending};
    request_property_map[QTYPE::ON_DEMAND_CACHE_FILL] = 
      {HMS::DRAM_RANK_ID, nullptr};
    request_property_map[QTYPE::ON_DEMAND_CACHE_META_FILL] =
      {HMS::DRAM_RANK_ID, nullptr};
    request_property_map[QTYPE::DRAM_CACHE_LINE_READ] = 
      {HMS::DRAM_RANK_ID, read_dram_cache_line_pending};

    request_property_map[QTYPE::PCIE_INVALIDATE] = {HMS::SCM_RANK_ID, pcie_pending};
    request_property_map[QTYPE::PCIE_VALIDATE] = {HMS::SCM_RANK_ID, nullptr};
    request_property_map[QTYPE::PCIE_INVALIDATE_CACHE_HIT] = {HMS::DRAM_RANK_ID, pcie_invalidate_cache_hit_pending};

    line_size_log_bits = log2(m_config->dram_cache_line_size);
}

std::vector<int> HMSRequestCreator::get_addr_vec(unsigned long mem_addr, 
                                                 int mem_type) {
  addrdec_t raw_addr;
  if (mem_type == HMS::SCM_RANK_ID)
    m_config->m_address_mapping.addrdec_tlx(mem_addr, &raw_addr);
  else if (mem_type == HMS::DRAM_RANK_ID) {
    m_config->m_address_mapping.addrdec_tlx_dram(mem_addr, &raw_addr);
  }

  int vector_size = int(HMS::Level::MAX);
  std::vector<int> addr_vec(vector_size);

  addr_vec[int(HMS::Level::Channel)] = 0;
  addr_vec[int(HMS::Level::BankGroup)] = raw_addr.bk & (4 - 1);
  addr_vec[int(HMS::Level::Bank)] = raw_addr.bk >> 2;
  assert(addr_vec[int(HMS::Level::BankGroup)] < 4 &&
         addr_vec[int(HMS::Level::Bank)] < 4);
  addr_vec[int(HMS::Level::Row)] = raw_addr.row;
  addr_vec[int(HMS::Level::Column)] = raw_addr.col >> 5;
  addr_vec[int(HMS::Level::Rank)] = raw_addr.rank; //this was changed from mem_type
  return addr_vec;
}


unsigned long HMSRequestCreator::get_dram_row_start_addr(mem_addr_t dram_addr) {
  //correctness checked2!
  //dram_addr : R--RBBB xxx B xxxxxxxx -> without channel bit
  //return dram start addr : R--RBBB 000 B 00000000 -> without channel bit (256B, line_size_log_bits : 8, offset bits : 3)
  //return dram start addr : R--RBBB 000 B 00000000 -> without channel bit (128B, line_size_log_bits : 7, offset bits : 4)
  //return dram start addr : R--RBBB 000 B 00000000 -> without channel bit (64B, line_size_log_bits : 6, offset bits : 5)

  //dram_addr : R--RBBB xx B xxxxxxxxx -> without channel bit
  //return dram start addr : R--RBBB 00 B 000000000 -> without channel bit (512B, line_size_log_bits : 9, offset bits : 2)

  //dram_addr : R--RBBB x B xxxxxxxxxx -> without channel bit
  //return dram start addr : R--RBBB 0 B 0000000000 -> without channel bit (1024B, line_size_log_bits : 10, offset bits : 1)

  //dram_addr : without channel bit
  //dram_row_num : DRAM's row addr |R--BBB  CCC   B|
  //                               |row num|off|row|

  if (line_size_log_bits == 8) {
    mem_addr_t dram_row_num = dram_addr >> line_size_log_bits;
    unsigned bank_group_bit = dram_row_num & 1;

    dram_row_num = dram_row_num >> (num_dram_cache_offset_bits + 1);

    return ((dram_row_num << (num_dram_cache_offset_bits + 1)) | bank_group_bit) << line_size_log_bits;

  } else if (line_size_log_bits == 7) {
    mem_addr_t dram_row_num = dram_addr >> (line_size_log_bits + 1);
    unsigned bank_group_bit = dram_row_num & 1;

    dram_row_num = dram_row_num >> ((num_dram_cache_offset_bits - 1) + 1);

    return ((dram_row_num << ((num_dram_cache_offset_bits - 1) + 1)) | bank_group_bit) << (line_size_log_bits + 1);

  } else if (line_size_log_bits == 6) {
    mem_addr_t dram_row_num = dram_addr >> (line_size_log_bits + 2);
    unsigned bank_group_bit = dram_row_num & 1;

    dram_row_num = dram_row_num >> ((num_dram_cache_offset_bits - 2) + 1);

    return ((dram_row_num << ((num_dram_cache_offset_bits - 2) + 1)) | bank_group_bit) << (line_size_log_bits + 2);

  } else if (line_size_log_bits == 9) {
    mem_addr_t dram_row_num = dram_addr >> line_size_log_bits;
    unsigned bank_group_bit = dram_row_num & 1;

    dram_row_num = dram_row_num >> (num_dram_cache_offset_bits + 1);

    return ((dram_row_num << (num_dram_cache_offset_bits + 1)) | bank_group_bit) << line_size_log_bits;

  } else if (line_size_log_bits == 10) {
    mem_addr_t dram_row_num = dram_addr >> line_size_log_bits;
    unsigned bank_group_bit = dram_row_num & 1;

    dram_row_num = dram_row_num >> (num_dram_cache_offset_bits + 1);

    return ((dram_row_num << (num_dram_cache_offset_bits + 1)) | bank_group_bit) << line_size_log_bits;

  } else {
    assert(0 && "Not supported");
  }
  
}

Request HMSRequestCreator::create_hms_request(unsigned long scm_addr,
                                              unsigned long dram_addr, 
                                              int num_bytes,
                                              RTYPE rtype,
                                              QTYPE qtype,
                                              unsigned long clk,
                                              int evict_target_index,
                                              mem_fetch* mf,
                                              function<void(Request&)> callback,
                                              unsigned long long req_receive_time,
                                              unsigned long long req_scheduled_time) {
  Request req(scm_addr, dram_addr, rtype, qtype, num_bytes, mf, callback);
  req.arrive = clk;
  req.target_pending = request_property_map[qtype].pending;
  req.evict_target_index = evict_target_index;
  req.req_receive_time = req_receive_time;
  req.req_scheduled_time = req_scheduled_time;

  if (request_property_map[qtype].rank_id == HMS::DRAM_RANK_ID) { //request to DRAM
    if (qtype == QTYPE::TAG_ARR_READ) {
      req.addr_vec = get_addr_vec(get_dram_row_start_addr(dram_addr), HMS::DRAM_RANK_ID);
    }
    else {
      req.addr_vec = get_addr_vec(dram_addr, HMS::DRAM_RANK_ID);
    }
  } else { //request to SCM
    req.addr_vec = get_addr_vec(scm_addr, HMS::SCM_RANK_ID);
  }
  return req;
}


}


