#include <cmath>

#include "HMSController.h"
#include <bitset>
#include <algorithm>
#include <random>

#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/addrdec.h"

#define ROW_ADDR(X) ((X) >> 11)

namespace ramulator {

HMSController::HMSController(const Config& configs, 
                             DRAM<HMS>* channel,
                             int channel_id,
                             const memory_config* m_config,
                             class memory_partition_unit* mp,
                             class gpgpu_sim* gpu) :
    Controller(configs, channel, channel_id, m_config, mp, gpu),
    LEVEL(m_config->metadata_level), m_config(m_config) {
  available_queue_size = m_config->gpgpu_frfcfs_dram_sched_queue_size;
  available_tag_access_queue_size = m_config->tag_arr_queue_size;

  // The DRAM cache policy determines whether to cache current data.
  std::string policy_str = std::string(m_config->dram_cache_policy);
  if (policy_str == "TWO_LEVEL_SCORE") {
    policy = POLICY::TWO_LEVEL_SCORE;
  } else if (policy_str == "ALWAYS") {
    policy = POLICY::ALWAYS;
  } else if (policy_str == "IDEAL_ACCESS_CNT") {
    policy = POLICY::IDEAL_ACCESS_CNT;
  } else {
    assert(false);
  }
  std::cout << "DRAM cache policy: " << policy_str << std::endl;;

  act_cnt_2MB = m_config->act_cnt_2MB;

  global_score = 0.0f;

  max_counter = 0;
  max_score = 0.0;
  max_mix = 0.0;

  min_counter = std::numeric_limits<unsigned long>::max();
  min_score = std::numeric_limits<float>::max();
  min_mix = std::numeric_limits<float>::max();

  line_size = m_config->dram_cache_line_size;
  line_size_log_bits = log2(line_size);

  assoc_of_dram_cache = m_config->assoc_of_dram_cache;
  num_dram_cache_offset_bits = 3;
  assert(assoc_of_dram_cache == 1);
  assert(line_size_log_bits == 8);
  assert(line_size == 256);
  assert(TAG_ARR_SECTOR_CHUNCK_SIZE == 8);
  assert(TAG_ARR_SECTOR_SIZE==4);
  assert(TAG_ARR_ENTRY_CHUNK_SIZE==8);

  enable_tag_cache = m_config->enable_tag_cache;

  num_scm_channels = m_config->m_n_mem;
  num_scm_channel_log_bits = log2(num_scm_channels);
  num_dram_channels = m_config->m_n_mem;

  unsigned num_dram_pages = m_config->num_dram_pages;
  unsigned num_dram_pages_per_ch = ceil(num_dram_pages / num_dram_channels);
  assert(num_dram_pages_per_ch != 0);
  num_cache_lines = ceil(((num_dram_pages_per_ch * 4096) / line_size)/assoc_of_dram_cache);
  // Applications with small footprints can have 0 dram page.
  if (num_cache_lines < 8)
    num_cache_lines = 8;

  std::cout<<"dram cache associativity : "<<assoc_of_dram_cache<<std::endl;
  std::cout<<"line size: "<<line_size<<", num_scm_channels: "<<num_scm_channels<<", num_dram_channels : "<<num_dram_channels<<std::endl;
  std::cout<<"num_dram_cache_offset_bits: "<<num_dram_cache_offset_bits<<std::endl;
  std::cout<<"line_size_log_bits: "<<line_size_log_bits<<std::endl;
  std::cout<<"num_dram_pages : "<<num_dram_pages<<", num dram pages per channel: "<<num_dram_pages_per_ch<<std::endl;
  std::cout<<"num dram pages per channel: "<<num_dram_pages_per_ch<<std::endl;
  std::cout<<"num dram cache lines: "<<num_cache_lines<<std::endl;

  req_creator = new HMSRequestCreator(m_config,
                                      num_dram_cache_offset_bits,
                                      &pcie_pending,
                                      &core_pending,
                                      &tag_arr_pending,
                                      &cache_probe_pending,
                                      &line_read_pending,
                                      &pcie_invalidate_cache_hit_pending,
                                      &read_dram_cache_line_pending);

  unsigned dram_row_buffer_size = 2048;
  unsigned num_cache_line_tags_per_entry = (dram_row_buffer_size/m_config->dram_cache_line_size)/(m_config->assoc_of_dram_cache);

  double total_num_tag_arr_entry = double(num_cache_lines)/double(num_cache_line_tags_per_entry);
  double total_num_tag_arr_line = double(total_num_tag_arr_entry)/double(TAG_ARR_SECTOR_CHUNCK_SIZE);
  if (total_num_tag_arr_line < m_config->num_tag_arr_assocs) {
    total_num_tag_arr_line = m_config->num_tag_arr_assocs;
  }
  unsigned temp_num_tag_arr_set = (unsigned)ceil(double(total_num_tag_arr_line) / double(m_config->num_tag_arr_assocs));

  unsigned temp_num_tag_arr_set_log = LOGB2_32(temp_num_tag_arr_set);
  if (pow(2, temp_num_tag_arr_set_log) != temp_num_tag_arr_set) {
    temp_num_tag_arr_set_log++;
  }

  //if tag_arr_on_chip_percent is 1 : decrease temp_num_tag_arr_set_log 1
  //ex) temp_num_tag_arr_set_log : 4, tag_arr_on_chip_percent : 1 
  // => temp_num_tag_arr_set_log becomes 3 => set size becomes 8 from 16
  if (temp_num_tag_arr_set_log <= m_config->tag_arr_on_chip_percent) {
    assert(0 && "Cannot reduce on-chip tag array size to that size");
  }
  else {
    temp_num_tag_arr_set_log -= m_config->tag_arr_on_chip_percent;
  }
  
  unsigned num_tag_arr_set_log = temp_num_tag_arr_set_log;
  unsigned num_tag_arr_set = pow(2, num_tag_arr_set_log);  

  std::cout<<"L2 tag array number of set original : "<<temp_num_tag_arr_set<<", reduced : "<<num_tag_arr_set<<std::endl;

  tag_arr_config = cache_config(m_config->tag_arr_line_size, num_tag_arr_set, m_config->num_tag_arr_assocs, m_config->tag_arr_evict_buffer_size, m_config->continuous);
  tag_arr = new tag_arr_tag_array(tag_arr_config, channel_id, &write_tag_arr_entry_queue);
  tag_arr_access_queue.max = m_config->tag_arr_queue_size;

  if (print_detail) {
    tag_arr_printer.open("output/tag_arr_ch_" + std::to_string(channel_id) + ".txt");
    assert(tag_arr_printer.is_open());
  }

  num_cache_hit = 0;
  num_cache_miss = 0;

  num_cache_hit_rd = 0;
  num_cache_miss_rd = 0;
  num_cache_hit_wr = 0;
  num_cache_miss_wr = 0;

  num_tag_arr_read = 0;
  num_tag_arr_write = 0;

  num_tag_arr_miss = 0;
  num_tag_arr_hit = 0;

  num_line_read = 0;
  num_cache_probe = 0;

  num_evict = 0;
  num_fill = 0;

  num_score_reg_made = 0;

  total_row_accesses = 0;
  total_num_activates = 0;

  num_replace_old_after_probe = 0;
  num_fail_after_probe = 0;
  num_fill_empty_after_probe = 0;

  post_processor[QTYPE::CACHE_HIT] = 
    std::bind(&HMSController::post_process_cache_hit, this, std::placeholders::_1);
  post_processor[QTYPE::CACHE_MISS] = 
    std::bind(&HMSController::post_process_cache_miss, this, std::placeholders::_1);
  post_processor[QTYPE::REMAIN_LINE_READ] = 
    std::bind(&HMSController::post_process_remain_line_read, this, std::placeholders::_1);

  if (ch_id < num_dram_channels) {
    std::cout<<"ch_id : "<<ch_id<<", this channel has scm and dram"<<std::endl;
    ch_type = CH_TYPE::SCMDRAM;
  }
  else {
    std::cout<<"ch_id : "<<ch_id<<", this channel has only scm"<<std::endl;
    ch_type = CH_TYPE::SCM;
  }
  assert(ch_type != CH_TYPE::MAX);  
}

int HMSController::get_global_bank_id(Request req) {
  return req.addr_vec[int(HMS::Level::BankGroup)] * 4 +
         req.addr_vec[int(HMS::Level::Bank)];
}


std::tuple<mem_addr_t, mem_addr_t, int> HMSController::get_dram_addr(mem_addr_t scm_mem_addr) {
  //get_scm_row_num
  //addr : R--RBBBCCCBHHHCCCSSSSS
  //return:R--RBBBCCCB            (256B, line_size_log_bits : 8)

  int num_dram_ch = m_config->m_n_mem; //number of dram channel in rank 0 
  mem_addr_t line_col_addr = scm_mem_addr & (line_size - 1);
  mem_addr_t addr_for_chip, rest_of_addr, line_addr;
  int dram_ch_tag;//tag for dram channel -> to identify scm addr
  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(scm_mem_addr, &raw_addr);
  int scm_ch = raw_addr.chip;
  int scm_rank = raw_addr.rank;

  int total_ch = scm_ch;

  addr_for_chip = total_ch % num_dram_ch; //dram channel number
  dram_ch_tag = total_ch / num_dram_ch; //dram channel tag

  mem_addr_t scm_row_num = get_scm_row_num(scm_mem_addr); //same as rest_of_addr
  line_addr = (scm_row_num % num_cache_lines) << line_size_log_bits;

  line_addr |= line_col_addr; //dram address without channel

  assert(dram_ch_tag == 0);
  assert(addr_for_chip == total_ch);
  assert(addr_for_chip == ch_id);

  return std::make_tuple(addr_for_chip, line_addr, dram_ch_tag);
}

new_addr_type HMSController::get_dram_tag(mem_addr_t scm_mem_addr) {
  int num_dram_ch = m_config->m_n_mem;
  mem_addr_t rest_of_addr, dram_tag;

  mem_addr_t scm_row_num = get_scm_row_num(scm_mem_addr); //same as rest_of_addr
  dram_tag = scm_row_num / num_cache_lines;

  return dram_tag;
}

bool HMSController::is_last_col(mem_addr_t dram_addr) {
  //dram_addr : |R--BBB  CCC   B|CCCSSSSS (without channel bits)
  //            |row num|off|row|line  
  //want      :          CCC     CCC
  //if CCCCCC is 63, return true, else return false

  //(256B, line_size_log_bits : 8, offset bits : 3)
  int num_of_sets_per_row_buffer = pow(2, num_dram_cache_offset_bits);
  int col_bit = 6;
  int last_col = (1 << col_bit) - 1;
  assert(last_col == 63);

  mem_addr_t dram_addr_wo_s = dram_addr >> 5;
  int col_at_line = dram_addr_wo_s & 7;
  int col = (dram_addr >> (line_size_log_bits + 1)) & (num_of_sets_per_row_buffer - 1);

  col = (col << 3) | col_at_line;

  return col == last_col;
}

unsigned long HMSController::get_dram_row_num(mem_addr_t dram_addr) {
  //dram_row_num : DRAM's row addr |R--BBB  CCC   B|
  //                               |row num|off|row|
  //dram_addr : R--RBBB 000 B 00000000 (without channel bit)
  //return    : R--RBBB     B (256B, line_size_log_bits : 8, offset bits : 3)

  mem_addr_t dram_row_num = dram_addr >> line_size_log_bits;
  return ((dram_row_num >> (num_dram_cache_offset_bits + 1)) << 1) | (dram_row_num & 1);
}

unsigned long HMSController::get_dram_row_start_addr(mem_addr_t dram_row_num) {
  //dram_row_num : R--RBBBB
  //return dram start addr : R--RBBB 000 B 00000000 -> without channel bit (256B, line_size_log_bits : 8, offset bits : 3)

  unsigned bank_group_bit = dram_row_num & 1;
  return (((dram_row_num >> 1) << (num_dram_cache_offset_bits + 1)) | bank_group_bit) << line_size_log_bits;
}

unsigned long HMSController::get_dram_cache_line_num(mem_addr_t dram_addr) {
  //dram_addr : R--RBBBCCCBCCCSSSSS (without channel bits)
  //return    : R--RBBBCCCB        (256B, line_size_log_bits : 8)
  return dram_addr >> line_size_log_bits;
}

bool HMSController::is_last_dram_cache_line(mem_addr_t dram_cache_line_num) {
  //dram_addr : R--RBBBCCCBCCCSSSSS (without channel bits)
  //dram_cache_line_num    : R--RBBBCCCB        (256B, line_size_log_bits : 8)
  //                                CCC  -> if CCC is 7 return true else false  
  int num_of_sets_per_row_buffer = pow(2, num_dram_cache_offset_bits);
  
  mem_addr_t dram_addr_wo_B = dram_cache_line_num >> 1;
  int col = dram_addr_wo_B & (num_of_sets_per_row_buffer - 1);
  return (col == 7);
}

unsigned long HMSController::get_dram_cache_line_from_dram_row_num(unsigned long dram_row_num, int offset) {
  //dram_row_num : ttt ssss (RR--RBBBB)
  //return : ttt sss ooo s   (256B, line_size_log_bits : 8, offset bits : 3)
  unsigned bank_group_bit = dram_row_num & 1;
  assert(offset < pow(2, num_dram_cache_offset_bits));
  return ((((dram_row_num >> 1) << num_dram_cache_offset_bits) | offset) << 1) | bank_group_bit;
  
}

unsigned long HMSController::get_scm_row_num(unsigned long addr) {
  //return without channel bits & col bits
  //addr : R--RBBBCCCBHHHCCCSSSSS
  //return:R--RBBBCCCB            (256B, line_size_log_bits : 8)
  unsigned int num_scm_ch = m_config->m_n_mem; //number of total scm chip
  return ((addr >> line_size_log_bits) / num_scm_ch);
}
unsigned long HMSController::get_scm_row_addr(unsigned long row_num) {
  //row_num : scm row num
  //input :R--RBBBCCCB            (256B, line_size_log_bits : 8)
  unsigned int num_scm_ch = m_config->m_n_mem; //number of total scm chip
  return ((row_num*num_scm_ch) << line_size_log_bits); 
}

bool HMSController::full(RTYPE type) {
  return 
    (qsize[QTYPE::CACHE_HIT] + qsize[QTYPE::CACHE_MISS] + qsize[QTYPE::CACHE_EVICT] >= available_queue_size) ||
    (qsize[QTYPE::TAG_ARR_READ] + qsize[QTYPE::TAG_ARR_WRITE] 
    + qsize[QTYPE::DRAM_CACHE_LINE_READ] + qsize[QTYPE::ON_DEMAND_CACHE_FILL] + qsize[QTYPE::ON_DEMAND_CACHE_META_FILL] + qsize[QTYPE::ON_DEMAND_CACHE_PROBE]
    + qsize[QTYPE::REMAIN_LINE_READ] >= available_queue_size)  ||
    (tag_arr_access_queue.size() >= available_tag_access_queue_size);
}

void HMSController::enqueue_pcie(Request& req) {
  req.arrive = clk;
  req.req_receive_time = clk;
  qsize[req.qtype] += 1;
  if (req.qtype == QTYPE::PCIE_INVALIDATE) {
    req.tag_cache_access_ready_cycle = clk + m_config->tag_cache_latency;
    assert(req.req_receive_time > 0);
    tag_arr_access_queue.q.push_back(req);
  }
  else if (req.qtype == QTYPE::PCIE_VALIDATE) {   
    pcie_queue.q.push_back(req);
    qsize[QTYPE::PCIE_VALIDATE] += 1;
    num_reqs[QTYPE::PCIE_VALIDATE] += req.num_bytes;
    num_traffic[QTYPE::PCIE_VALIDATE] += req.num_bytes;
  }
  else {
    assert(0 && "INVALID QTYPE\n");
  }
}

bool HMSController::enqueue(Request& req) {
  if (req.type == RTYPE::R_WRITE || req.type == RTYPE::R_READ) {
    req.arrive = clk;
    req.req_receive_time = clk;

    if (full(req.type))
      return false;
    
    if (!enable_tag_cache) {
      //there is no tag cache
      mem_addr_t scm_row_addr = req.addr;
      mem_addr_t scm_row_num = get_scm_row_num(req.addr);
      auto dram_addr_info = get_dram_addr(req.addr);
      mem_addr_t dram_ch = std::get<0>(dram_addr_info);
      mem_addr_t dram_addr = std::get<1>(dram_addr_info);
      int dram_ch_tag = std::get<2>(dram_addr_info);
      mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
      new_addr_type dram_tag = get_dram_tag(req.addr);

      std::tuple<unsigned long, unsigned, unsigned> key;
      addrdec_t raw_addr;
      m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
      unsigned scm_rank = raw_addr.rank;

      key = std::make_tuple(scm_row_num, scm_rank, ch_id);
      
      //check mshr
      if ((hit_merge_reg.find(key) != hit_merge_reg.end()) && (hit_merge_reg[key].dirty || req.type == RTYPE::R_READ)) {               
        hit_merge_reg[key].num_cols += (req.original_num_bytes / transfer_bytes_per_clk);

        if (is_last_col(dram_addr)) {
          assert(req.num_bytes == req.original_num_bytes);
          num_cache_miss += 1;
          if (req.type == RTYPE::R_READ) 
            num_cache_miss_rd += 1;
          else if (req.type == RTYPE::R_WRITE)
            num_cache_miss_wr += 1;
          else
            assert(0 && "should not happen\n");

          int req_bytes_dram = req.num_bytes;

          if (req_bytes_dram > transfer_bytes_per_clk) {
            req_bytes_dram -= transfer_bytes_per_clk;
            assert(req_bytes_dram > 0);

            hit_merge_reg[key].count += (req_bytes_dram / transfer_bytes_per_clk);
            // Generate cache hit request
            assert(req.req_receive_time > 0);
            Request cache_hit_req = 
              req_creator->create_hms_request(
                req.addr, dram_addr, 
                req_bytes_dram, req.type, QTYPE::CACHE_HIT, clk, -1, req.mf, req.callback, req.req_receive_time);

            commonq.q.push_back(cache_hit_req); 
            qsize[QTYPE::CACHE_HIT] += 1;
            num_reqs[QTYPE::CACHE_HIT] += req_bytes_dram;

            if (req.type == RTYPE::R_READ)
              num_traffic[QTYPE::CACHE_HIT_RD] += req_bytes_dram;
            else if (req.type == RTYPE::R_WRITE)
              num_traffic[QTYPE::CACHE_HIT_WR] += req_bytes_dram;

            Request cache_hit_scm_req = 
            req_creator->create_hms_request(
              req.addr, dram_addr,
              transfer_bytes_per_clk, req.type, QTYPE::CACHE_MISS, clk);

            commonq.q.push_back(cache_hit_scm_req); 
            qsize[QTYPE::CACHE_MISS] += 1;
            num_reqs[QTYPE::CACHE_MISS] += transfer_bytes_per_clk;

            if (req.type == RTYPE::R_READ)
              num_traffic[QTYPE::CACHE_MISS_RD] += transfer_bytes_per_clk;
            else if (req.type == RTYPE::R_WRITE)
              num_traffic[QTYPE::CACHE_MISS_WR] += transfer_bytes_per_clk; 

          } else {
            // Generate scm request even if cache is hit
            Request cache_hit_scm_req = 
            req_creator->create_hms_request(
              req.addr, dram_addr,
              req.num_bytes, req.type, QTYPE::CACHE_MISS, clk, -1, req.mf, req.callback, req.req_receive_time);
            cache_hit_scm_req.original_num_bytes = req.original_num_bytes;
        
            commonq.q.push_back(cache_hit_scm_req); 
            qsize[QTYPE::CACHE_MISS] += 1;
            num_reqs[QTYPE::CACHE_MISS] += req.num_bytes;

            if (req.type == RTYPE::R_READ)
              num_traffic[QTYPE::CACHE_MISS_RD] += req.num_bytes;
            else if (req.type == RTYPE::R_WRITE)
              num_traffic[QTYPE::CACHE_MISS_WR] += req.num_bytes; 
          }
        } 
        else {
          num_cache_hit += 1;
          if (req.type == RTYPE::R_READ)
            num_cache_hit_rd += 1;
          else if (req.type == RTYPE::R_WRITE)
            num_cache_hit_wr += 1;
          else
            assert(0 && "should not happen\n");
          //normal access
          hit_merge_reg[key].count += (req.original_num_bytes / transfer_bytes_per_clk);

          // Generate cache hit request
          assert(req.req_receive_time > 0);
          Request cache_hit_req = 
            req_creator->create_hms_request(
              scm_row_addr, dram_addr, 
              req.num_bytes, req.type, QTYPE::CACHE_HIT, clk, -1, req.mf, req.callback, req.req_receive_time);

          commonq.q.push_back(cache_hit_req); 
          qsize[QTYPE::CACHE_HIT] += 1;
          num_reqs[QTYPE::CACHE_HIT] += req.num_bytes;

          if (req.type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_HIT_RD] += req.num_bytes;
          else if (req.type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_HIT_WR] += req.num_bytes;
          else 
            assert(0);
        }
        
      }
      else if (probe_candidate.find(key) != probe_candidate.end()) {
        num_cache_miss += 1;
        if (req.type == RTYPE::R_READ)
          num_cache_miss_rd += 1;
        else if (req.type == RTYPE::R_WRITE)
          num_cache_miss_wr += 1;
        else
          assert(0 && "should not happen\n");

        if (probe_candidate[key].line_read_done) {
          assert(req.callback != nullptr);
          req.callback(req);
          if (num_traffic[QTYPE::REMAIN_LINE_READ] > req.num_bytes) {
            num_traffic[QTYPE::REMAIN_LINE_READ] -= req.num_bytes;
            if (req.type == RTYPE::R_READ)
              num_traffic[QTYPE::CACHE_MISS_RD] += req.num_bytes;
            else if (req.type == RTYPE::R_WRITE)
              num_traffic[QTYPE::CACHE_MISS_WR] += req.num_bytes;
          }
          probe_candidate[key].dirty |= (req.type == RTYPE::R_WRITE);
        }
        else {
          probe_candidate[key].pending.push_back(req);
        }
      } else if (miss_merge_reg.find(key) != miss_merge_reg.end()) {
        num_cache_miss += 1;
        if (req.type == RTYPE::R_READ)
          num_cache_miss_rd += 1;
        else if (req.type == RTYPE::R_WRITE)
          num_cache_miss_wr += 1;
        else
          assert(0 && "should not happen\n");

        assert(req.original_num_bytes > 0);
        miss_merge_reg[key].count += (req.original_num_bytes / transfer_bytes_per_clk);
        miss_merge_reg[key].num_cols += (req.original_num_bytes / transfer_bytes_per_clk);
        miss_merge_reg[key].dirty |= (req.type == RTYPE::R_WRITE);

        if (miss_merge_reg[key].pending_invalidate.size() > 0) {
          assert(0 && "This should not happen, if there is pending_invalidate in miss_merge_reg, there will be no more request to same scm row\n");
        }
        assert(req.req_receive_time > 0);
        Request cache_miss_req = 
          req_creator->create_hms_request(
            scm_row_addr, dram_addr,
            req.num_bytes, req.type, QTYPE::CACHE_MISS, clk, -1, req.mf, req.callback, req.req_receive_time);
        cache_miss_req.original_num_bytes = req.original_num_bytes;
        commonq.q.push_back(cache_miss_req);
        qsize[QTYPE::CACHE_MISS] += 1;
        num_reqs[QTYPE::CACHE_MISS] += req.num_bytes;

        if (req.type == RTYPE::R_READ)
          num_traffic[QTYPE::CACHE_MISS_RD] += req.num_bytes;
        else if (req.type == RTYPE::R_WRITE)
          num_traffic[QTYPE::CACHE_MISS_WR] += req.num_bytes;
      } else {
        assert(req.req_receive_time > 0);
        Request tag_read_req =
          req_creator->create_hms_request( 
              scm_row_addr, dram_addr, transfer_bytes_per_clk,
              RTYPE::R_READ, QTYPE::TAG_ARR_READ, clk, -1, req.mf, req.callback, req.req_receive_time);
        assert(req.original_num_bytes == req.num_bytes);
        tag_read_req.original_type = req.type;
        tag_read_req.original_num_bytes = req.original_num_bytes;

        tag_arr_queue.q.push_back(tag_read_req);
        qsize[QTYPE::TAG_ARR_READ] += 1;
        num_reqs[QTYPE::TAG_ARR_READ] += transfer_bytes_per_clk;
        num_traffic[QTYPE::TAG_ARR_READ] += transfer_bytes_per_clk;
        
      }
      return true;
    }
    else {
      req.tag_cache_access_ready_cycle = clk + m_config->tag_cache_latency;
      assert(req.req_receive_time > 0);
      tag_arr_access_queue.q.push_back(req);
      return true;
    }
  } else {
    req.arrive = clk;
    otherq.q.push_back(req);
    return true;
  }
}

bool HMSController::do_tag_arr_miss(mem_addr_t scm_row_addr, unsigned long dram_addr) {
  assert(ch_type == CH_TYPE::SCMDRAM);
  mem_addr_t dram_row_num = get_dram_row_num(dram_addr);
  mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
  int num_of_sets_per_row_buffer = pow(2, num_dram_cache_offset_bits);

  if (read_dram_tags_reg.find(dram_row_num) != read_dram_tags_reg.end()) {
    //this dram_row_num's tags are being read from dram. so skip
    return false;
  }
  if (tag_arr_queue.size() >= available_tag_access_queue_size) {
    return false;
  }
  assert(read_dram_tags_reg.find(dram_row_num) == read_dram_tags_reg.end());
  read_dram_tags_reg[dram_row_num] = false; //set read_dram_tags_reg to be not requested again

  int req_bytes = transfer_bytes_per_clk;
  Request tag_read_req =
    req_creator->create_hms_request( 
        scm_row_addr, dram_addr, req_bytes, 
        RTYPE::R_READ, QTYPE::TAG_ARR_READ, clk); //request for read all tags of one row buffer of DRAM
  tag_arr_queue.q.push_back(tag_read_req);
  qsize[QTYPE::TAG_ARR_READ] += 1;
  num_reqs[QTYPE::TAG_ARR_READ] += req_bytes;

  num_traffic[QTYPE::TAG_ARR_READ] += req_bytes;
  return true;
}


void HMSController::check_cache_hit(Request& req) {
  //access dram_cache_line directly to check info
  assert(req.req_receive_time > 0);
  assert(!enable_tag_cache);

  mem_addr_t scm_row_num = get_scm_row_num(req.addr);
  auto dram_addr_info = get_dram_addr(req.addr);
  mem_addr_t dram_ch = std::get<0>(dram_addr_info);
  mem_addr_t dram_addr = std::get<1>(dram_addr_info);
  int dram_ch_tag = std::get<2>(dram_addr_info);
  mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
  new_addr_type dram_tag = get_dram_tag(req.addr);

  std::tuple<unsigned long, unsigned, unsigned> key;
  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
  unsigned scm_rank = raw_addr.rank;
  assert(scm_rank == 1);

  key = std::make_tuple(scm_row_num, scm_rank, ch_id);
  
  assert(req.original_num_bytes > 0);
  assert(req.original_type != Type::MAX);

  bool empty_line = false;
  bool victim_dirty = false;
  bool cache_miss = false;

  //check dram cache hit
  if (dram_cache_line.find(dram_cache_line_num) != dram_cache_line.end()) {
    assert(dram_cache_line[dram_cache_line_num].tags.find(0) != dram_cache_line[dram_cache_line_num].tags.end());
    if ((dram_cache_line[dram_cache_line_num].tags[0] == dram_tag) && (dram_cache_line[dram_cache_line_num].valid_bits[0] == true)) {
      int request_bytes = req.original_num_bytes;
      
      if (hit_merge_reg.find(key) == hit_merge_reg.end()) {
        hit_merge_reg[key].num_cols = 0;
        hit_merge_reg[key].count = 0;
        hit_merge_reg[key].dirty = false;
      }
      assert(req.original_num_bytes > 0);
      
      hit_merge_reg[key].num_cols += (req.original_num_bytes / transfer_bytes_per_clk);
      hit_merge_reg[key].dirty |= (req.original_type == RTYPE::R_WRITE);
      
      if (is_last_col(dram_addr)) {
        num_cache_miss += 1;
        if (req.type == RTYPE::R_READ) 
          num_cache_miss_rd += 1;
        else if (req.type == RTYPE::R_WRITE)
          num_cache_miss_wr += 1;
        else
          assert(0 && "should not happen\n");
        
        int req_bytes_dram = request_bytes;

        if (req_bytes_dram > transfer_bytes_per_clk) {
          req_bytes_dram -= transfer_bytes_per_clk;
          assert(req_bytes_dram > 0);

          hit_merge_reg[key].count += (req_bytes_dram / transfer_bytes_per_clk);
          Request cache_hit_req = 
            req_creator->create_hms_request(
              req.addr, dram_addr, 
              req_bytes_dram, req.original_type, QTYPE::CACHE_HIT, clk, -1, req.mf, req.callback, req.req_receive_time);
          cache_hit_req.original_num_bytes = req.original_num_bytes;

          commonq.q.push_back(cache_hit_req); 
          qsize[QTYPE::CACHE_HIT] += 1;
          num_reqs[QTYPE::CACHE_HIT] += req_bytes_dram;

          if (req.type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_HIT_RD] += req_bytes_dram;
          else if (req.type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_HIT_WR] += req_bytes_dram;

          Request cache_hit_scm_req = 
          req_creator->create_hms_request(
            req.addr, dram_addr,
            transfer_bytes_per_clk, req.original_type, QTYPE::CACHE_MISS, clk);

          commonq.q.push_back(cache_hit_scm_req); 
          qsize[QTYPE::CACHE_MISS] += 1;
          num_reqs[QTYPE::CACHE_MISS] += transfer_bytes_per_clk;

          if (req.type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_MISS_RD] += transfer_bytes_per_clk;
          else if (req.type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_MISS_WR] += transfer_bytes_per_clk; 

        } else {
          Request cache_hit_scm_req = 
          req_creator->create_hms_request(
            req.addr, dram_addr,
            req_bytes_dram, req.original_type, QTYPE::CACHE_MISS, clk, -1, req.mf, req.callback, req.req_receive_time);
          cache_hit_scm_req.original_num_bytes = req.original_num_bytes;
      
          commonq.q.push_back(cache_hit_scm_req); 
          qsize[QTYPE::CACHE_MISS] += 1;
          num_reqs[QTYPE::CACHE_MISS] += req_bytes_dram;

          if (req.type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_MISS_RD] += req_bytes_dram;
          else if (req.type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_MISS_WR] += req_bytes_dram; 
        }
      } else {
        //dram cache hit
        num_cache_hit += 1;
        if (req.original_type == RTYPE::R_READ)
          num_cache_hit_rd += 1;
        else if (req.original_type == RTYPE::R_WRITE)
          num_cache_hit_wr += 1;
        else
          assert(0 && "should not happen\n");

        //normal access
        hit_merge_reg[key].count += (request_bytes / transfer_bytes_per_clk);

        Request cache_hit_req = 
          req_creator->create_hms_request(
            req.addr, dram_addr, 
            request_bytes, req.original_type, QTYPE::CACHE_HIT, clk, -1, req.mf, req.callback, req.req_receive_time);
        cache_hit_req.original_num_bytes = req.original_num_bytes;

        assert(request_bytes % transfer_bytes_per_clk == 0);

        commonq.q.push_back(cache_hit_req);
        qsize[QTYPE::CACHE_HIT] += 1;
        num_reqs[QTYPE::CACHE_HIT] += request_bytes;

        if (req.original_type == RTYPE::R_WRITE) 
          num_traffic[QTYPE::CACHE_HIT_WR] += request_bytes;
        else if (req.original_type == RTYPE::R_READ)
          num_traffic[QTYPE::CACHE_HIT_RD] += request_bytes;
      }
      if ((req.original_type == RTYPE::R_WRITE) && (dram_cache_line[dram_cache_line_num].dirty_bits[0] == false)) {
        dram_cache_line[dram_cache_line_num].dirty_bits[0] = true;
        assert(!enable_tag_cache);
        Request tag_arr_wb_req = 
          req_creator->create_hms_request(
              req.addr, dram_addr, transfer_bytes_per_clk,
              RTYPE::R_WRITE, QTYPE::TAG_ARR_WRITE, clk);

        tag_arr_queue.q.push_back(tag_arr_wb_req);
        qsize[QTYPE::TAG_ARR_WRITE] += 1;
        num_reqs[QTYPE::TAG_ARR_WRITE] += transfer_bytes_per_clk;

        num_traffic[QTYPE::TAG_ARR_WRITE] += transfer_bytes_per_clk; 
      }
      return;
    } else {
      //dram cache miss
      num_cache_miss += 1;
      if (req.original_type == RTYPE::R_READ)
        num_cache_miss_rd += 1;
      else if (req.original_type == RTYPE::R_WRITE)
        num_cache_miss_wr += 1;
      else
        assert(0 && "should not happen\n");

      cache_miss = true;
      if (dram_cache_line[dram_cache_line_num].valid_bits[0] == false) {
        //empty line
        empty_line = true;
        victim_dirty = false;
      }
      else {
        //not empty
        if (dram_cache_line[dram_cache_line_num].dirty_bits[0] == true) {
          //victim is dirty
          empty_line = false;
          victim_dirty = true;
        }
        else {
          empty_line = false;
          victim_dirty = false;
        }
      }
    }
  } else {
    num_cache_miss += 1;
    if (req.original_type == RTYPE::R_READ)
      num_cache_miss_rd += 1;
    else if (req.original_type == RTYPE::R_WRITE)
      num_cache_miss_wr += 1;
    else
      assert(0 && "should not happen\n");

    cache_miss = true;
    //empty line
    empty_line = true;
    victim_dirty = false;
  }

  assert(cache_miss);
  bool probe_candidate_exist = (probe_candidate.find(key) != probe_candidate.end());
  bool pending_invalidate_exist = false;
  if (probe_candidate_exist) {
    pending_invalidate_exist = (probe_candidate[key].pending_invalidate.size() > 0);
    if (miss_merge_reg.find(key) != miss_merge_reg.end()) {
      if (pending_invalidate_exist == false) {
        pending_invalidate_exist = (miss_merge_reg[key].pending_invalidate.size() > 0);
      }
    }
  }
  if (pending_invalidate_exist) {
    assert(0 && "This should not happen, if there is pending_invalidate in miss_merge_reg or probe_candidate_exist, there will be no more request to same scm row\n");
  }
  if (probe_candidate_exist && (!pending_invalidate_exist)) {
    //only use probe_candidate when pending_invalidate does not exist
    if (probe_candidate[key].line_read_done) {
      assert(req.callback != nullptr);
      req.callback(req);
      if (num_traffic[QTYPE::REMAIN_LINE_READ] > req.original_num_bytes) {
        num_traffic[QTYPE::REMAIN_LINE_READ] -= req.original_num_bytes;
        if (req.original_type == RTYPE::R_READ)
          num_traffic[QTYPE::CACHE_MISS_RD] += req.original_num_bytes;
        else if (req.original_type == RTYPE::R_WRITE)
          num_traffic[QTYPE::CACHE_MISS_WR] += req.original_num_bytes;
      }
      probe_candidate[key].dirty |= (req.original_type == RTYPE::R_WRITE);
    } else {
      assert(ch_type == CH_TYPE::SCMDRAM);
      probe_candidate[key].pending.push_back(req);
    }                    
  } else {
    if (miss_merge_reg.find(key) == miss_merge_reg.end()) {
      miss_merge_reg[key].num_cols = 0;
      miss_merge_reg[key].count = 0;
      miss_merge_reg[key].dirty = false;
      miss_merge_reg[key].trigger_lock = false;

      //check evict target
      
      bool have_to_evict = !empty_line;
      int target_index = 0;
      bool evict_target_dirty = victim_dirty;
      
      miss_merge_reg[key].have_to_evict = have_to_evict;
      miss_merge_reg[key].evict_target_idx = target_index;
      miss_merge_reg[key].evict_target_dirty = evict_target_dirty;

    }
    assert(req.original_num_bytes > 0);
    miss_merge_reg[key].count += (req.original_num_bytes / transfer_bytes_per_clk);
    miss_merge_reg[key].num_cols += (req.original_num_bytes / transfer_bytes_per_clk);
    miss_merge_reg[key].dirty |= (req.original_type == RTYPE::R_WRITE);

    if (miss_merge_reg[key].pending_invalidate.size() > 0) {
      assert(0 && "This should not happen, if there is pending_invalidate in miss_merge_reg, there will be no more request to same scm row\n");
    }

    Request cache_miss_req = 
      req_creator->create_hms_request(
        req.addr, dram_addr,
        req.original_num_bytes, req.original_type, QTYPE::CACHE_MISS, clk, -1, req.mf, req.callback, req.req_receive_time);
    cache_miss_req.original_num_bytes = req.original_num_bytes;

    assert(req.original_num_bytes % transfer_bytes_per_clk == 0);
    commonq.q.push_back(cache_miss_req); 
    qsize[QTYPE::CACHE_MISS] += 1;
    num_reqs[QTYPE::CACHE_MISS] += req.original_num_bytes;

    if (req.original_type == RTYPE::R_READ)
      num_traffic[QTYPE::CACHE_MISS_RD] += req.original_num_bytes;
    else if (req.original_type == RTYPE::R_WRITE)
      num_traffic[QTYPE::CACHE_MISS_WR] += req.original_num_bytes;
  }

}

void HMSController::access_tag_arr() {
  std::vector<std::list<Request>::iterator> erase_list;
  auto it = tag_arr_access_queue.q.begin();
  int iterate_cnt = 0;
  for (; it != tag_arr_access_queue.q.end(); ++it) {
    if (iterate_cnt >= available_tag_access_queue_size) {
      break;
    }
    mem_addr_t scm_row_num = get_scm_row_num(it->addr);
    auto dram_addr_info = get_dram_addr(it->addr);
    mem_addr_t dram_ch = std::get<0>(dram_addr_info);
    mem_addr_t dram_addr = std::get<1>(dram_addr_info);
    int dram_ch_tag = std::get<2>(dram_addr_info);
    mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
    new_addr_type dram_tag = get_dram_tag(it->addr);

    std::tuple<unsigned long, unsigned, unsigned> key;
    unsigned scm_rank = it->addr_vec[int(HMS::Level::Rank)];
    key = std::make_tuple(scm_row_num, scm_rank, ch_id);

    if ((hit_merge_reg.find(key) != hit_merge_reg.end()) && (it->qtype != QTYPE::PCIE_INVALIDATE) && (it->qtype != QTYPE::PCIE_VALIDATE) && (hit_merge_reg[key].dirty || it->type == RTYPE::R_READ)) {
      //if request is read, it is okay to check hit_merge_reg and directly send request without read tag array
      //if request is write and hit_merge_reg[key].dirty, it is okay to directly send request without read tag aray
      assert(it->original_num_bytes > 0);
            
      hit_merge_reg[key].num_cols += (it->original_num_bytes / transfer_bytes_per_clk);

      if (is_last_col(dram_addr)) {
        num_cache_miss += 1;
        if (it->type == RTYPE::R_READ) 
          num_cache_miss_rd += 1;
        else if (it->type == RTYPE::R_WRITE)
          num_cache_miss_wr += 1;
        else
          assert(0 && "should not happen\n");
        
        int req_bytes_dram = it->num_bytes;

        if (req_bytes_dram > transfer_bytes_per_clk) {
          req_bytes_dram -= transfer_bytes_per_clk;
          assert(req_bytes_dram > 0);

          hit_merge_reg[key].count += (req_bytes_dram / transfer_bytes_per_clk);
          Request cache_hit_req = 
            req_creator->create_hms_request(
              it->addr, dram_addr, 
              req_bytes_dram, it->type, QTYPE::CACHE_HIT, clk, -1, it->mf, it->callback, it->req_receive_time);

          commonq.q.push_back(cache_hit_req); 
          qsize[QTYPE::CACHE_HIT] += 1;
          num_reqs[QTYPE::CACHE_HIT] += req_bytes_dram;

          if (it->type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_HIT_RD] += req_bytes_dram;
          else if (it->type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_HIT_WR] += req_bytes_dram;

          Request cache_hit_scm_req = 
          req_creator->create_hms_request(
            it->addr, dram_addr,
            transfer_bytes_per_clk, it->type, QTYPE::CACHE_MISS, clk);

          commonq.q.push_back(cache_hit_scm_req); 
          qsize[QTYPE::CACHE_MISS] += 1;
          num_reqs[QTYPE::CACHE_MISS] += transfer_bytes_per_clk;

          if (it->type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_MISS_RD] += transfer_bytes_per_clk;
          else if (it->type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_MISS_WR] += transfer_bytes_per_clk; 

        } else {
          Request cache_hit_scm_req = 
          req_creator->create_hms_request(
            it->addr, dram_addr,
            it->num_bytes, it->type, QTYPE::CACHE_MISS, clk, -1, it->mf, it->callback, it->req_receive_time);
          cache_hit_scm_req.original_num_bytes = it->original_num_bytes;
      
          commonq.q.push_back(cache_hit_scm_req); 
          qsize[QTYPE::CACHE_MISS] += 1;
          num_reqs[QTYPE::CACHE_MISS] += it->num_bytes;

          if (it->type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_MISS_RD] += it->num_bytes;
          else if (it->type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_MISS_WR] += it->num_bytes; 
        }

      } else {
        num_cache_hit += 1;
        if (it->type == RTYPE::R_READ)
          num_cache_hit_rd += 1;
        else if (it->type == RTYPE::R_WRITE)
          num_cache_hit_wr += 1;
        else
          assert(0 && "should not happen\n");

        //normal access
        hit_merge_reg[key].count += (it->original_num_bytes / transfer_bytes_per_clk);
        // Generate cache hit request
        Request cache_hit_req = 
          req_creator->create_hms_request(
            it->addr, dram_addr, 
            it->num_bytes, it->type, QTYPE::CACHE_HIT, clk, -1, it->mf, it->callback, it->req_receive_time);

        commonq.q.push_back(cache_hit_req); 
        qsize[QTYPE::CACHE_HIT] += 1;
        num_reqs[QTYPE::CACHE_HIT] += it->num_bytes;

        if (it->type == RTYPE::R_READ)
          num_traffic[QTYPE::CACHE_HIT_RD] += it->num_bytes;
        else if (it->type == RTYPE::R_WRITE)
          num_traffic[QTYPE::CACHE_HIT_WR] += it->num_bytes;
      }

      unsigned tag_arr_cache_idx = (unsigned) - 1;
      enum dram_cache_status dram_cache_stat = dram_cache_status::DONT_KNOW;
      enum cache_request_status tag_arr_cache_stat = tag_arr->probe(dram_cache_line_num, tag_arr_cache_idx);

      if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
        tag_arr->access(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx, dram_cache_stat);
        num_tag_arr_read++;
      }

      erase_list.push_back(it);
      
    } else if ((probe_candidate.find(key) != probe_candidate.end()) && (it->qtype != QTYPE::PCIE_INVALIDATE) && (it->qtype != QTYPE::PCIE_VALIDATE)) {
      num_cache_miss += 1;
      if (it->type == RTYPE::R_READ)
        num_cache_miss_rd += 1;
      else if (it->type == RTYPE::R_WRITE)
        num_cache_miss_wr += 1;
      else
        assert(0 && "should not happen\n");

      if (probe_candidate[key].line_read_done) {
        assert(it->callback != nullptr);
        it->callback(*it); 
        if (num_traffic[QTYPE::REMAIN_LINE_READ] > it->num_bytes) {
          num_traffic[QTYPE::REMAIN_LINE_READ] -= it->num_bytes;
          if (it->type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_MISS_RD] += it->num_bytes;
          else if (it->type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_MISS_WR] += it->num_bytes;
        }
        
        probe_candidate[key].dirty |= (it->type == RTYPE::R_WRITE);
      }
      else {
        assert(ch_type == CH_TYPE::SCMDRAM);
        probe_candidate[key].pending.push_back(*it);
      }

      unsigned tag_arr_cache_idx = (unsigned) - 1;
      enum dram_cache_status dram_cache_stat = dram_cache_status::DONT_KNOW;
      enum cache_request_status tag_arr_cache_stat = tag_arr->probe(dram_cache_line_num, tag_arr_cache_idx);

      if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
        tag_arr->access(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx, dram_cache_stat); 
        num_tag_arr_read++;
      }

      erase_list.push_back(it);
      
    } else if ((miss_merge_reg.find(key) != miss_merge_reg.end()) && (it->qtype != QTYPE::PCIE_INVALIDATE) && (it->qtype != QTYPE::PCIE_VALIDATE)) {
      num_cache_miss += 1;
      if (it->type == RTYPE::R_READ)
        num_cache_miss_rd += 1;
      else if (it->type == RTYPE::R_WRITE)
        num_cache_miss_wr += 1;
      else
        assert(0 && "should not happen\n");

      assert(it->original_num_bytes > 0);
      miss_merge_reg[key].count += (it->original_num_bytes / transfer_bytes_per_clk);
      miss_merge_reg[key].num_cols += (it->original_num_bytes / transfer_bytes_per_clk);
      miss_merge_reg[key].dirty |= (it->type == RTYPE::R_WRITE);

      if (miss_merge_reg[key].pending_invalidate.size() > 0) {
        assert(0 && "This should not happen, if there is pending_invalidate in miss_merge_reg, there will be no more request to same scm row\n");
      }

      Request cache_miss_req = 
        req_creator->create_hms_request(
          it->addr, dram_addr,
          it->num_bytes, it->type, QTYPE::CACHE_MISS, clk, -1, it->mf, it->callback, it->req_receive_time);
      cache_miss_req.original_num_bytes = it->original_num_bytes;
      commonq.q.push_back(cache_miss_req);
      qsize[QTYPE::CACHE_MISS] += 1;
      num_reqs[QTYPE::CACHE_MISS] += it->num_bytes;

      if (it->type == RTYPE::R_READ)
        num_traffic[QTYPE::CACHE_MISS_RD] += it->num_bytes;
      else if (it->type == RTYPE::R_WRITE)
        num_traffic[QTYPE::CACHE_MISS_WR] += it->num_bytes; 


      unsigned tag_arr_cache_idx = (unsigned) - 1;
      enum dram_cache_status dram_cache_stat = dram_cache_status::DONT_KNOW;
      enum cache_request_status tag_arr_cache_stat = tag_arr->probe(dram_cache_line_num, tag_arr_cache_idx);

      if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
        tag_arr->access(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx, dram_cache_stat);
        num_tag_arr_read++;
      }

      erase_list.push_back(it);
      
    } else {
      //tag array check
      //wait for tag cache access cycle
      if (clk < it->tag_cache_access_ready_cycle) {
        break;
      }
      
      unsigned tag_arr_cache_idx = (unsigned) - 1;
      enum dram_cache_status dram_cache_stat = dram_cache_status::DONT_KNOW;
      enum cache_request_status tag_arr_cache_stat = tag_arr->access(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx, dram_cache_stat);
      num_tag_arr_read++;

      if (tag_arr_cache_stat == cache_request_status::MISS || tag_arr_cache_stat == cache_request_status::SECTOR_MISS) { //tag array miss - have to check dram cache
        iterate_cnt ++;
        if (it->is_tag_arr_hit == 0) {
          //first check to tag arr
          it->is_tag_arr_hit = -1;
          num_tag_arr_miss += 1;
        }
        do_tag_arr_miss(it->addr, dram_addr);
        
      } else if (tag_arr_cache_stat == cache_request_status::RESERVATION_FAIL) {
        if (it->is_tag_arr_hit == 0) {
          //first check tag arr
          it->is_tag_arr_hit = -1;
          num_tag_arr_miss += 1;
        }
      } else { //tag array hit
        iterate_cnt ++;
        // If the entry is being read from the memory, we need to wait
        if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
          num_tag_arr_read += 1;
          if (it->is_tag_arr_hit == 0) {
            //first check tag arr
            it->is_tag_arr_hit = 1;
            num_tag_arr_hit += 1;
          }

          if (dram_cache_stat == dram_cache_status::HIT_CLEAN || dram_cache_stat == dram_cache_status::HIT_DIRTY) { // DRAM cache hit
            assert(it->is_tag_arr_hit != 0);
            if (it->qtype == QTYPE::PCIE_INVALIDATE) {
              tag_arr->set_col_invalid(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx);
              num_tag_arr_write += 1;

              //invalidate dram_cache_line
              assert(dram_cache_line.find(dram_cache_line_num) != dram_cache_line.end());
              dram_cache_line[dram_cache_line_num].tags[0] = ULONG_MAX;
              dram_cache_line[dram_cache_line_num].valid_bits[0] = false;
              dram_cache_line[dram_cache_line_num].dirty_bits[0] = false;
              dram_cache_line[dram_cache_line_num].dram_ch_tags[0] = ULONG_MAX;
              dram_cache_line[dram_cache_line_num].scm_addrs[0] = ULONG_MAX;
              dram_cache_line[dram_cache_line_num].metric_levels[0] = 0;

              assert(it->type == RTYPE::R_READ);

              if (dram_cache_stat == dram_cache_status::HIT_DIRTY) {
                Request cache_hit_pcie_invalidate_req =
                  req_creator->create_hms_request(
                    it->addr, dram_addr,
                    it->num_bytes, it->type, QTYPE::PCIE_INVALIDATE_CACHE_HIT, clk, -1, it->mf, it->callback, it->req_receive_time);
                cache_hit_pcie_invalidate_req.original_num_bytes = it->original_num_bytes;
                
                commonq.q.push_back(cache_hit_pcie_invalidate_req);
                qsize[QTYPE::PCIE_INVALIDATE_CACHE_HIT] += 1;
                num_reqs[QTYPE::PCIE_INVALIDATE_CACHE_HIT] += it->num_bytes;
              }
              else {
                assert(it->callback != nullptr);
                it->callback(*it); 
              }
            }
            else {                           
              // merge request
              if (hit_merge_reg.find(key) == hit_merge_reg.end()) {
                hit_merge_reg[key].num_cols = 0;
                hit_merge_reg[key].count = 0;
                hit_merge_reg[key].dirty = false;
              }
              assert(it->original_num_bytes > 0);
              
              hit_merge_reg[key].num_cols += (it->original_num_bytes / transfer_bytes_per_clk);
              hit_merge_reg[key].dirty |= (it->type == RTYPE::R_WRITE);

              if (is_last_col(dram_addr)) {
                num_cache_miss += 1;
                if (it->type == RTYPE::R_READ) 
                  num_cache_miss_rd += 1;
                else if (it->type == RTYPE::R_WRITE)
                  num_cache_miss_wr += 1;
                else
                  assert(0 && "should not happen\n");
                
                int req_bytes_dram = it->num_bytes;

                if (req_bytes_dram > transfer_bytes_per_clk) {
                  req_bytes_dram -= transfer_bytes_per_clk;
                  assert(req_bytes_dram > 0);

                  hit_merge_reg[key].count += (req_bytes_dram / transfer_bytes_per_clk);
                  // Generate cache hit request
                  Request cache_hit_req = 
                    req_creator->create_hms_request(
                      it->addr, dram_addr, 
                      req_bytes_dram, it->type, QTYPE::CACHE_HIT, clk, -1, it->mf, it->callback, it->req_receive_time);

                  commonq.q.push_back(cache_hit_req); 
                  qsize[QTYPE::CACHE_HIT] += 1;
                  num_reqs[QTYPE::CACHE_HIT] += req_bytes_dram;

                  if (it->type == RTYPE::R_READ)
                    num_traffic[QTYPE::CACHE_HIT_RD] += req_bytes_dram;
                  else if (it->type == RTYPE::R_WRITE)
                    num_traffic[QTYPE::CACHE_HIT_WR] += req_bytes_dram;

                  Request cache_hit_scm_req = 
                  req_creator->create_hms_request(
                    it->addr, dram_addr,
                    transfer_bytes_per_clk, it->type, QTYPE::CACHE_MISS, clk);

                  commonq.q.push_back(cache_hit_scm_req); 
                  qsize[QTYPE::CACHE_MISS] += 1;
                  num_reqs[QTYPE::CACHE_MISS] += transfer_bytes_per_clk;

                  if (it->type == RTYPE::R_READ)
                    num_traffic[QTYPE::CACHE_MISS_RD] += transfer_bytes_per_clk;
                  else if (it->type == RTYPE::R_WRITE)
                    num_traffic[QTYPE::CACHE_MISS_WR] += transfer_bytes_per_clk; 

                } else {
                  // Generate scm request even if cache is hit
                  Request cache_hit_scm_req = 
                  req_creator->create_hms_request(
                    it->addr, dram_addr,
                    it->num_bytes, it->type, QTYPE::CACHE_MISS, clk, -1, it->mf, it->callback, it->req_receive_time);
                  cache_hit_scm_req.original_num_bytes = it->original_num_bytes;
              
                  commonq.q.push_back(cache_hit_scm_req); 
                  qsize[QTYPE::CACHE_MISS] += 1;
                  num_reqs[QTYPE::CACHE_MISS] += it->num_bytes;

                  if (it->type == RTYPE::R_READ)
                    num_traffic[QTYPE::CACHE_MISS_RD] += it->num_bytes;
                  else if (it->type == RTYPE::R_WRITE)
                    num_traffic[QTYPE::CACHE_MISS_WR] += it->num_bytes; 
                }
              } else {
                num_cache_hit += 1;
                if (it->type == RTYPE::R_READ)
                  num_cache_hit_rd += 1;
                else if (it->type == RTYPE::R_WRITE)
                  num_cache_hit_wr += 1;
                else
                  assert(0 && "should not happen\n");

                if ((it->type == RTYPE::R_WRITE) && dram_cache_stat == dram_cache_status::HIT_CLEAN) {
                  tag_arr->set_col_dirty(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx);
                  num_tag_arr_write += 1;
                }

                hit_merge_reg[key].count += (it->original_num_bytes / transfer_bytes_per_clk);
                // Generate cache hit request
                Request cache_hit_req = 
                  req_creator->create_hms_request(
                    it->addr, dram_addr, 
                    it->num_bytes, it->type, QTYPE::CACHE_HIT, clk, -1, it->mf, it->callback, it->req_receive_time);

                commonq.q.push_back(cache_hit_req); 
                qsize[QTYPE::CACHE_HIT] += 1;
                num_reqs[QTYPE::CACHE_HIT] += it->num_bytes;

                if (it->type == RTYPE::R_READ)
                  num_traffic[QTYPE::CACHE_HIT_RD] += it->num_bytes;
                else if (it->type == RTYPE::R_WRITE)
                  num_traffic[QTYPE::CACHE_HIT_WR] += it->num_bytes;
              }              
            }
          } else if (dram_cache_stat == dram_cache_status::MISS_CLEAN || dram_cache_stat == dram_cache_status::MISS_DIRTY || dram_cache_stat == dram_cache_status::MISS_EMPTY) {  // DRAM cache miss
            num_cache_miss += 1;
            assert(it->is_tag_arr_hit != 0);

            if (it->type == RTYPE::R_READ)
              num_cache_miss_rd += 1;
            else if (it->type == RTYPE::R_WRITE)
              num_cache_miss_wr += 1;
            else
              assert(0 && "should not happen\n");
            
            if (it->qtype == QTYPE::PCIE_INVALIDATE) {              
              Request pcie_invalidate_req = 
                req_creator->create_hms_request(
                  it->addr, dram_addr, 
                  it->num_bytes, it->type, it->qtype, clk, -1, it->mf, it->callback, it->req_receive_time);
              pcie_invalidate_req.original_num_bytes = it->original_num_bytes;
              if (probe_candidate.find(key) != probe_candidate.end()) {
                probe_candidate[key].pending_invalidate.push_back(pcie_invalidate_req);
                assert(probe_candidate[key].pending_invalidate.size()== 1);
              } else if (miss_merge_reg.find(key) != miss_merge_reg.end()) {
                miss_merge_reg[key].pending_invalidate.push_back(pcie_invalidate_req);
                assert(miss_merge_reg[key].pending_invalidate.size()==1);
              } else {
                pcie_queue.q.push_back(pcie_invalidate_req); 
                qsize[QTYPE::PCIE_INVALIDATE] += 1;
                num_reqs[QTYPE::PCIE_INVALIDATE] += it->num_bytes;
              }
            }
            else {
              bool probe_candidate_exist = (probe_candidate.find(key) != probe_candidate.end());
              bool pending_invalidate_exist = false;
              if (probe_candidate_exist) {
                pending_invalidate_exist = (probe_candidate[key].pending_invalidate.size() > 0);
                if (miss_merge_reg.find(key) != miss_merge_reg.end()) {
                  if (pending_invalidate_exist == false) {
                    pending_invalidate_exist = (miss_merge_reg[key].pending_invalidate.size() > 0);
                  }
                }
              }
              
              if (probe_candidate_exist && (!pending_invalidate_exist)) {
                //only use probe_candidate when pending_invalidate does not exist
                if (probe_candidate[key].line_read_done) {
                  assert(it->callback != nullptr);
                  it->callback(*it);
                  if (num_traffic[QTYPE::REMAIN_LINE_READ] > it->num_bytes) {
                    num_traffic[QTYPE::REMAIN_LINE_READ] -= it->num_bytes;
                    if (it->type == RTYPE::R_READ)
                      num_traffic[QTYPE::CACHE_MISS_RD] += it->num_bytes;
                    else if (it->type == RTYPE::R_WRITE)
                      num_traffic[QTYPE::CACHE_MISS_WR] += it->num_bytes;
                  }
                  
                  probe_candidate[key].dirty |= (it->type == RTYPE::R_WRITE);
                } else {
                  probe_candidate[key].pending.push_back(*it);
                }
              } else {
                if (miss_merge_reg.find(key) == miss_merge_reg.end()) {
                  miss_merge_reg[key].num_cols = 0;
                  miss_merge_reg[key].count = 0;
                  miss_merge_reg[key].dirty = false;
                  miss_merge_reg[key].trigger_lock = false;

                  //check evict target
                  
                  bool have_to_evict = true;
                  int target_index = 0;
                  bool evict_target_dirty = false;
                  
                  if (dram_cache_stat == dram_cache_status::MISS_CLEAN || dram_cache_stat == dram_cache_status::MISS_DIRTY) {
                    //have to evict
                    have_to_evict = true;
                    std::random_device rd;
                    std::uniform_int_distribution<int> distribution(0, assoc_of_dram_cache-1);
                    std::mt19937 engine(rd());

                    target_index = distribution(engine);
                    assert(target_index < assoc_of_dram_cache);

                    assert(dram_cache_line.find(dram_cache_line_num) != dram_cache_line.end());
                    std::map<int, new_addr_type> tags = dram_cache_line[dram_cache_line_num].tags;
                    std::map<int, mem_addr_t> scm_addrs = dram_cache_line[dram_cache_line_num].scm_addrs;
                    std::map<int, int> dram_ch_tags = dram_cache_line[dram_cache_line_num].dram_ch_tags;

                    assert(tags.find(target_index) != tags.end());
                    assert(scm_addrs.find(target_index) != scm_addrs.end());
                    assert(dram_ch_tags.find(target_index) != dram_ch_tags.end());

                    mem_addr_t evicted_scm_addr = scm_addrs[target_index];
                    mem_addr_t evicted_scm_start_addr = (evicted_scm_addr>>line_size_log_bits)<<line_size_log_bits;
                    auto evicted_dram_addr_info = get_dram_addr(evicted_scm_start_addr);
                    mem_addr_t evicted_dram_ch = std::get<0>(evicted_dram_addr_info);
                    mem_addr_t evicted_dram_addr = std::get<1>(evicted_dram_addr_info);
                    int evicted_dram_ch_tag = std::get<2>(evicted_dram_addr_info);

                    mem_addr_t evicted_dram_cache_line_num = get_dram_cache_line_num(evicted_dram_addr);
                    new_addr_type evicted_dram_tag = get_dram_tag(evicted_scm_addr);

                    unsigned evict_tag_arr_cache_idx = (unsigned) - 1;
                    enum dram_cache_status evict_dram_cache_stat = dram_cache_status::DONT_KNOW;
                    enum cache_request_status evict_tag_arr_cache_stat = tag_arr->access(evicted_dram_cache_line_num, evicted_dram_tag, clk, evict_tag_arr_cache_idx, evict_dram_cache_stat);

                    if (dram_cache_stat == dram_cache_status::MISS_DIRTY) {
                      evict_target_dirty = true;
                    }

                  }
                  else if (dram_cache_stat == dram_cache_status::MISS_EMPTY) {
                    have_to_evict = false;
                  }
                  miss_merge_reg[key].have_to_evict = have_to_evict;
                  miss_merge_reg[key].evict_target_idx = target_index;
                  miss_merge_reg[key].evict_target_dirty = evict_target_dirty;

                }
                assert(it->original_num_bytes > 0);
                miss_merge_reg[key].count += (it->original_num_bytes / transfer_bytes_per_clk);
                miss_merge_reg[key].num_cols += (it->original_num_bytes / transfer_bytes_per_clk);
                miss_merge_reg[key].dirty |= (it->type == RTYPE::R_WRITE);

                if (miss_merge_reg[key].pending_invalidate.size() > 0) {
                  assert(0 && "This should not happen, if there is pending_invalidate in miss_merge_reg, there will be no more request to same scm row\n");
                }

                Request cache_miss_req = 
                  req_creator->create_hms_request(
                    it->addr, dram_addr,
                    it->num_bytes, it->type, QTYPE::CACHE_MISS, clk, -1, it->mf, it->callback, it->req_receive_time);
                cache_miss_req.original_num_bytes = it->original_num_bytes;
                commonq.q.push_back(cache_miss_req);
                qsize[QTYPE::CACHE_MISS] += 1;
                num_reqs[QTYPE::CACHE_MISS] += it->num_bytes;

                if (it->type == RTYPE::R_READ)
                  num_traffic[QTYPE::CACHE_MISS_RD] += it->num_bytes;
                else if (it->type == RTYPE::R_WRITE)
                  num_traffic[QTYPE::CACHE_MISS_WR] += it->num_bytes;
                else
                  assert(0);
                                  
              }
            }
          }
          erase_list.push_back(it);

        }
      }
    }
  }
    
  // If the request is scheduled, remove the request from the waiting queue.
  for (auto erase_it = erase_list.begin(); erase_it != erase_list.end(); ++erase_it)
    tag_arr_access_queue.q.erase((*erase_it));
}

void HMSController::write_tag_arr() {
  int num_of_sets_per_row_buffer = pow(2, num_dram_cache_offset_bits);
  auto evict_tag_arr_it = write_tag_arr_entry_queue.begin();
  std::vector<std::list<tag_arr_entry_evict_entry>::iterator> erase_list;
  for (; evict_tag_arr_it != write_tag_arr_entry_queue.end(); evict_tag_arr_it++) {
    assert(evict_tag_arr_it->state == cache_block_state::MODIFIED);
    unsigned long evicted_dram_start_addr = get_dram_row_start_addr(evict_tag_arr_it->tag_arr_entry_addr); //tag_arr_entry_addr : RR--RBBBB

    int req_bytes = transfer_bytes_per_clk;
    Request tag_arr_wb_req = 
      req_creator->create_hms_request(
          0, evicted_dram_start_addr, req_bytes,
          RTYPE::R_WRITE, QTYPE::TAG_ARR_WRITE, clk);
    tag_arr_queue.q.push_back(tag_arr_wb_req);
    qsize[QTYPE::TAG_ARR_WRITE] += 1;
    num_reqs[QTYPE::TAG_ARR_WRITE] += req_bytes;

    num_traffic[QTYPE::TAG_ARR_WRITE] += req_bytes;

    num_tag_arr_write_req += 1;

    //store tag info of tag_arr to dram_cache_line
    for(int i = 0; i < num_of_sets_per_row_buffer; i++) {
      unsigned long dram_cache_line_num = get_dram_cache_line_from_dram_row_num(evict_tag_arr_it->tag_arr_entry_addr, i); //tag_arr_entry_addr : RR--RBBBB
      if (evict_tag_arr_it->tag_arr_entry.valid_bits[i]) {
        //current tag_arr is valid - have data
        dram_cache_line[dram_cache_line_num].valid_bits[0] = true;
        dram_cache_line[dram_cache_line_num].tags[0] = evict_tag_arr_it->tag_arr_entry.tags[i];
        dram_cache_line[dram_cache_line_num].dirty_bits[0] = evict_tag_arr_it->tag_arr_entry.dirty_bits[i];

      }
      else {
        //current tag_arr is invalid - empty dram cache line
        dram_cache_line[dram_cache_line_num].valid_bits[0] = false;
        dram_cache_line[dram_cache_line_num].tags[0] = evict_tag_arr_it->tag_arr_entry.tags[i];
        dram_cache_line[dram_cache_line_num].dirty_bits[0] = false;
      }
    }
    erase_list.push_back(evict_tag_arr_it);
  }

  for (auto erase_it = erase_list.begin(); erase_it != erase_list.end(); ++erase_it)
    write_tag_arr_entry_queue.erase((*erase_it));
}

void HMSController::tick() {
  if (print_detail) {
    if (clk % 2000 == 0) {
      int total_line = num_cache_lines * assoc_of_dram_cache;
      int empty_line = get_empty_cache_line();
      int fill_line = total_line - empty_line;

      tag_arr_printer << clk << " num_cacheable_page " << m_gpu->is_global_cached.size() << "(" << m_config->num_dram_pages << ")" << " " 
                       << "num_cache_lines " << fill_line << "(" << num_cache_lines << ")" << " "
                       << "fill_pending_size " << fill_pending.size() << " "
                       << "evict_pending_size " << evict_pending.size() << " "
                       << "num_tag_arr_read " << num_tag_arr_read <<" "
                       << "num_tag_arr_write " << num_tag_arr_write <<" "
                       << "num_tag_arr_hit " << num_tag_arr_hit <<" "
                       << "num_tag_arr_miss " << num_tag_arr_miss << " "
                       << "num_score_reg_made "<<num_score_reg_made<<" "
                       << "num_cache_hit " << num_cache_hit << " "
                       << "num_cache_miss " << num_cache_miss << " " 
                       << "num_fill_empty_after_probe " <<num_fill_empty_after_probe<<" "
                       << "num_replace_old_after_probe " <<num_replace_old_after_probe <<" "
                       << "num_fail_after_probe " << num_fail_after_probe <<" "
                       << "TAG_ARR_READ " << num_reqs[QTYPE::TAG_ARR_READ] <<" "
                       << "TAG_ARR_WRITE " << num_reqs[QTYPE::TAG_ARR_WRITE] <<" "
                       << "CACHE_HIT " <<num_reqs[QTYPE::CACHE_HIT]<<" "
                       << "CACHE_MISS "<<num_reqs[QTYPE::CACHE_MISS]<<" "
                       << "REMAIN_LINE_READ "<<num_reqs[QTYPE::REMAIN_LINE_READ] <<" "
                       << "CACHE_EVICT "<<num_reqs[QTYPE::CACHE_EVICT] <<" "
                       <<"ON_DEMAND_CACHE_PROBE "<<num_reqs[QTYPE::ON_DEMAND_CACHE_PROBE]<<" "
                       <<"ON_DEMAND_CACHE_FILL "<<num_reqs[QTYPE::ON_DEMAND_CACHE_FILL] <<" "
                       <<"ON_DEMAND_CACHE_META_FILL "<<num_reqs[QTYPE::ON_DEMAND_CACHE_META_FILL]<<" "
                       <<"DRAM_CACHE_LINE_READ "<<num_reqs[QTYPE::DRAM_CACHE_LINE_READ]<<" "
                       <<"PCIE_VALIDATE "<<num_reqs[QTYPE::PCIE_VALIDATE]<<" "
                       <<"PCIE_INVALIDATE "<<num_reqs[QTYPE::PCIE_INVALIDATE]<<" "
                       <<"PCIE_INVALIDATE_CACHE_HIT "<<num_reqs[QTYPE::PCIE_INVALIDATE_CACHE_HIT]<<std::endl;
      done = 0;
    }

  }

  clk++;

  if (tag_arr_pending.size()) {
    Request &req = tag_arr_pending[0];
    assert(req.depart != -1);
    if (req.depart <= clk) {
      assert(req.depart != -1);
      if (enable_tag_cache) {
        mem_addr_t dram_addr = req.dram_addr;
        mem_addr_t dram_row_num = get_dram_row_num(dram_addr);
        mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);

        unsigned tag_arr_cache_idx = (unsigned) - 1;
        enum dram_cache_status dram_cache_stat = dram_cache_status::DONT_KNOW;
        enum cache_request_status tag_arr_cache_stat = tag_arr->probe(dram_cache_line_num, tag_arr_cache_idx);

        assert(tag_arr_cache_stat == cache_request_status::HIT_RESERVED);

        int num_of_sets_per_row_buffer = pow(2, num_dram_cache_offset_bits);

        num_tag_arr_write += 1;
        std::map<int, new_addr_type> _tags;
        std::map<int, bool> _dirty_bits;
        std::map<int, bool> _valid_bits;
        //fill tag_arr with new tags in one dram row buffer 
        for(int i = 0; i < num_of_sets_per_row_buffer; i++) {
          unsigned long target_dram_cache_line_num = get_dram_cache_line_from_dram_row_num(dram_row_num, i);
          if (dram_cache_line.find(target_dram_cache_line_num) != dram_cache_line.end()) {
            _tags[i] = dram_cache_line[target_dram_cache_line_num].tags[0];
            _dirty_bits[i] = dram_cache_line[target_dram_cache_line_num].dirty_bits[0];
            _valid_bits[i] = dram_cache_line[target_dram_cache_line_num].valid_bits[0];
          }
          else {
            _tags[i] = ULONG_MAX;
            _dirty_bits[i] = false;
            _valid_bits[i] = false;
          }
        }

        tag_arr->fill(tag_arr_cache_idx, dram_cache_line_num, clk, tag_arr_entry_data(_tags, _dirty_bits, _valid_bits));
        num_tag_arr_write++;

        assert(read_dram_tags_reg.find(dram_row_num) != read_dram_tags_reg.end());
        read_dram_tags_reg.erase(dram_row_num);
      } else {
        //no tag cache
        assert(req.original_type != Type::MAX);
        assert(req.req_scheduled_time > 0);
        assert(req.req_receive_time > 0);
        check_cache_hit(req);
      }

      tag_arr_pending.pop_front();

    }
  }

  if (enable_tag_cache) {
    std::vector<std::deque<std::pair<mem_addr_t, int>>::iterator> evict_erasure;
    auto evict_iter = evict_pending.begin();
    for (; evict_iter != evict_pending.end(); ++evict_iter) {
      assert(ch_type == CH_TYPE::SCMDRAM);
      mem_addr_t evicted_scm_addr = evict_iter->first;

      addrdec_t raw_addr;
      m_config->m_address_mapping.addrdec_tlx(evicted_scm_addr, &raw_addr);
      
      auto evicted_dram_addr_info = get_dram_addr(evicted_scm_addr);
      mem_addr_t evicted_dram_ch = std::get<0>(evicted_dram_addr_info);
      mem_addr_t evicted_dram_addr = std::get<1>(evicted_dram_addr_info);
      int evicted_dram_ch_tag = std::get<2>(evicted_dram_addr_info);

      mem_addr_t evicted_dram_cache_line_num = get_dram_cache_line_num(evicted_dram_addr);
      new_addr_type evicted_dram_tag = get_dram_tag(evicted_scm_addr);

      int evict_target_index = evict_iter->second;
      assert(evict_target_index >= 0 && evict_target_index < assoc_of_dram_cache);

      unsigned tag_arr_cache_idx = (unsigned) - 1;
      enum dram_cache_status dram_cache_stat = dram_cache_status::DONT_KNOW;
      enum cache_request_status tag_arr_cache_stat = tag_arr->access(evicted_dram_cache_line_num, evicted_dram_tag, clk, tag_arr_cache_idx, dram_cache_stat);
      num_tag_arr_read++;

      if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::HIT_RESERVED || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
        if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
          assert(tag_arr->probe(evicted_dram_cache_line_num, tag_arr_cache_idx) == cache_request_status::HIT);
          if (dram_cache_stat == dram_cache_status::MISS_CLEAN || dram_cache_stat == dram_cache_status::MISS_DIRTY || dram_cache_stat == dram_cache_status::MISS_EMPTY) {
            //this can happen. when INVALIDATE request already invalidate evict candidates
            //if this happens, just pass writeback part and erase from evict_pending
          }
          else {
            // If the DRAM cache line is dirty, we need to evict the DRAM cache line to SCM.
            tag_arr->set_col_invalid(evicted_dram_cache_line_num, evicted_dram_tag, clk, tag_arr_cache_idx);
            num_tag_arr_write++;
            
            assert(dram_cache_line.find(evicted_dram_cache_line_num) != dram_cache_line.end());
            dram_cache_line[evicted_dram_cache_line_num].valid_bits[0] = false;
            dram_cache_line[evicted_dram_cache_line_num].dirty_bits[0] = false;
            dram_cache_line[evicted_dram_cache_line_num].tags[0] = ULONG_MAX;
            dram_cache_line[evicted_dram_cache_line_num].dram_ch_tags[0] = ULONG_MAX;
            dram_cache_line[evicted_dram_cache_line_num].scm_addrs[0] = ULONG_MAX;
            dram_cache_line[evicted_dram_cache_line_num].metric_levels[0] = 0;

            if (dram_cache_stat == dram_cache_status::HIT_DIRTY) {
              // Generate dirty write back request
              Request wb_req =
                req_creator->create_hms_request(
                  evicted_scm_addr, evicted_dram_addr, 
                  line_size, RTYPE::R_WRITE, QTYPE::CACHE_EVICT, clk);
              commonq.q.push_back(wb_req); 
              qsize[QTYPE::CACHE_EVICT] += 1;
              num_reqs[QTYPE::CACHE_EVICT] += line_size;

              num_traffic[QTYPE::CACHE_EVICT] += line_size;
            }
          }
          evict_erasure.push_back(evict_iter);
        }
        break;
      } else if (tag_arr_cache_stat == cache_request_status::MISS || tag_arr_cache_stat == cache_request_status::SECTOR_MISS){
        do_tag_arr_miss(evicted_scm_addr, evicted_dram_addr);
        break;
      } else {
        break;
      }
    }
    for (auto erase_it = evict_erasure.begin(); erase_it != evict_erasure.end(); ++erase_it)
      evict_pending.erase(*erase_it);
    evict_erasure.clear();
  } else {
    //no tag cache
    std::vector<std::deque<std::pair<mem_addr_t, int>>::iterator> evict_erasure;
    auto evict_iter = evict_pending.begin();
    for (; evict_iter != evict_pending.end(); ++evict_iter) {
      assert(ch_type == CH_TYPE::SCMDRAM);
      mem_addr_t evicted_scm_addr = evict_iter->first;

      addrdec_t raw_addr;
      m_config->m_address_mapping.addrdec_tlx(evicted_scm_addr, &raw_addr);

      auto evicted_dram_addr_info = get_dram_addr(evicted_scm_addr);
      mem_addr_t evicted_dram_ch = std::get<0>(evicted_dram_addr_info);
      mem_addr_t evicted_dram_addr = std::get<1>(evicted_dram_addr_info);
      int evicted_dram_ch_tag = std::get<2>(evicted_dram_addr_info);

      mem_addr_t evicted_dram_cache_line_num = get_dram_cache_line_num(evicted_dram_addr);
      new_addr_type evicted_dram_tag = get_dram_tag(evicted_scm_addr);

      int evict_target_index = evict_iter->second;
      assert(evict_target_index >= 0 && evict_target_index < assoc_of_dram_cache);

      if (dram_cache_line.find(evicted_dram_cache_line_num) != dram_cache_line.end()) {
        assert(dram_cache_line[evicted_dram_cache_line_num].valid_bits.find(0) != dram_cache_line[evicted_dram_cache_line_num].valid_bits.end());
        assert(dram_cache_line[evicted_dram_cache_line_num].dirty_bits.find(0) != dram_cache_line[evicted_dram_cache_line_num].dirty_bits.end());
        assert(dram_cache_line[evicted_dram_cache_line_num].tags.find(0) != dram_cache_line[evicted_dram_cache_line_num].tags.end());
        
        //make TAG_ARR_WRITE request
        Request tag_arr_wb_req = 
          req_creator->create_hms_request(
              0, evicted_dram_addr, transfer_bytes_per_clk,
              RTYPE::R_WRITE, QTYPE::TAG_ARR_WRITE, clk);
        tag_arr_queue.q.push_back(tag_arr_wb_req);
        qsize[QTYPE::TAG_ARR_WRITE] += 1;
        num_reqs[QTYPE::TAG_ARR_WRITE] += transfer_bytes_per_clk;

        num_traffic[QTYPE::TAG_ARR_WRITE] += transfer_bytes_per_clk;
        

        bool evict_target_dirty = false;
        if (dram_cache_line[evicted_dram_cache_line_num].dirty_bits[0]) {
          evict_target_dirty = true;
        }

        //evict from dram cache line
        dram_cache_line[evicted_dram_cache_line_num].valid_bits[0] = false;
        dram_cache_line[evicted_dram_cache_line_num].dirty_bits[0] = false;
        dram_cache_line[evicted_dram_cache_line_num].tags[0] = ULONG_MAX;
        dram_cache_line[evicted_dram_cache_line_num].dram_ch_tags[0] = ULONG_MAX;
        dram_cache_line[evicted_dram_cache_line_num].scm_addrs[0] = ULONG_MAX;
        dram_cache_line[evicted_dram_cache_line_num].metric_levels[0] = 0;

        if (evict_target_dirty) {
          // Generate dirty write back request
          Request wb_req =
            req_creator->create_hms_request(
              evicted_scm_addr, evicted_dram_addr, 
              line_size, RTYPE::R_WRITE, QTYPE::CACHE_EVICT, clk);
        
          commonq.q.push_back(wb_req); 
          qsize[QTYPE::CACHE_EVICT] += 1;
          num_reqs[QTYPE::CACHE_EVICT] += line_size;

          num_traffic[QTYPE::CACHE_EVICT] += line_size;          
        }
      }

      evict_erasure.push_back(evict_iter);
      break;
    }
    for (auto erase_it = evict_erasure.begin(); erase_it != evict_erasure.end(); ++erase_it)
      evict_pending.erase(*erase_it);
    evict_erasure.clear();
  }

  if (enable_tag_cache) {
    std::vector<std::deque<std::pair<mem_addr_t, int>>::iterator> fill_erasure;
    auto fill_iter = fill_pending.begin();
    for (; fill_iter != fill_pending.end(); ++fill_iter) {
      assert(ch_type == CH_TYPE::SCMDRAM);
      mem_addr_t scm_row_addr = fill_iter->first;
      mem_addr_t scm_row_num = get_scm_row_num(scm_row_addr);

      addrdec_t raw_addr;
      m_config->m_address_mapping.addrdec_tlx(scm_row_addr, &raw_addr);

      unsigned scm_rank = raw_addr.rank;
      unsigned scm_ch = raw_addr.chip;
      auto key = std::make_tuple(scm_row_num, scm_rank, scm_ch);

      auto dram_addr_info = get_dram_addr(scm_row_addr);
      mem_addr_t dram_ch = std::get<0>(dram_addr_info);
      mem_addr_t dram_addr = std::get<1>(dram_addr_info);
      int dram_ch_tag = std::get<2>(dram_addr_info);
      mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
      new_addr_type dram_tag = get_dram_tag(scm_row_addr);

      int target_index = fill_iter->second;
      assert(target_index >= 0 && target_index < assoc_of_dram_cache);

      unsigned tag_arr_cache_idx = (unsigned) - 1;
      enum dram_cache_status dram_cache_stat = dram_cache_status::DONT_KNOW;
      enum cache_request_status tag_arr_cache_stat = tag_arr->access(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx, dram_cache_stat);
      num_tag_arr_read++;

      if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::HIT_RESERVED || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
        if (tag_arr_cache_stat == cache_request_status::HIT || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_MISS || tag_arr_cache_stat == cache_request_status::EVICT_BUFFER_HIT_SECTOR_MISS) {
          assert(tag_arr->probe(dram_cache_line_num, tag_arr_cache_idx) == cache_request_status::HIT);
          assert(probe_candidate.find(key) != probe_candidate.end());

          if (probe_candidate[key].pending_invalidate.size() > 0) {
            assert(probe_candidate[key].pending_invalidate.size() == 1);
            //if there is pending invalidate at this key, stop this process and just callback pending_invalidate
            //have to stop here
            //and have to callback pending_invalidate

            Request & pending_invalidate_req = probe_candidate[key].pending_invalidate.front();
            assert(pending_invalidate_req.qtype == QTYPE::PCIE_INVALIDATE);
            assert(pending_invalidate_req.callback != nullptr);
            pending_invalidate_req.callback(pending_invalidate_req);
          
            probe_candidate[key].pending_invalidate.pop_back();
            assert(probe_candidate[key].pending_invalidate.size() == 0);

            probe_candidate.erase(key);

            fill_erasure.push_back(fill_iter);
          }
          else {
            tag_arr->set_col(dram_cache_line_num, dram_tag, clk, tag_arr_cache_idx, probe_candidate[key].dirty);            
            num_tag_arr_write += 1;

            fill_cache_line(scm_row_addr, probe_candidate[key].metric_level, target_index);
            assert(probe_candidate[key].pending.empty());
            assert(probe_candidate[key].pending_invalidate.size() == 0);

            probe_candidate.erase(key);
            fill_erasure.push_back(fill_iter);
          }
        }
        break;
      } else if (tag_arr_cache_stat == cache_request_status::MISS || tag_arr_cache_stat == cache_request_status::SECTOR_MISS) {
        do_tag_arr_miss(scm_row_addr, dram_addr);
        break;
      } else {
        break;
      }
    }
    for (auto erase_it = fill_erasure.begin(); erase_it != fill_erasure.end(); ++erase_it)
      fill_pending.erase(*erase_it);
    fill_erasure.clear();
  } else {
    //no tag cache
    std::vector<std::deque<std::pair<mem_addr_t, int>>::iterator> fill_erasure;
    auto fill_iter = fill_pending.begin();
    for (; fill_iter != fill_pending.end(); ++fill_iter) {
      assert(ch_type == CH_TYPE::SCMDRAM);
      mem_addr_t scm_row_addr = fill_iter->first;
      mem_addr_t scm_row_num = get_scm_row_num(scm_row_addr);

      addrdec_t raw_addr;
      m_config->m_address_mapping.addrdec_tlx(scm_row_addr, &raw_addr);

      unsigned scm_rank = raw_addr.rank;
      unsigned scm_ch = raw_addr.chip;
      auto key = std::make_tuple(scm_row_num, scm_rank, scm_ch);

      int target_index = fill_iter->second;
      assert(target_index >= 0 && target_index < assoc_of_dram_cache);
      assert(probe_candidate.find(key) != probe_candidate.end());
      fill_cache_line(scm_row_addr, probe_candidate[key].metric_level, target_index);
      assert(probe_candidate[key].pending.empty());
      assert(probe_candidate[key].pending_invalidate.size() == 0);

      probe_candidate.erase(key);
      fill_erasure.push_back(fill_iter);

      break;
    }
    for (auto erase_it = fill_erasure.begin(); erase_it != fill_erasure.end(); ++erase_it)
      fill_pending.erase(*erase_it);
    fill_erasure.clear();
  }

  if (pcie_pending.size()) {
    Request &req = pcie_pending[0];
    assert(req.depart != -1);
    if (req.depart <= clk) {
      assert(req.depart != -1);
      assert(req.callback != nullptr);
      req.callback(req);
      pcie_pending.pop_front();
    }
  }

  if (pcie_invalidate_cache_hit_pending.size()) {
    Request &req = pcie_invalidate_cache_hit_pending[0];
    assert(req.depart != -1);
    if (req.depart <= clk) {
      assert(req.depart != -1);
      mem_addr_t scm_row_num = get_scm_row_num(req.addr);
      auto dram_addr_info = get_dram_addr(req.addr);
      mem_addr_t dram_ch = std::get<0>(dram_addr_info);
      mem_addr_t dram_addr = std::get<1>(dram_addr_info);
      int dram_ch_tag = std::get<2>(dram_addr_info);
      assert(dram_addr == req.dram_addr);
      mem_addr_t dram_cache_line_num = get_dram_cache_line_num(req.dram_addr);
      new_addr_type dram_tag = get_dram_tag(req.addr);
      assert(req.callback != nullptr);
      req.callback(req);

      pcie_invalidate_cache_hit_pending.pop_front();
    }
  }

  if (line_read_pending.size()) {
    Request &req = line_read_pending[0];
    assert(req.depart != -1);
    if (req.depart <= clk) {
      mem_addr_t scm_row_num = get_scm_row_num(req.addr);
      addrdec_t raw_addr;
      m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
      unsigned scm_rank = req.addr_vec[int(HMS::Level::Rank)];
      auto key = std::make_tuple(scm_row_num, scm_rank, ch_id);

      assert(probe_candidate.find(key) != probe_candidate.end());
      for (Request &req : probe_candidate[key].pending) {
        assert(req.callback != nullptr);
        req.callback(req);
        probe_candidate[key].dirty |= (req.type == RTYPE::R_WRITE);
        if (num_traffic[QTYPE::REMAIN_LINE_READ] > req.num_bytes) {
          num_traffic[QTYPE::REMAIN_LINE_READ] -= req.num_bytes;
          if (req.type == RTYPE::R_READ)
            num_traffic[QTYPE::CACHE_MISS_RD] += req.num_bytes;
          else if (req.type == RTYPE::R_WRITE)
            num_traffic[QTYPE::CACHE_MISS_WR] += req.num_bytes;
          else
            assert(0);
        }
      }
      probe_candidate[key].pending.clear();
      probe_candidate[key].line_read_done = true;

      if (probe_candidate[key].pending_invalidate.size() > 0) {
        assert(probe_candidate[key].pending_invalidate.size() == 1);
        //have to stop here
        //and have to callback pending_invalidate
        
        Request & pending_invalidate_req = probe_candidate[key].pending_invalidate.front();
        assert(pending_invalidate_req.qtype == QTYPE::PCIE_INVALIDATE);
        assert(pending_invalidate_req.callback != nullptr);
        assert(pending_invalidate_req.target_pending == &pcie_pending);
        pending_invalidate_req.callback(pending_invalidate_req);
        probe_candidate[key].pending_invalidate.pop_back();
        assert(probe_candidate[key].pending_invalidate.size() == 0);

        probe_candidate.erase(key);
      }
      else {
        do_cache_line_replacement(req);
      }

      line_read_pending.pop_front();
    }
  }

  if (cache_probe_pending.size()) {
    assert(ch_type == CH_TYPE::SCMDRAM);
    Request &req = cache_probe_pending[0];
    assert(req.depart != -1);
    if (req.depart <= clk) {
      assert(req.depart != -1);
      if (req.qtype == QTYPE::ON_DEMAND_CACHE_PROBE) {
        check_old_cache_and_evict(req);
      } 
      else {
        assert(0 && "CANNOT HAPPEN\n");
      }
      cache_probe_pending.pop_front();
    }
  }

  if (read_dram_cache_line_pending.size()) {
    Request &req = read_dram_cache_line_pending[0];
    assert(req.depart != -1);
    if (req.depart <= clk) {
      assert(req.depart != -1);
      assert(req.evict_target_index >= 0 && req.evict_target_index < assoc_of_dram_cache);
      mem_addr_t scm_row_addr = req.addr;
      mem_addr_t scm_row_num = get_scm_row_num(req.addr);
      auto dram_addr_info = get_dram_addr(req.addr);
      mem_addr_t dram_ch = std::get<0>(dram_addr_info);
      mem_addr_t dram_addr = std::get<1>(dram_addr_info);
      int dram_ch_tag = std::get<2>(dram_addr_info);
      assert(dram_addr == req.dram_addr);
      mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);

      addrdec_t raw_addr;
      m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);

      unsigned scm_rank = raw_addr.rank;
      unsigned scm_ch = raw_addr.chip;
      auto key = std::make_tuple(scm_row_num, scm_rank, scm_ch);

      assert(probe_candidate.find(key) != probe_candidate.end());
      if (probe_candidate[key].pending_invalidate.size() > 0) {
        assert(probe_candidate[key].pending_invalidate.size() == 1);
        //if there is pending invalidate at this key, stop this process and just callback pending_invalidate
        //have to stop here
        //and have to callback pending_invalidate

        Request & pending_invalidate_req = probe_candidate[key].pending_invalidate.front();
        assert(pending_invalidate_req.qtype == QTYPE::PCIE_INVALIDATE);       
        assert(pending_invalidate_req.callback != nullptr);
        pending_invalidate_req.callback(pending_invalidate_req);
        
        probe_candidate[key].pending_invalidate.pop_back();
        assert(probe_candidate[key].pending_invalidate.size() == 0);

        probe_candidate.erase(key);
      } else {
        if (dram_cache_line.find(dram_cache_line_num) != dram_cache_line.end()) {      
          std::map<int, new_addr_type> tags = dram_cache_line[dram_cache_line_num].tags;
          std::map<int, mem_addr_t> scm_addrs = dram_cache_line[dram_cache_line_num].scm_addrs;
          std::map<int, int> dram_ch_tags = dram_cache_line[dram_cache_line_num].dram_ch_tags;
          std::map<int, bool> valid_bits = dram_cache_line[dram_cache_line_num].valid_bits;

          if (tags.find(req.evict_target_index) != tags.end()) {
            assert(scm_addrs.find(req.evict_target_index) != scm_addrs.end());
            assert(dram_ch_tags.find(req.evict_target_index) != dram_ch_tags.end());
            assert(valid_bits.find(req.evict_target_index) != valid_bits.end());

            if (valid_bits[req.evict_target_index] == true) {
              unsigned long target_tag = tags[req.evict_target_index];
              int target_dram_ch_tag = dram_ch_tags[req.evict_target_index];         

              mem_addr_t evicted_scm_addr = scm_addrs[req.evict_target_index];
              assert(evicted_scm_addr != ULONG_MAX);
              assert(req.evict_target_index == 0);
              mem_addr_t evicted_scm_start_addr = (evicted_scm_addr>>line_size_log_bits)<<line_size_log_bits;

              evict_pending.push_back(make_pair(evicted_scm_start_addr, req.evict_target_index));
            }
          }
        }

        if (req.qtype == QTYPE::DRAM_CACHE_LINE_READ) {
          fill_pending.push_back(make_pair(req.addr, req.evict_target_index));
        }
        else {
          assert(0 &&"CANNOT HAPPEN\n");
        }
      }
      read_dram_cache_line_pending.pop_front();
    }
  }
  /*** 1. Serve completed reads ***/
  //core_pending is used for CACHE_HIT, CACHE_MISS
  if (core_pending.size()) {
    Request& req = core_pending[0];
    assert(req.depart != -1);
    if (req.depart <= clk) {  // if the r/w latency meets
      assert(req.depart != -1);
      if (req.qtype == QTYPE::CACHE_HIT) {
        if (req.callback != nullptr) {
          req.callback(req);
        }
      } else if (req.qtype == QTYPE::CACHE_MISS) {
        if (req.callback != nullptr) {
          req.callback(req);
        }
      } else {
        assert(0 && "Invalid qtype\n");
      }
      core_pending.pop_front();
    }
  }

  if (enable_tag_cache) {
    access_tag_arr();
    write_tag_arr(); //writeback tag array data
  }

  /*** 2. Refresh scheduler ***/
  refresh->tick_ref();

  if (m_config->seperate_write_queue_enabled) {
    /*** 3. Should we schedule writes? ***/
    if (!write_mode) {
      // yes -- write queue is almost full or read queue is empty
      if (writeq.size() > int(wr_high_watermark * writeq.max) || readq.size() == 0)
          write_mode = true;
    } else {
      // no -- write queue is almost empty and read queue is not empty
      if (writeq.size() < int(wr_low_watermark * writeq.max) && readq.size() != 0)
          write_mode = false;
    }
  }


  /*** 4. Find the best command to schedule, if any ***/

  // First check the actq (which has higher priority) to see if there
  // are requests available to service in this cycle
  Queue* queue = &actq;
  HMS::Command cmd;
  auto req = scheduler->get_head(queue->q);

  bool is_valid_req = (req != queue->q.end());

  if (is_valid_req) {
    cmd = get_first_cmd(req);
    is_valid_req = is_ready(cmd, req->addr_vec);
  }
  // Priority 1: TagArray queue (Tag Arr queue)

   if (!is_valid_req) {
    queue = &tag_arr_queue;
    if (otherq.size())
      queue = &otherq;
    
    req = scheduler->get_head(queue->q);
    is_valid_req = (req != queue->q.end());
    if (is_valid_req) {
      cmd = get_first_cmd(req);
      is_valid_req = is_ready(cmd, req->addr_vec);
    }
  }

  if (!is_valid_req) {
    queue = &pcie_dram_queue;
    if (otherq.size())
      queue = &otherq;

    req = scheduler->get_head(queue->q);
    is_valid_req = (req != queue->q.end());
    if (is_valid_req) {
      cmd = get_first_cmd(req);
      is_valid_req = is_ready(cmd, req->addr_vec);
    }
  }
  // PCIE invalidation queue
  if (!is_valid_req) {
    queue = &pcie_queue;
    if (otherq.size())
      queue = &otherq;

    req = scheduler->get_head(queue->q);
    is_valid_req = (req != queue->q.end());
    if (is_valid_req) {
      cmd = get_first_cmd(req);
      is_valid_req = is_ready(cmd, req->addr_vec);
    }
  }
  // Others (including pcie read data caching)
  if (!is_valid_req) {
    queue = &commonq;
    if (otherq.size())
      queue = &otherq;
    req = scheduler->get_head(queue->q);
    is_valid_req = (req != queue->q.end());
    if (is_valid_req) {
      cmd = get_first_cmd(req);
      is_valid_req = is_ready(cmd, req->addr_vec);
    }
  }

  if (!is_valid_req) {
      // we couldn't find a command to schedule -- let's try to be speculative
    auto cmd = HMS::Command::PRE;
    vector<int> victim = rowpolicy->get_victim(cmd);
    if (!victim.empty()) {
        issue_cmd(cmd, victim);
    }
    return;  // nothing more to be done this cycle
  }

  // issue command on behalf of request
  issue_cmd(cmd, get_addr_vec(cmd, req));

  if (cmd == HMS::Command::ACT) {
    total_num_activates ++;
  } else if (cmd == HMS::Command::RD || cmd == HMS::Command::WR) {
    total_row_accesses ++;
  }

  // check whether this is the last command (which finishes the request)
  if (cmd != channel->spec->translate[int(req->type)]) {
    if (channel->spec->is_opening(cmd)) {
      // promote the request that caused issuing activation to actq
      actq.q.push_back(*req);
      queue->q.erase(req);
    }
    return;
  }

  if (req->req_scheduled_time == 0) {
    req->req_scheduled_time = clk;
  }

  if (req->type == RTYPE::R_READ ||
      req->type == RTYPE::R_WRITE) {
    req->num_bytes -= transfer_bytes_per_clk;
    assert(req->num_bytes >= 0);
    if (req->num_bytes == 0) {
      req->done = true;
      if (req->type == RTYPE::R_READ) {
        req->depart = clk + channel->spec->read_latency;
      } else if (req->type == RTYPE::R_WRITE) {
        req->depart = clk + channel->spec->write_latency;
      }
    }
  } else {
    assert(req->num_bytes == 0);
    req->done = true;
  }

  if (post_processor.find(req->qtype) != post_processor.end())
    post_processor[req->qtype](*req);
  // Post-process requests
  // All read requests must go through pending queue
  if (req->type == RTYPE::R_READ)  {
    if (req->done) {
      assert(req->target_pending != nullptr);
      req->target_pending->push_back(*req);
    }
  } else if (req->type == RTYPE::R_WRITE) {
    if (req->done) {
      if (req->callback != nullptr) {
        req->callback(*req);
      }        
    }
  }


  if (req->done) {
    if (req->type == RTYPE::R_READ || req->type == RTYPE::R_WRITE) {
      assert(qsize[req->qtype] > 0);
      qsize[req->qtype] -= 1;
    }
    if (req->qtype != QTYPE::PCIE_INVALIDATE && req->qtype != QTYPE::PCIE_VALIDATE && req->qtype != QTYPE::PCIE_INVALIDATE_CACHE_HIT) {
      done += 1;
    }
    queue->q.erase(req);
  }
}

float HMSController::calculate_score(unsigned num_col_accesses,
                                     bool is_dirty) {
  if (num_col_accesses > (line_size / transfer_bytes_per_clk))
    num_col_accesses = line_size / transfer_bytes_per_clk;

  HMS::SCM_SpeedEntry &s = channel->spec->scm_speed_entry;
  HMS::DRAM_SpeedEntry &d = channel->spec->dram_speed_entry;

  int latency_diff = 0; 
  if (is_dirty) {
    latency_diff = (s.nRCDR - d.nRCDR) + (s.nWR - d.nWR);
  } else {
    latency_diff = (s.nRCDR - d.nRCDR);
  }
  assert(latency_diff > 0);
  float scm_penalty_score = (float)latency_diff / (float)num_col_accesses;

  return scm_penalty_score;
}

int HMSController::descritize_metric(float metric, float metric_max) {
  float interval = metric_max / (float)LEVEL;
  for (int i = 0; i < LEVEL; ++i) {
    float min = (float) i * interval;
    float max = (float) (i + 1) * interval;
    if (min <= metric && metric < max)
      return i;
  }
  return LEVEL - 1;
}


bool HMSController::drop_counter(unsigned long counter) {
  std::random_device rd;
  std::uniform_int_distribution<int> distribution(1, 10000);
  std::mt19937 engine(rd());

  int value = distribution(engine);
  double prob = (((double) counter / ((double)max_counter)) * 100.0);
  prob /= ((double)800 / LEVEL);
  prob *= 100.0;
  if (value <= prob) {
    cur_threshold = prob;
    return true;
  } else {
    return false;
  }
}

void HMSController::post_process_cache_hit(const Request& req) {
  assert(ch_type == CH_TYPE::SCMDRAM);

  mem_addr_t scm_row_num = get_scm_row_num(req.addr);

  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
  unsigned scm_rank = raw_addr.rank;
  unsigned scm_ch_id = raw_addr.chip;
  auto key = std::make_tuple(scm_row_num, scm_rank, scm_ch_id);
  assert(hit_merge_reg.find(key) != hit_merge_reg.end());
  if (score_reg.find(key) == score_reg.end()) {
    req.mf->increase_activation_counter = true;
    if (act_cnt_2MB) {
      //update act_cnt
      mem_addr_t page_num = calc_page_num(req.addr);
      update_act_cnt(page_num);
    }

    float score = calculate_score(hit_merge_reg[key].num_cols, 
                                  hit_merge_reg[key].dirty);
    score_reg[key].score = score;
    score_reg[key].score_level = descritize_metric(score, max_score);
  }
  assert(hit_merge_reg[key].count > 0);
  hit_merge_reg[key].count -= 1;
  // If there are no more pending requests for this cache line,
  // we need to update memory controller status registers.
  unsigned long activation_counter = req.mf->activation_counter;
  if (hit_merge_reg[key].count == 0) {
    if (act_cnt_2MB) {
      activation_counter = get_act_cnt(calc_page_num(req.addr));
      assert(activation_counter >= 0);
    }
    // update the maximum value of the memory controller
    max_counter = std::max(activation_counter, max_counter);
    max_score = std::max(score_reg[key].score, max_score);
    max_mix = std::max(score_reg[key].score * activation_counter, max_mix);

    min_counter = std::min(activation_counter, min_counter);
    min_score = std::min(score_reg[key].score, min_score);
    min_mix = std::min(score_reg[key].score * activation_counter, min_mix);

    update_global_score(score_reg[key].score);

    hit_merge_reg.erase(key);
    score_reg.erase(key);
  }
}

void HMSController::post_process_cache_miss(const Request& req) {
  mem_addr_t scm_row_num = get_scm_row_num(req.addr);
  std::tuple<unsigned long, unsigned, unsigned> key;
  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
  assert(req.addr_vec.size() == int(HMS::Level::MAX));
  unsigned scm_rank = req.addr_vec[int(HMS::Level::Rank)];
  assert(scm_rank == raw_addr.rank);
  assert(raw_addr.chip == ch_id);
  key = std::make_tuple(scm_row_num, scm_rank, ch_id);

  if (miss_merge_reg.find(key) == miss_merge_reg.end()) {
    return;
  }
  if (score_reg.find(key) == score_reg.end()) {
    num_score_reg_made += 1;
    req.mf->increase_activation_counter = true;
    if (act_cnt_2MB) {
      //update act_cnt
      mem_addr_t page_num = calc_page_num(req.addr);
      update_act_cnt(page_num);
    }

    float score = calculate_score(miss_merge_reg[key].num_cols,
                                  miss_merge_reg[key].dirty);
    
    score_reg[key].score = score;
    score_reg[key].activation_counter = req.mf->activation_counter;
    score_reg[key].dirty = miss_merge_reg[key].dirty;
    score_reg[key].num_cols = miss_merge_reg[key].num_cols;

    if (act_cnt_2MB) {
      unsigned long activation_counter = get_act_cnt(calc_page_num(req.addr));
      assert(activation_counter >= 0);
      score_reg[key].activation_counter = activation_counter;
    }
    score_reg[key].score_level = descritize_metric(score, max_score);
  
    // We check whether the current cache line can be candidate to probe the
    // DRAM cache line.
    bool is_candidate = false;
    int global_score_level = descritize_metric(global_score, max_score);
    global_penalty_level = global_score_level;
    if (policy == POLICY::ALWAYS) {
      is_candidate = true;
    } else {
      is_candidate = score_reg[key].score_level > global_score_level;
    }

    if (is_candidate && (probe_candidate.find(key) == probe_candidate.end()) && (miss_merge_reg[key].pending_invalidate.size() == 0)) {
      // The cache line is candidate to probe the DRAM cache line.
      // if there exists pending invalidate for this key, we cannot make probe_candidate for this
      unsigned long activation_counter = req.mf->activation_counter;
      if (act_cnt_2MB) {
        activation_counter = get_act_cnt(calc_page_num(req.addr));
        assert(activation_counter >= 0);
      }
      probe_candidate[key].num_cols = 1;
      probe_candidate[key].metric = score;
      if (policy == POLICY::IDEAL_ACCESS_CNT) {
        probe_candidate[key].metric = score * activation_counter;
      }

      probe_candidate[key].activation_counter = activation_counter;
      probe_candidate[key].dirty = miss_merge_reg[key].dirty;
      probe_candidate[key].line_read_done = false;
      probe_candidate[key].trigger_lock = miss_merge_reg[key].trigger_lock;
      probe_candidate[key].metric_level = 0;

      //fill evict information
      probe_candidate[key].have_to_evict = miss_merge_reg[key].have_to_evict;
      probe_candidate[key].evict_target_idx = miss_merge_reg[key].evict_target_idx;
      probe_candidate[key].evict_target_dirty = miss_merge_reg[key].evict_target_dirty;
    }
  } else {
    // update score with new num_cols, dirty
    if (probe_candidate.find(key) != probe_candidate.end()) {
      probe_candidate[key].num_cols += 1;
    }
  }

  assert(miss_merge_reg[key].count > 0);
  miss_merge_reg[key].count -= 1;
  // If there are no more pending requests for this cache line,
  // we need to update memory controller status registers.
  if (miss_merge_reg[key].count == 0) {
    assert(score_reg.find(key) != score_reg.end());
    unsigned long activation_counter = req.mf->activation_counter;
    if (act_cnt_2MB) {
      activation_counter = get_act_cnt(calc_page_num(req.addr));
      assert(activation_counter >= 0);
    }
    max_counter = std::max(activation_counter, max_counter);
    max_score = std::max(score_reg[key].score, max_score);
    max_mix = std::max(score_reg[key].score * activation_counter, max_mix);

    min_counter = std::min(activation_counter, min_counter);
    min_score = std::min(score_reg[key].score, min_score);
    min_mix = std::min(score_reg[key].score * activation_counter, min_mix);
  
    update_global_score(score_reg[key].score);  
    
    if (probe_candidate.find(key) != probe_candidate.end()) {
      // If we need to read more data within the cache line
      auto dram_addr_info = get_dram_addr(req.addr);
      mem_addr_t dram_ch = std::get<0>(dram_addr_info);
      mem_addr_t dram_addr = std::get<1>(dram_addr_info);
      int dram_ch_tag = std::get<2>(dram_addr_info);
      assert(dram_addr == req.dram_addr);
      mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
      
      if (probe_candidate[key].num_cols < (line_size / transfer_bytes_per_clk)) {
        unsigned num_bytes_to_read = 
          (line_size - probe_candidate[key].num_cols * transfer_bytes_per_clk);
        
        if (num_bytes_to_read <= 0) {
          line_read_pending.push_back(req);
        } else {     
          auto dram_addr_info = get_dram_addr(req.addr);
          mem_addr_t dram_ch = std::get<0>(dram_addr_info);
          mem_addr_t dram_addr = std::get<1>(dram_addr_info);
          int dram_ch_tag = std::get<2>(dram_addr_info);
          assert(dram_addr == req.dram_addr);

          Request remain_read_req = 
            req_creator->create_hms_request(
              req.addr, dram_addr, num_bytes_to_read,
              RTYPE::R_READ, QTYPE::REMAIN_LINE_READ, clk);

          commonq.q.push_back(remain_read_req);
          qsize[QTYPE::REMAIN_LINE_READ] += 1;
          num_reqs[QTYPE::REMAIN_LINE_READ] += num_bytes_to_read;

          num_traffic[QTYPE::REMAIN_LINE_READ] += num_bytes_to_read;
        }
        
      } else {
        line_read_pending.push_back(req);
      }
    }
    else {
      //not exist probe_candidate
      if (miss_merge_reg[key].pending_invalidate.size() != 0) {
        //if there is pending invalidate for this key
        assert(miss_merge_reg[key].pending_invalidate.size() == 1);
        Request & pending_invalidate_req = miss_merge_reg[key].pending_invalidate.front();
        assert(pending_invalidate_req.qtype == QTYPE::PCIE_INVALIDATE);
        assert(pending_invalidate_req.callback != nullptr);
        assert(pending_invalidate_req.target_pending == &pcie_pending);
        assert(miss_merge_reg[key].num_cols > 0);
        pending_invalidate_req.num_bytes -= miss_merge_reg[key].num_cols * transfer_bytes_per_clk;
        if (pending_invalidate_req.num_bytes <= 0) {
          if (pending_invalidate_req.num_bytes < 0) {
            pending_invalidate_req.num_bytes = 0;
          }
          assert(pending_invalidate_req.num_bytes == 0);
          pending_invalidate_req.depart = req.depart;
          assert(pending_invalidate_req.depart > 0);
          pcie_pending.push_back(pending_invalidate_req);
        } else {
          assert(pending_invalidate_req.num_bytes > 0);
          pcie_queue.q.push_back(pending_invalidate_req);
          qsize[QTYPE::PCIE_INVALIDATE] += 1;
          num_reqs[QTYPE::PCIE_INVALIDATE] += pending_invalidate_req.num_bytes;
        }
        miss_merge_reg[key].pending_invalidate.pop_back();
        assert(miss_merge_reg[key].pending_invalidate.size() == 0);
      }
    }
    miss_merge_reg.erase(key);
    score_reg.erase(key);
  }
}

void HMSController::post_process_remain_line_read(const Request& req) {
  //REMAIN_LINE_READ
  mem_addr_t scm_row_num = get_scm_row_num(req.addr); 
  std::tuple<unsigned long, unsigned, unsigned> key;
  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
  assert(req.addr_vec.size() == int(HMS::Level::MAX));
  unsigned scm_rank = req.addr_vec[int(HMS::Level::Rank)];
  assert(scm_rank == raw_addr.rank);
  assert(raw_addr.chip == ch_id);
  key = std::make_tuple(scm_row_num, scm_rank, ch_id);

  assert(probe_candidate.find(key) != probe_candidate.end());
  assert(probe_candidate[key].num_cols >= 0);
  probe_candidate[key].num_cols += 1;
  
}

void HMSController::fill_cache_line(mem_addr_t scm_addr, int new_metric_level, int index) {
  assert(ch_type == CH_TYPE::SCMDRAM);
  mem_addr_t scm_row_num = get_scm_row_num(scm_addr);

  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(scm_addr, &raw_addr);
  
  auto dram_addr_info = get_dram_addr(scm_addr);
  mem_addr_t dram_ch = std::get<0>(dram_addr_info);
  mem_addr_t dram_addr = std::get<1>(dram_addr_info);
  int dram_ch_tag = std::get<2>(dram_addr_info);
  mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
  new_addr_type dram_tag = get_dram_tag(scm_addr);

  unsigned req_bytes = line_size;

  if (!is_last_dram_cache_line(dram_cache_line_num)) {
    //if this cacheline is not first dram cacheline, have to write one more 32B to update current cacheline's DRAM affinity level.
    req_bytes += transfer_bytes_per_clk;
  }

  Request cache_fill_req =
    req_creator->create_hms_request(
        scm_addr, dram_addr,
        req_bytes, RTYPE::R_WRITE, QTYPE::ON_DEMAND_CACHE_FILL, clk); //write to DRAM cache line
  commonq.q.push_back(cache_fill_req);
  qsize[QTYPE::ON_DEMAND_CACHE_FILL] += 1;
  num_reqs[QTYPE::ON_DEMAND_CACHE_FILL] += req_bytes;

  num_traffic[QTYPE::ON_DEMAND_CACHE_FILL] += req_bytes;//DRAM write -migration

  (dram_cache_line[dram_cache_line_num].tags)[index] = dram_tag;
  (dram_cache_line[dram_cache_line_num].dram_ch_tags)[index] = dram_ch_tag;
  (dram_cache_line[dram_cache_line_num].scm_addrs)[index] = scm_addr;//source scm_addr
  (dram_cache_line[dram_cache_line_num].metric_levels)[index] = new_metric_level;
  (dram_cache_line[dram_cache_line_num].dirty_bits)[index] = false;
  (dram_cache_line[dram_cache_line_num].valid_bits)[index] = true;
}

void HMSController::fill_cache_line_pcie(Request& req, int new_metric_level, int index) {
  assert(ch_type == CH_TYPE::SCMDRAM);
  mem_addr_t scm_addr = req.addr;
  mem_addr_t scm_row_num = get_scm_row_num(scm_addr);

  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(scm_addr, &raw_addr);

  auto dram_addr_info = get_dram_addr(scm_addr);
  mem_addr_t dram_ch = std::get<0>(dram_addr_info);
  mem_addr_t dram_addr = std::get<1>(dram_addr_info);
  int dram_ch_tag = std::get<2>(dram_addr_info);
  mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);
  new_addr_type dram_tag = get_dram_tag(scm_addr);
  assert(dram_addr == req.dram_addr);
  assert(req.callback != nullptr);
  Request cache_fill_req =
    req_creator->create_hms_request(
        scm_addr, dram_addr,
        line_size, RTYPE::R_WRITE, QTYPE::ON_DEMAND_CACHE_FILL, clk, -1, req.mf, req.callback, req.req_receive_time); //write to DRAM cache line
  cache_fill_req.original_num_bytes = req.original_num_bytes;
  commonq.q.push_back(cache_fill_req);
  qsize[QTYPE::ON_DEMAND_CACHE_FILL] += 1;
  num_reqs[QTYPE::ON_DEMAND_CACHE_FILL] += line_size;

  (dram_cache_line[dram_cache_line_num].tags)[index] = dram_tag;
  (dram_cache_line[dram_cache_line_num].dram_ch_tags)[index] = dram_ch_tag;
  (dram_cache_line[dram_cache_line_num].scm_addrs)[index] = scm_addr;//source scm_addr
  (dram_cache_line[dram_cache_line_num].metric_levels)[index] = new_metric_level;
  (dram_cache_line[dram_cache_line_num].dirty_bits)[index] = false;
  (dram_cache_line[dram_cache_line_num].valid_bits)[index] = true;
}

void HMSController::fill_cache_line_metadata(mem_addr_t scm_row_addr) {
  assert(ch_type == CH_TYPE::SCMDRAM);
  auto dram_addr_info = get_dram_addr(scm_row_addr);
  mem_addr_t dram_ch = std::get<0>(dram_addr_info);
  mem_addr_t dram_addr = std::get<1>(dram_addr_info);
  int dram_ch_tag = std::get<2>(dram_addr_info);
  Request meta_fill_req =
    req_creator->create_hms_request(
        scm_row_addr, dram_addr,
        transfer_bytes_per_clk, RTYPE::R_WRITE, QTYPE::ON_DEMAND_CACHE_META_FILL, clk);
  commonq.q.push_back(meta_fill_req);
  qsize[QTYPE::ON_DEMAND_CACHE_META_FILL] += 1;
  num_reqs[QTYPE::ON_DEMAND_CACHE_META_FILL] += transfer_bytes_per_clk;

  num_traffic[QTYPE::ON_DEMAND_CACHE_META_FILL] += transfer_bytes_per_clk;//DRAM wr
}

void HMSController::do_cache_line_replacement(Request& req) {
  assert(ch_type == CH_TYPE::SCMDRAM);
    // Read Remain line read is done. Now we can access and modify the TagArray entry.
    //this function is called when tag arr hit, dram cache miss
  mem_addr_t scm_row_addr = req.addr;
  mem_addr_t scm_row_num = get_scm_row_num(req.addr);

  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
  unsigned scm_rank = raw_addr.rank;
  unsigned scm_ch = raw_addr.chip;
  auto key = std::make_tuple(scm_row_num, scm_rank, scm_ch);

  auto dram_addr_info = get_dram_addr(req.addr);
  mem_addr_t dram_ch = std::get<0>(dram_addr_info);
  mem_addr_t dram_addr = std::get<1>(dram_addr_info);
  int dram_ch_tag = std::get<2>(dram_addr_info);
  assert(dram_addr == req.dram_addr);
  mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);

  assert(probe_candidate.find(key) != probe_candidate.end());

  if (probe_candidate[key].pending_invalidate.size() > 0) {
    assert(probe_candidate[key].pending_invalidate.size() == 1);
    //if there is pending invalidate at this key, stop this process and just callback pending_invalidate
    //have to stop here
    //and have to callback pending_invalidate

    Request & pending_invalidate_req = probe_candidate[key].pending_invalidate.front();
    assert(pending_invalidate_req.qtype == QTYPE::PCIE_INVALIDATE);
    assert(pending_invalidate_req.callback != nullptr);
    pending_invalidate_req.callback(pending_invalidate_req);

    probe_candidate[key].pending_invalidate.pop_back();
    assert(probe_candidate[key].pending_invalidate.size() == 0);

    probe_candidate.erase(key);//erase probe_candidate at here
    return;
  }

  bool have_to_evict = probe_candidate[key].have_to_evict;
  int target_index = probe_candidate[key].evict_target_idx;

  // If the DRAM cache line is already owned by someone.
  if (have_to_evict) {
    assert(dram_cache_line.find(dram_cache_line_num) != dram_cache_line.end());
    assert(probe_candidate[key].pending_invalidate.size() == 0);
    Request cache_probe_req = 
      req_creator->create_hms_request(
        req.addr, dram_addr, transfer_bytes_per_clk,
        RTYPE::R_READ, QTYPE::ON_DEMAND_CACHE_PROBE, clk);

    commonq.q.push_back(cache_probe_req);
    qsize[QTYPE::ON_DEMAND_CACHE_PROBE] += 1;
    num_reqs[QTYPE::ON_DEMAND_CACHE_PROBE] += transfer_bytes_per_clk;

    num_traffic[QTYPE::ON_DEMAND_CACHE_PROBE] += transfer_bytes_per_clk;//DRAM rd - probe
  } else {  // If dram cache line is not owned by someone.
    num_fill_empty_after_probe += 1;
    probe_candidate[key].metric_level = descritize_metric(probe_candidate[key].metric, max_score);
    if (policy == POLICY::IDEAL_ACCESS_CNT) {
      probe_candidate[key].metric_level = descritize_metric(probe_candidate[key].metric, max_mix);
    }

    fill_pending.push_back(make_pair(req.addr, target_index));
    assert(probe_candidate[key].pending_invalidate.size() == 0);
  }
}

void HMSController::check_old_cache_and_evict(const Request& req) {
  //Now cache probe is done.
  assert(ch_type == CH_TYPE::SCMDRAM);
  mem_addr_t scm_row_addr = req.addr;
  mem_addr_t scm_row_num = get_scm_row_num(req.addr);

  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);

  unsigned scm_rank = raw_addr.rank;
  unsigned scm_ch = raw_addr.chip;
  auto key = std::make_tuple(scm_row_num, scm_rank, scm_ch);

  auto dram_addr_info = get_dram_addr(req.addr);
  mem_addr_t dram_ch = std::get<0>(dram_addr_info);
  mem_addr_t dram_addr = std::get<1>(dram_addr_info);
  int dram_ch_tag = std::get<2>(dram_addr_info);
  assert(dram_addr == req.dram_addr);
  mem_addr_t dram_cache_line_num = get_dram_cache_line_num(dram_addr);

  assert(probe_candidate.find(key) != probe_candidate.end());
  if (probe_candidate[key].pending_invalidate.size() > 0) {
    assert(probe_candidate[key].pending_invalidate.size() == 1);
    //if there is pending invalidate at this key, stop this process and just callback pending_invalidate
    //have to stop here
    //and have to callback pending_invalidate

    Request & pending_invalidate_req = probe_candidate[key].pending_invalidate.front();
    assert(pending_invalidate_req.qtype == QTYPE::PCIE_INVALIDATE);
    assert(pending_invalidate_req.callback != nullptr);
    pending_invalidate_req.callback(pending_invalidate_req);


    probe_candidate[key].pending_invalidate.pop_back();
    assert(probe_candidate[key].pending_invalidate.size() == 0);
    assert(probe_candidate[key].pending.empty());

    probe_candidate.erase(key);//erase probe_candidate at here
    
    return;
  }

  unsigned new_metric_level = descritize_metric(probe_candidate[key].metric, max_score);
  if (policy == POLICY::IDEAL_ACCESS_CNT) {
    new_metric_level = descritize_metric(probe_candidate[key].metric, max_mix);
  }
  
  int evict_target_index = probe_candidate[key].evict_target_idx;

  assert(evict_target_index >= 0 && evict_target_index < assoc_of_dram_cache);

  if (dram_cache_line.find(dram_cache_line_num) == dram_cache_line.end()) {
    //This can happen when PCIE INVALIDATE request is comming to target dram cache line
    //don't have to evict that line
    //just push to fill_pending directly
    num_fill_empty_after_probe += 1;
    assert(probe_candidate.find(key) != probe_candidate.end());
    assert(probe_candidate[key].pending_invalidate.size() == 0);
    probe_candidate[key].metric_level = new_metric_level;

    fill_pending.push_back(make_pair(req.addr, evict_target_index));
    return;
  }

  assert(dram_cache_line.find(dram_cache_line_num) != dram_cache_line.end());
  std::map<int, new_addr_type> tags = dram_cache_line[dram_cache_line_num].tags;
  std::map<int, int> metric_levels = dram_cache_line[dram_cache_line_num].metric_levels;
  std::map<int, mem_addr_t> scm_addrs = dram_cache_line[dram_cache_line_num].scm_addrs;
  std::map<int, int> dram_ch_tags = dram_cache_line[dram_cache_line_num].dram_ch_tags;
  std::map<int, bool> valid_bits = dram_cache_line[dram_cache_line_num].valid_bits;

  if (tags.find(evict_target_index) == tags.end()) {
    //This can happen when PCIE INVALIDATE request is comming to target dram cache line
    //don't have to evict that line
    //just push to fill_pending directly
    num_fill_empty_after_probe += 1;
    assert(probe_candidate.find(key) != probe_candidate.end());
    assert(probe_candidate[key].pending_invalidate.size() == 0);
    probe_candidate[key].metric_level = new_metric_level;

    fill_pending.push_back(make_pair(req.addr, evict_target_index));
    return;
  }
  

  // The SCM row address that owns the DRAM cache line
  assert(tags.find(evict_target_index) != tags.end());
  assert(scm_addrs.find(evict_target_index) != scm_addrs.end());
  assert(dram_ch_tags.find(evict_target_index) != dram_ch_tags.end());
  assert(valid_bits.find(evict_target_index) != valid_bits.end());

  if (valid_bits[evict_target_index] == false) {
    //This can happen when PCIE INVALIDATE request is comming to target dram cache line
    //don't have to evict that line
    //just push to fill_pending directly
    num_fill_empty_after_probe += 1;
    assert(probe_candidate.find(key) != probe_candidate.end());
    assert(probe_candidate[key].pending_invalidate.size() == 0);
    probe_candidate[key].metric_level = new_metric_level;

    fill_pending.push_back(make_pair(req.addr, evict_target_index));
    return;
  }

  bool evict_cache_line = false;
  assert(metric_levels.find(evict_target_index) != metric_levels.end());
  if (policy == POLICY::ALWAYS) {
    evict_cache_line = true;
  } else {
    evict_cache_line = new_metric_level > metric_levels[evict_target_index];
  }
  
  // Probe the TagArray first
  assert(probe_candidate.find(key) != probe_candidate.end());

  if (evict_cache_line) {
    num_replace_old_after_probe += 1;
    //check dirty 
    if (probe_candidate[key].evict_target_dirty) {
      //victim DRAM cache line is dirty
      unsigned req_bytes = line_size;
      if (is_last_dram_cache_line(dram_cache_line_num)) {
        //already read 32B of dram cacheline during probing metadata 
        req_bytes -= transfer_bytes_per_clk;
      }
      
      probe_candidate[key].metric_level = new_metric_level;
      Request cache_line_read_req = 
        req_creator->create_hms_request(
          req.addr, dram_addr, req_bytes,
          RTYPE::R_READ, QTYPE::DRAM_CACHE_LINE_READ, clk, evict_target_index);
      commonq.q.push_back(cache_line_read_req);
      qsize[QTYPE::DRAM_CACHE_LINE_READ] += 1;
      num_reqs[QTYPE::DRAM_CACHE_LINE_READ] += req_bytes;
      num_traffic[QTYPE::DRAM_CACHE_LINE_READ] += req_bytes;
      return;
    }

    assert(probe_candidate.find(key) != probe_candidate.end());
    assert(probe_candidate[key].pending_invalidate.size() == 0);
    probe_candidate[key].metric_level = new_metric_level;

    fill_pending.push_back(make_pair(req.addr, evict_target_index));
  } else { // Fail to cache data due to low counter
    num_fail_after_probe += 1;
    
    if (drop_counter(probe_candidate[key].activation_counter)) {
      assert(dram_cache_line[dram_cache_line_num].metric_levels.find(evict_target_index) != dram_cache_line[dram_cache_line_num].metric_levels.end());
      (dram_cache_line[dram_cache_line_num].metric_levels)[evict_target_index] =  
        std::max((dram_cache_line[dram_cache_line_num].metric_levels)[evict_target_index] - 1, 0);
      
      fill_cache_line_metadata(req.addr);
    }
    
    assert(probe_candidate[key].pending.empty());
    assert(probe_candidate[key].pending_invalidate.size() == 0);

    probe_candidate.erase(key);
  }
}

}

