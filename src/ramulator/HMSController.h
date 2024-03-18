#ifndef HMS_CONTROLLER_H
#define HMS_CONTROLLER_H

#include <bitset>
#include <cassert>
#include <cstdio>
#include <deque>
#include <fstream>
#include <list>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

#include "Controller.h"

#include "HBM.h"
#include "PCM.h"
#include "HMS.h"

#include "HMSRequest.h"

#include "../gpgpu-sim/addrdec.h"
#include "../gpgpu-sim/gpu-cache.h"

using namespace std;

namespace ramulator {

// Inherit Controller<T> to make our HMS compatible with Ramulator
class HMSController : public Controller<HMS> {
 
public: // sharable variables
  enum class MODE : int { CACHE, FLAT, MAX };

  /* Constructor */
  HMSController(const Config& configs, 
                DRAM<HMS>* channel,  
                int channel_id,
                const memory_config* m_config,
                class memory_partition_unit* mp,
                class gpgpu_sim* gpu);
  ~HMSController() {
    delete tag_arr;
  }

  // calculate bank id from bank group and bank
  int get_global_bank_id(Request req);
  bool enqueue(Request& req);
  void enqueue_pcie(Request& req);
  bool full(request::Type type);
  void check_cache_hit(Request& req);

  void tick();

private:
  const memory_config* m_config;
  enum class POLICY { TWO_LEVEL_SCORE, ALWAYS, IDEAL_ACCESS_CNT, MAX };
  enum class DRAM_CACHE_STAT { CHECKING, DONE, MAX };
  enum class CH_TYPE { SCMDRAM, SCM, DRAM, MAX };

  CH_TYPE ch_type = CH_TYPE::MAX;

  POLICY policy = POLICY::MAX;

  HMSRequestCreator* req_creator;

  typedef request::Type RTYPE;
  typedef request::QueueType QTYPE;

  unsigned line_size;
  unsigned num_cache_lines;
  unsigned line_size_log_bits;
  unsigned num_scm_channels;
  unsigned num_scm_channel_log_bits;
  unsigned num_dram_channels;
  unsigned num_dram_cache_offset_bits;//number of offset bits per dram cache row buffer
  unsigned assoc_of_dram_cache;//associativity of dram cache

  unsigned enable_tag_cache;

  Queue tag_arr_access_queue;//requests are waiting to access TagArray

  Queue tag_arr_queue;
  deque<Request> pcie_pending;
  deque<Request> core_pending;

  deque<Request> tag_arr_pending;
  deque<Request> cache_probe_pending;
  deque<Request> line_read_pending;
  deque<Request> read_dram_cache_line_pending;
  deque<Request> pcie_invalidate_cache_hit_pending;

  list<tag_arr_entry_evict_entry> write_tag_arr_entry_queue;

  void access_tag_arr();
  void write_tag_arr();

  deque<Request> miss_pending;
  // To evict the cache line, we need to probe the metadata and data
  deque<Request> probe_pending;

  float global_score;

  static unsigned int LOGB2_32(unsigned int v) {
    unsigned int shift;
    unsigned int r;

    r = 0;

    shift = ((v & 0xFFFF0000) != 0) << 4;
    v >>= shift;
    r |= shift;
    shift = ((v & 0xFF00) != 0) << 3;
    v >>= shift;
    r |= shift;
    shift = ((v & 0xF0) != 0) << 2;
    v >>= shift;
    r |= shift;
    shift = ((v & 0xC) != 0) << 1;
    v >>= shift;
    r |= shift;
    shift = ((v & 0x2) != 0) << 0;
    v >>= shift;
    r |= shift;

    return r;
  }

  tag_arr_tag_array* tag_arr;

  std::tuple<mem_addr_t, mem_addr_t, int> get_dram_addr(mem_addr_t scm_mem_addr);
  new_addr_type get_dram_tag(mem_addr_t scm_mem_addr);
  unsigned long get_dram_row_num(mem_addr_t dram_addr);
  unsigned long get_dram_cache_line_num(mem_addr_t dram_addr);
  unsigned long get_dram_cache_line_from_dram_row_num(unsigned long dram_row_num, int offset);
  unsigned long get_dram_row_start_addr(mem_addr_t dram_row_num);
  bool is_last_col(mem_addr_t dram_addr);
  bool is_last_dram_cache_line(mem_addr_t dram_cache_line_num);
  void update_global_score(float score) {
    if (global_score == 0.0f)
      global_score = score;
    else
      global_score = global_score * 0.99 + score * 0.01;
  }

  mem_addr_t calc_page_num(mem_addr_t addr) {
    return addr >> 12;
  }

  void update_act_cnt(mem_addr_t page_num) {    
    update_act_cnt_table += 1;
    /*2MB ideal without saturation*/
    mem_addr_t _2MB_addr = page_num >> 9; //2MB gran
    if (act_cnt_table.find(_2MB_addr) == act_cnt_table.end()) {
      act_cnt_table[_2MB_addr] = 1;
    } else {
      act_cnt_table[_2MB_addr] += 1;
    }
  }

  unsigned long get_act_cnt(mem_addr_t page_num) {
    access_act_cnt_table += 1;
    mem_addr_t _2MB_addr = page_num >> 9; //2MB gran
    if (act_cnt_table.find(_2MB_addr) == act_cnt_table.end()) {
      //no act_cnt_table entry, return 1;
      act_cnt_table[_2MB_addr] = 1;
      return 1;
    } else {
      return act_cnt_table[_2MB_addr];
    } 

  }

  unsigned get_empty_cache_line() {
    unsigned fill_line_num = 0;
    auto dram_line = dram_cache_line.begin();
    for(; dram_line != dram_cache_line.end(); ++dram_line) {
      auto tags = dram_line->second.tags;
      for (int i = 0; i < assoc_of_dram_cache; i++) { 
        if (tags.find(i) != tags.end()) {
          if (dram_line->second.valid_bits[i]) {
            fill_line_num += 1;
          }
        }
      }
    }
    assert(fill_line_num <= (num_cache_lines * assoc_of_dram_cache));
    return ((num_cache_lines * assoc_of_dram_cache) - fill_line_num);
  }
 
  unsigned long num_tag_arr_read;
  unsigned long num_tag_arr_write;
  unsigned long num_tag_arr_miss;
  unsigned long num_tag_arr_hit;

  unsigned long num_line_read;
  unsigned long num_cache_probe;

  unsigned long num_evict;
  unsigned long num_fill;
  unsigned long num_pcie_write;

  unsigned long long total_row_accesses;
  unsigned long long total_num_activates;

  cache_config tag_arr_config;

  unsigned long max_counter;
  float max_score;
  float max_mix;
  unsigned long min_counter;
  float min_score;
  float min_mix;

  const int LEVEL;
  bool act_cnt_2MB;//manage activation counter at controller with 2MiB gran.
  int cur_threshold = 0;

  std::map<QTYPE, unsigned long> num_reqs;
  std::map<QTYPE, unsigned long> qsize;

  unsigned long num_replace_old_after_probe;
  unsigned long num_fail_after_probe;
  unsigned long num_fill_empty_after_probe;

  unsigned long available_queue_size;
  unsigned long available_tag_access_queue_size;

  typedef struct merger {
    int num_cols;
    int count;
    bool dirty;
    bool trigger_lock;
    std::vector<Request> pending_invalidate;
    bool have_to_evict;
    int evict_target_idx;
    bool evict_target_dirty;
  } merger;

  typedef struct tag_arr_merger {
    std::vector<Request> pending;
    DRAM_CACHE_STAT dram_cache_stat;
    std::vector<int> dirty_idxs;
  } tag_arr_merger;

  typedef struct cache_line_score {
    float score;
    int score_level;
    unsigned long activation_counter;
    int num_cols;
    bool dirty;
  } cache_line_score;

  typedef struct cache_line_metadta {
    std::map<int, new_addr_type> tags;
    std::map<int, mem_addr_t> scm_addrs;
    std::map<int, int> dram_ch_tags;//tag for dram channel -> to identify original scm ch 
    std::map<int, int> metric_levels;
    std::map<int, bool> dirty_bits;
    std::map<int, bool> valid_bits;
    int evict_target_index;
  } cache_line_metadata;

  typedef struct cache_line_probe_register {
    int num_cols;
    float metric;
    int metric_level;
    unsigned long activation_counter;
    bool dirty;
    bool line_read_done;
    bool trigger_lock;
    std::vector<Request> pending;
    std::vector<Request> pending_invalidate;//pending PCIE_INVALIDATE request
    bool have_to_evict;
    int evict_target_idx;
    bool evict_target_dirty;
  } cache_line_probe_register;

  typedef struct first_col_merger {
    bool line_read_done;
    std::vector<Request> pending;
  } first_col_merger;


  std::map<std::tuple<unsigned long, unsigned, unsigned>, merger> miss_merge_reg;
  std::map<std::tuple<unsigned long, unsigned, unsigned>, merger> hit_merge_reg;
  std::map<std::tuple<unsigned long, unsigned, unsigned>, cache_line_score> score_reg;
  std::unordered_map<unsigned long, cache_line_metadata> dram_cache_line;
  std::map<std::tuple<unsigned long, unsigned, unsigned>, cache_line_probe_register> probe_candidate;
  std::unordered_map<unsigned long, bool> read_dram_tags_reg;
  std::map<mem_addr_t, unsigned long> act_cnt_table;
  

  unsigned long get_scm_row_num(unsigned long addr);
  unsigned long get_scm_row_addr(unsigned long row_num);

  bool do_tag_arr_miss(mem_addr_t scm_row_addr, unsigned long dram_addr);

  bool drop_counter(unsigned long counter);
  int descritize_metric(float metric, float metric_max);
  float calculate_score(unsigned num_cols, bool is_dirty);

  void post_process_cache_hit(const Request& req);
  void post_process_cache_miss(const Request& req);
  void post_process_remain_line_read(const Request& req);

  void fill_cache_line(mem_addr_t scm_addr, int new_metric_level, int index);
  void fill_cache_line_pcie(Request& req, int new_metric_level, int index);
  void fill_cache_line_metadata(mem_addr_t scm_row_addr);
  void do_cache_line_replacement(Request& req);
  void check_old_cache_and_evict(const Request& req);

  std::unordered_map<QTYPE, function<void(const Request&)>> post_processor;

  std::ofstream tag_arr_printer;

  unsigned long num_score_reg_made;

  std::deque<std::pair<mem_addr_t, int>> fill_pending;
  std::deque<std::pair<mem_addr_t, int>> evict_pending;
  unsigned long num_tag_arr_write_req = 0;
  unsigned long num_tag_arr_read_req = 0;
  int done = 0;

  Queue pcie_dram_queue;
};

} /*namespace ramulator*/

#endif /*__CONTROLLER_H*/
