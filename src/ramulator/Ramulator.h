#ifndef RAMULATOR_H_
#define RAMULATOR_H_

#include <string>
#include <deque>
#include <map>
#include <functional>
#include <zlib.h>

#include "MemoryFactory.h"
#include "Config.h"
#include "Request.h"

#include "../gpgpu-sim/delayqueue.h"
#include "../gpgpu-sim/dram.h"
#include "../gpgpu-sim/gpu-sim.h"


namespace ramulator{
class Request;
class MemoryBase;
}


class Ramulator : public dram_t {
public:
  Ramulator(unsigned int mem_partition_id, 
            const memory_config *m_config,
            class memory_stats_t *stats, 
            class memory_partition_unit *mp,
            class gpgpu_sim *gpu);
  ~Ramulator();
  // These functions override the functions in dram.h
  bool full(bool is_write) const;
  bool returnq_full() const; 
  class mem_fetch* return_queue_top();
  class mem_fetch* return_queue_pop();

  void print(FILE* fp) const;
  void visualizer_print(gzFile visualizer_file) {}

  void push(class mem_fetch* mf);
  void cycle();

  void set_dram_power_stats(unsigned &cmd, unsigned &activity, unsigned &nop,
                            unsigned &act, unsigned &pre, unsigned &rd,
                            unsigned &wr, unsigned &req) const {}

  void dram_log(int task) {}
  virtual unsigned que_length() const {}
  virtual unsigned int queue_limit() const {
    return m_config->gpgpu_frfcfs_dram_sched_queue_size;
  }

  double tCK;

  fifo_pipeline<mem_fetch> *r_rwq;
  fifo_pipeline<mem_fetch> *r_returnq;
  fifo_pipeline<mem_fetch> *from_gpgpusim;

private:
  std::string std_name;

  ramulator::MemoryBase* memory;


  std::map<unsigned long long, std::deque<mem_fetch*>> reads;
  std::map<unsigned long long, std::deque<mem_fetch*>> writes;

  unsigned num_cores;

  // callback functions
  std::function<void(ramulator::Request&)> read_cb_func;
  std::function<void(ramulator::Request&)> write_cb_func;
  std::function<void(ramulator::Request&)> pcie_cb_func;

  void readComplete(ramulator::Request& req);
  void writeComplete(ramulator::Request& req);
  void pcieComplete(ramulator::Request& req);

  unsigned long num_read_req = 0;
  unsigned long num_write_req = 0;
  unsigned long num_read_callback = 0;
  unsigned long num_write_callback = 0;

  ramulator::Config ramulator_configs;

  bool send(ramulator::Request& req);
  void send_pcie(ramulator::Request& req);

  //std::map<std::pair<mem_addr_t, int>, std::list<mem_addr_t>> pcie_row_mshr;
  std::list<mem_addr_t> pcie_req_list;
  std::list<mem_addr_t> pcie_invalidate_req_list;

  std::pair<mem_addr_t, int> get_row_bank_key(mem_addr_t page_addr);
  unsigned long clk = 0;
  unsigned long num_invalidate_done = 0;
  unsigned long num_evict_processing = 0;
};


#endif
