#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <cassert>
#include <cstdio>
#include <deque>
#include <fstream>
#include <list>
#include <string>
#include <vector>

#include "Config.h"
#include "DRAM.h"
#include "Refresh.h"
#include "Request.h"
#include "Scheduler.h"
#include "Statistics.h"

#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/l2cache.h"


class Ramulator;

using namespace std;

namespace ramulator {
extern bool warmup_complete;

template <typename T>
class Controller {

public:
  std::set<std::pair<int, int>> is_written_row;
  /* Member Variables */
  long clk = 0;
  DRAM<T>* channel;

  Scheduler<T>* scheduler;  // determines the highest priority request whose commands will be issued
  RowPolicy<T>* rowpolicy;  // determines the row-policy (e.g., closed-row vs. open-row)
  RowTable<T>* rowtable;  // tracks metadata about rows (e.g., which are open and for how long)
  Refresh<T>* refresh;

  const memory_config* m_config;

  struct Queue {
      list<Request> q;
      unsigned int max = 32;
      unsigned int size() {return q.size();}
  };

  // common queue is used when the user does not want to use seperate r/w queue
  Queue commonq;

  Queue readq;  // queue for read requests
  Queue writeq;  // queue for write requests
  Queue actq; // read and write requests for which activate was issued are moved to 
                 // actq, which has higher priority than readq and writeq.
                 // This is an optimization
                 // for avoiding useless activations (i.e., PRECHARGE
                 // after ACTIVATE w/o READ of WRITE command)
  Queue otherq;  // queue for all "other" requests (e.g., refresh)

  Queue pcie_queue;
  deque<Request> pcie_pending;

  deque<Request> pending;  // read requests that are about to receive data from DRAM
  bool write_mode = false;  // whether write requests should be prioritized over reads
  float wr_high_watermark = 0.8f; // threshold for switching to write mode
  float wr_low_watermark = 0.2f; // threshold for switching back to read mode
  //long refreshed = 0;  // last time refresh requests were generated

  /* Command trace for DRAMPower 3.1 */
  string cmd_trace_prefix = "cmd-trace-";
  vector<ofstream> cmd_trace_files;
  bool record_cmd_trace = false;
  /* Commands to stdout */
  bool print_cmd_trace = false;

  unsigned ch_id;
  
  unsigned long long num_act_d = 0;
  unsigned long long num_pre_d = 0;
  unsigned long long num_pre_w_d = 0;
  unsigned long long num_prea_d = 0;
  unsigned long long num_prea_w_d = 0;
  unsigned long long num_ref_d = 0;
  unsigned long long num_rd_d = 0;
  unsigned long long num_wr_d = 0;

  unsigned long long num_act_s = 0;
  unsigned long long num_pre_w_s = 0;
  unsigned long long num_rd_s = 0;
  unsigned long long num_wr_s = 0;

  std::ofstream cachePrinter;
  std::ofstream trafficPrinter;

  std::map<request::QueueType, unsigned long> num_traffic;

  int global_penalty_level;

  int max_num_on_the_fly_buffer = 0;
  int num_on_the_fly_buffer = 0;


  class memory_partition_unit* m_mp;
  class gpgpu_sim* m_gpu;

  int transfer_bytes_per_clk;

  bool print_detail;
  bool stat_print;
  
  unsigned long num_cache_hit;
  unsigned long num_cache_miss;
  unsigned long num_cache_hit_rd;
  unsigned long num_cache_miss_rd;
  unsigned long num_cache_hit_wr;
  unsigned long num_cache_miss_wr;
  unsigned long access_act_cnt_table = 0;
  unsigned long update_act_cnt_table = 0;
  unsigned long right_shift_act_cnt_table = 0;
  
  /* Constructor */
  Controller(const Config& configs, DRAM<T>* channel, int channel_id,
             const memory_config* m_config,
             class memory_partition_unit* mp,
             class gpgpu_sim* gpu) 
      : channel(channel),
        scheduler(new Scheduler<T>(this)),
        rowpolicy(new RowPolicy<T>(this)),
        rowtable(new RowTable<T>(this)),
        refresh(new Refresh<T>(this)),
        cmd_trace_files(channel->children.size()),
        m_config(m_config),
        m_gpu(gpu), m_mp(mp) {
    transfer_bytes_per_clk = 
      (channel->spec->prefetch_size * channel->spec->channel_width) / 8;
    ch_id = channel_id;
    record_cmd_trace = configs.record_cmd_trace();
    print_cmd_trace = configs.print_cmd_trace();

    cachePrinter.open("output/cache_ch_"+to_string(channel_id)+".txt");
    assert(cachePrinter.is_open());
    trafficPrinter.open("output/traffic_ch_"+to_string(channel_id)+".txt");
    assert(trafficPrinter.is_open());

    if (m_config->enable_ctrl_printer)
      print_detail = true;
    else
      print_detail = false;
    if (m_config->enable_stat_printer)
      stat_print = true;
    else
      stat_print = false;

    // set Queue size
    if (m_config->seperate_write_queue_enabled) {
      writeq.max = m_config->gpgpu_frfcfs_dram_write_queue_size;
      readq.max = m_config->gpgpu_frfcfs_dram_sched_queue_size;
    } else {
      commonq.max = m_config->gpgpu_frfcfs_dram_sched_queue_size;
    }
  }

  ~Controller(){
    delete scheduler;
    delete rowpolicy;
    delete rowtable;
    delete channel;
    delete refresh;
    assert(cachePrinter.is_open());
    cachePrinter.close();
    assert(trafficPrinter.is_open());
    trafficPrinter.close();
  }

  virtual bool full(request::Type type) {
    Queue q = get_queue(type);
    return (q.size() >= q.max);
  }

  virtual void finish(long read_req, long dram_cycles, unsigned kernel_id) {
    // call finish function of each channel
    channel->finish(dram_cycles);

    double total_access = num_cache_hit + num_cache_miss;
    double cache_miss_rate = (total_access != 0)? (num_cache_miss/total_access) : 0;
    double total_access_rd = num_cache_hit_rd + num_cache_miss_rd;
    double cache_miss_rate_rd = (total_access_rd != 0)? (num_cache_miss_rd/total_access_rd) : 0;
    double total_access_wr = num_cache_hit_wr + num_cache_miss_wr;
    double cache_miss_rate_wr = (total_access_wr != 0)? (num_cache_miss_wr/total_access_wr) : 0;
    cachePrinter << "num_cache_hit "<<num_cache_hit<<" "
                 << "num_cache_miss "<<num_cache_miss<<" "
                 << "cache_miss_rate "<<cache_miss_rate<<" "
                 << "num_cache_hit_rd "<<num_cache_hit_rd<<" "
                 << "num_cache_miss_rd "<<num_cache_miss_rd<<" "
                 << "cache_miss_rate_rd "<<cache_miss_rate_rd<<" "
                 << "num_cache_hit_wr "<<num_cache_hit_wr<<" "
                 << "num_cache_miss_wr "<<num_cache_miss_wr<<" "
                 << "cache_miss_rate_wr "<<cache_miss_rate_wr<<std::endl;
    cachePrinter << "kernel " << kernel_id << " end" << std::endl;

    trafficPrinter << "kernel " << kernel_id << " "
                   << "cycle "<<dram_cycles<<" "
                   << "PCIE_VALIDATE "<<num_traffic[request::QueueType::PCIE_VALIDATE]<<" "
                   << "TAG_ARR_READ " <<num_traffic[request::QueueType::TAG_ARR_READ]<<" "
                   << "TAG_ARR_WRITE "<<num_traffic[request::QueueType::TAG_ARR_WRITE]<<" "
                   << "CACHE_MISS_RD " <<num_traffic[request::QueueType::CACHE_MISS_RD] <<" "
                   << "CACHE_MISS_WR " <<num_traffic[request::QueueType::CACHE_MISS_WR] <<" "
                   << "CACHE_HIT_RD " <<num_traffic[request::QueueType::CACHE_HIT_RD] <<" "
                   << "CACHE_HIT_WR " <<num_traffic[request::QueueType::CACHE_HIT_WR] <<" "
                   << "DRAM_CACHE_LINE_READ " <<num_traffic[request::QueueType::DRAM_CACHE_LINE_READ]<<" "
                   << "REMAIN_LINE_READ " <<num_traffic[request::QueueType::REMAIN_LINE_READ]<<" "
                   << "CACHE_EVICT " <<num_traffic[request::QueueType::CACHE_EVICT]<<" "
                   << "ON_DEMAND_CACHE_FILL " <<num_traffic[request::QueueType::ON_DEMAND_CACHE_FILL]<<" "
                   << "ON_DEMAND_CACHE_META_FILL "<<num_traffic[request::QueueType::ON_DEMAND_CACHE_META_FILL]<<" "
                   << "ON_DEMAND_CACHE_PROBE "<<num_traffic[request::QueueType::ON_DEMAND_CACHE_PROBE]<<std::endl;
  }

  /* Member Functions */
  Queue& get_queue(request::Type type) {//Request::Type
    if (m_config->seperate_write_queue_enabled) {
      switch (int(type)) {
        case int(request::Type::R_READ): return readq;
        case int(request::Type::R_WRITE): return writeq;
        default: return otherq;
      }
    } else {
      switch (int(type)) {
        case int(request::Type::R_READ):
        case int(request::Type::R_WRITE):
          return commonq;
        default:
          return otherq;
      }
    }
  }

  virtual void enqueue_pcie(Request& req) {
    req.arrive = clk;
    pcie_queue.q.push_back(req);
    if (req.qtype == request::QueueType::PCIE_VALIDATE) {
      num_traffic[request::QueueType::PCIE_VALIDATE] += req.num_bytes;
    }
  }

  virtual bool enqueue(Request& req) {
    Queue& queue = get_queue(req.type);
    if (queue.max == queue.size())
        return false;

    req.arrive = clk;
    queue.q.push_back(req);

    if (req.type == request::Type::R_READ) {
      num_traffic[request::QueueType::CACHE_HIT_RD] += req.num_bytes;
    } else if (req.type == request::Type::R_WRITE) {
      num_traffic[request::QueueType::CACHE_HIT_WR] += req.num_bytes;
    }

    return true;
  }
  
  virtual void tick() {
    clk++;

    /*** Serve pcie request ***/
    if (pcie_pending.size()) {
      Request& req = pcie_pending[0];
      if (req.depart <= clk) {
        req.callback(req);
        pcie_pending.pop_front();
      }
    }
    /*** 1. Serve completed reads ***/
    if (pending.size()) {
      Request& req = pending[0];
      if (req.depart <= clk) {
        req.callback(req);
        pending.pop_front();
      }
    }

    /*** 2. Refresh scheduler ***/
    if (channel->spec->refresh)
      refresh->tick_ref();

    if (m_config->seperate_write_queue_enabled) {
      /*** 3. Should we schedule writes? ***/
      if (!write_mode) {
        // yes -- write queue is almost full or read queue is empty
        if (writeq.size() > int(wr_high_watermark * writeq.max) || readq.size() == 0)
            write_mode = true;
      }
      else {
        // no -- write queue is almost empty and read queue is not empty
        if (writeq.size() < int(wr_low_watermark * writeq.max) && readq.size() != 0)
            write_mode = false;
      }
    }

    /*** 4. Find the best command to schedule, if any ***/

    // First check the actq (which has higher priority) to see if there
    // are requests available to service in this cycle
    Queue* queue = &actq;
    typename T::Command cmd;
    auto req = scheduler->get_head(queue->q);

    bool is_valid_req = (req != queue->q.end());

    if (is_valid_req) {
      cmd = get_first_cmd(req);
      is_valid_req = is_ready(cmd, req->addr_vec);
    }

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

    if (!is_valid_req) {
      if (m_config->seperate_write_queue_enabled) {
        queue = !write_mode ? &readq : &writeq;
      } else {
        queue = &commonq;
      }

      if (otherq.size())
          queue = &otherq;  // "other" requests are rare, so we give them precedence over reads/writes

      req = scheduler->get_head(queue->q);

      is_valid_req = (req != queue->q.end());

      if(is_valid_req){
        cmd = get_first_cmd(req);
        is_valid_req = is_ready(cmd, req->addr_vec);
      }
    }

    if (!is_valid_req) {
      // we couldn't find a command to schedule -- let's try to be speculative
      auto cmd = T::Command::PRE;
      vector<int> victim = rowpolicy->get_victim(cmd);
      if (!victim.empty()){
          issue_cmd(cmd, victim);
      }
      return;  // nothing more to be done this cycle
    }

    // issue command on behalf of request
    issue_cmd(cmd, get_addr_vec(cmd, req));

    // check whether this is the last command (which finishes the request)
    if (cmd != channel->spec->translate[int(req->type)]) {
      if (channel->spec->is_opening(cmd)) {
        // promote the request that caused issuing activation to actq
        actq.q.push_back(*req);
        queue->q.erase(req);
      }
      return;
    }

    // Because the R / W command is issued, decrease num bytes.
    if (req->type == request::Type::R_READ ||
        req->type == request::Type::R_WRITE) {
      req->num_bytes -= transfer_bytes_per_clk;
      assert(req->num_bytes >= 0);
      if (req->num_bytes == 0) {
        req->done = true;
        if (req->type == request::Type::R_READ) {
          req->depart = clk + channel->spec->read_latency;
        } else if (req->type == request::Type::R_WRITE) {
          req->depart = clk + channel->spec->write_latency;
        }
      }
    } else {
      assert(req->num_bytes == 0);
      req->done = true;
    }

    // set a future completion time for read requests
    if (req->qtype == request::QueueType::PCIE_VALIDATE) {
      assert(req->type == request::Type::R_WRITE);
      if (req->done) {
        req->callback(*req);
      }
    } else if (req->qtype == request::QueueType::PCIE_INVALIDATE) {
      assert(req->type == request::Type::R_READ);
      if (req->done) {
        pcie_pending.push_back(*req);
      }
    } else if (req->type == request::Type::R_READ) {
      if (req->done) {
        pending.push_back(*req);
      }
    } else if (req->type == request::Type::R_WRITE) {
      if (req->done) {
        req->callback(*req);
      }
    }

    // remove request from queue
    if (req->done)
      queue->q.erase(req);
  }

  bool is_ready(list<Request>::iterator req) {
    typename T::Command cmd = get_first_cmd(req);
    return channel->check(cmd, req->addr_vec.data(), clk);
  }

  bool is_ready(typename T::Command cmd, const vector<int>& addr_vec) {
    return channel->check(cmd, addr_vec.data(), clk);
  }

  bool is_row_hit(list<Request>::iterator req) {
    // cmd must be decided by the request type, not the first cmd
    typename T::Command cmd = channel->spec->translate[int(req->type)];
    return channel->check_row_hit(cmd, req->addr_vec.data());
  }

  bool is_row_hit(typename T::Command cmd, const vector<int>& addr_vec) {
    return channel->check_row_hit(cmd, addr_vec.data());
  }

  bool is_row_open(list<Request>::iterator req) {
    // cmd must be decided by the request type, not the first cmd
    typename T::Command cmd = channel->spec->translate[int(req->type)];
    return channel->check_row_open(cmd, req->addr_vec.data());
  }

  bool is_row_open(typename T::Command cmd, const vector<int>& addr_vec) {
    return channel->check_row_open(cmd, addr_vec.data());
  }

  // For telling whether this channel is busying in processing read or write
  bool is_active() {
    return (channel->cur_serving_requests > 0);
  }

  // For telling whether this channel is under refresh
  bool is_refresh() {
    return clk <= channel->end_of_refreshing;
  }

  void set_high_writeq_watermark(const float watermark) {
    wr_high_watermark = watermark; 
  }

  void set_low_writeq_watermark(const float watermark) {
    wr_low_watermark = watermark;
  }

protected:
  typename T::Command get_first_cmd(list<Request>::iterator req) {
    typename T::Command cmd = channel->spec->translate[int(req->type)];
    return channel->decode(cmd, req->addr_vec.data());
  }

  // upgrade to an autoprecharge command
  void cmd_issue_autoprecharge(typename T::Command& cmd,
                               const vector<int>& addr_vec) {
    // currently, autoprecharge is only used with closed row policy
    if(channel->spec->is_accessing(cmd) && rowpolicy->type == RowPolicy<T>::Type::ClosedAP) {
      // check if it is the last request to the opened row
      Queue* queue = write_mode ? &writeq : &readq;

      auto begin = addr_vec.begin();
      vector<int> rowgroup(begin, begin + int(T::Level::Row) + 1);

      int num_row_hits = 0;

      for (auto itr = queue->q.begin(); itr != queue->q.end(); ++itr) {
        if (is_row_hit(itr)) { 
          auto begin2 = itr->addr_vec.begin();
          vector<int> rowgroup2(begin2, begin2 + int(T::Level::Row) + 1);
          if (rowgroup == rowgroup2)
              num_row_hits++;
        }
      }

      if (num_row_hits == 0) {
        Queue* queue = &actq;
        for (auto itr = queue->q.begin(); itr != queue->q.end(); ++itr) {
          if (is_row_hit(itr)) {
            auto begin2 = itr->addr_vec.begin();
            vector<int> rowgroup2(begin2, begin2 + int(T::Level::Row) + 1);
            if (rowgroup == rowgroup2)
                num_row_hits++;
          }
        }
      }

      assert(num_row_hits > 0); // The current request should be a hit, 
                                // so there should be at least one request 
                                // that hits in the current open row
      if (num_row_hits == 1) {
        if (cmd == T::Command::RD)
            cmd = T::Command::RDA;
        else if (cmd == T::Command::WR)
            cmd = T::Command::WRA;
        else
            assert(false && "Unimplemented command type.");
      }
    }
  }

  void issue_cmd(typename T::Command cmd, const vector<int>& addr_vec) {
    cmd_issue_autoprecharge(cmd, addr_vec);
    assert(is_ready(cmd, addr_vec));
    channel->update(cmd, addr_vec.data(), clk);

    rowtable->update(cmd, addr_vec, clk);
    if (record_cmd_trace) {
      std::string cmd_name = channel->spec->command_name[int(cmd)];
      int bank = addr_vec[int(HMS::Level::Bank)];
      int bg = addr_vec[int(HMS::Level::Bank) - 1];
      int rank = addr_vec[int(HMS::Level::Rank)];
      if (cmd == T::Command::WR) {
        auto key = std::make_pair(rank, bg * 4 + bank);
        is_written_row.insert(key);
      } else if (cmd == T::Command::PRE || cmd == T::Command::PREA) {
        auto key = std::make_pair(rank, bg * 4 + bank);
        if (is_written_row.find(key) != is_written_row.end()) {
          cmd_name += "_W";
          is_written_row.erase(key);
        }
      }

      if (rank == int(HMS::DRAM_RANK_ID)) {
        //DRAM 
        if (cmd_name == "ACT") {
          num_act_d ++;
        } else if (cmd_name == "PRE") {
          num_pre_d ++;
        } else if (cmd_name == "PRE_W") {
          num_pre_w_d ++;
        } else if (cmd_name == "PREA") {
          num_prea_d ++;
        } else if (cmd_name == "PREA_W") {
          num_prea_w_d ++;
        } else if (cmd_name == "REF") {
          num_ref_d ++;
        } else if (cmd_name == "RD") {
          num_rd_d ++;
        } else if (cmd_name == "WR") {
          num_wr_d ++;
        }
      } else if (rank == int(HMS::SCM_RANK_ID)) {
        //SCM
        if (cmd_name == "ACT") {
          num_act_s ++;
        } else if (cmd_name == "PRE_W") {
          num_pre_w_s ++;
        }  else if (cmd_name == "RD") {
          num_rd_s ++;
        } else if (cmd_name == "WR") {
          num_wr_s ++;
        }
      }
    }
  }
  vector<int> get_addr_vec(typename T::Command cmd, list<Request>::iterator req) {
    return req->addr_vec;
  }
};

//template <>
//vector<int> Controller<SALP>::get_addr_vec(
    //SALP::Command cmd, list<Request>::iterator req);

//template <>
//bool Controller<SALP>::is_ready(list<Request>::iterator req);

//template <>
//void Controller<ALDRAM>::update_temp(ALDRAM::Temp current_temperature);

//template <>
//void Controller<TLDRAM>::tick();

//template <>
//void Controller<TLDRAM>::cmd_issue_autoprecharge(typename TLDRAM::Command& cmd,
                                                    //const vector<int>& addr_vec);

} /*namespace ramulator*/

#endif /*__CONTROLLER_H*/
