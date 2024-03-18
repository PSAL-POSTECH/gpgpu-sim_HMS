#ifndef __REQUEST_H
#define __REQUEST_H

#include <vector>
#include <functional>
#include "../gpgpu-sim/mem_fetch.h"
#include "../cuda-sim/memory.h"
#include "qtype.h"

using namespace std;
using namespace request;

namespace ramulator {
class Request {
public:

  bool is_first_command;

  unsigned long addr;
  unsigned long dram_addr; //jeongmin - this dram_addr is without channel bits
  vector<int> addr_vec;

  int original_num_bytes = 0;
  int num_bytes = 0;

  int client_rank = -1;//rank of client
  int client_ch = -1;//ch id of client (who sends request to supplier)
  int supplier_rank = -1;//rank of supplier
  int supplier_ch = -1;//ch id of supplier (who receives request from client)

  bool done = false;

  bool is_pcie = false;
  bool cache_hit = false;
  int is_tag_arr_hit = 0; // 0 : initial, 1: hit, -1 : miss at first glance

  // specify which core this request sent from, for virtual address translation
  int coreid = -1;

  int evict_target_index = -1;//jeongmin

  std::deque<Request>* target_pending;

  Type type;
  Type original_type = Type::MAX;
  QueueType qtype = QueueType::MAX;

  long arrive = 0;
  long depart = -1;

  unsigned long long req_receive_time = 0; //when demand request comes into ctrl
  unsigned long long req_scheduled_time = 0; //when demand request is scheduled and issued

  long tag_cache_access_ready_cycle = 0;

  function<void(Request&)> callback; // call back with more info

  mem_fetch* mf = nullptr;

  Request(unsigned long addr, Type type, QueueType qtype, int coreid = 0)
      : addr(addr), coreid(coreid), type(type), qtype(qtype),
        callback([](Request& req){}) {}

  Request(unsigned long addr, Type type, function<void(Request&)> callback, 
          int coreid = 0)
      : addr(addr), coreid(coreid), type(type), callback(callback) {}

  Request(unsigned long addr, Type type, function<void(Request&)> callback, 
          int coreid, mem_fetch *mf)
      : addr(addr), coreid(coreid), type(type), callback(callback), mf(mf) {
    if (type == Type::R_READ) {
      assert(mf->get_data_size() == SECTOR_SIZE);
      num_bytes = mf->get_data_size();
      original_num_bytes = mf->get_data_size();
    } else if (type == Type::R_WRITE) {
      auto acc_type = mf->get_access_type();
      assert(acc_type == L2_WRBK_ACC || acc_type == L1_WRBK_ACC);
      // The data size of the write request varies from 32 to 128 B.
      // Assume the single memory commands handle 32 B,
      // the numer of memory commands needs to be issued is 'num_bytes' / 32
      //
      // Note that the address (32 B granuarlity) of the write requests 
      // is bounds to the same row buffer.
      // As a result, instead of calculating the column address of the memory,
      // we just issue serveral commands to the memory until 'num_bytes' become 0.
      num_bytes = mf->get_data_size();
      original_num_bytes = mf->get_data_size();
    }
  }

  Request(vector<int>& addr_vec, Type type, function<void(Request&)> callback, 
          int coreid = 0)
      : addr_vec(addr_vec), coreid(coreid), type(type), callback(callback) {}

  // used for HMS
  Request(unsigned long scm_addr, unsigned long dram_addr, Type type, QueueType qtype, 
          int num_bytes, mem_fetch* mf, function<void(Request&)> callback)
      : addr(scm_addr), dram_addr(dram_addr), type(type), qtype(qtype), 
        num_bytes(num_bytes), mf(mf), callback(callback) { coreid = 0; }

  //used for HMS
  Request(unsigned long scm_addr, unsigned long dram_addr, Type type, QueueType qtype, 
        int num_bytes, int original_num_bytes, int client_rank, int client_ch, int supplier_rank, int supplier_ch,
        mem_fetch* mf, long arrive)
      : addr(scm_addr), dram_addr(dram_addr), type(type), qtype(qtype),
        num_bytes(num_bytes), original_num_bytes(original_num_bytes), client_rank(client_rank), client_ch(client_ch),
        supplier_rank(supplier_rank), supplier_ch(supplier_ch), mf(mf), arrive(arrive), callback(nullptr) {coreid = 0;}

  Request() {}

  bool operator==(const Request &other) const {
    if ((other.addr == addr) &&  
    (other.num_bytes == num_bytes) && (other.original_num_bytes == original_num_bytes) &&
    (other.type == type) ) {
      return true;
    }
    else
      return false;
  }
};

} /*namespace ramulator*/

#endif /*__REQUEST_H*/

