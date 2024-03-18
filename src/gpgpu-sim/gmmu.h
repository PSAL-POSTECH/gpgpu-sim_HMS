#ifndef GMMU_H_
#define GMMU_H_

#include <map>
#include <set>
#include <list>
#include <functional>
#include <unordered_set>

#include "../cuda-sim/memory.h"
#include "gpu-sim.h"

// this class simulate the gmmu unit on chip
class gmmu_t {
public:
  bool write_issued = false;
  unsigned long write_issue_count = 0;
  gmmu_t(class gpgpu_sim* gpu, const gpgpu_sim_config &config);
  unsigned long long calculate_transfer_time(size_t data_size);
  void calculate_devicesync_time(size_t data_size);
  void cycle();
  void register_tlbflush_callback(std::function<void(mem_addr_t)> cb_tlb);
  void tlb_flush(mem_addr_t page_num);
  void page_eviction_procedure();
  bool is_block_evictable(mem_addr_t bb_addr, size_t size);

  // add a new accessed page or refresh the position of the page in the LRU page list
  // being called on detecting tlb hit or when memory fetch comes back from the upward (gmmu to cu) queue
  void refresh_valid_pages(mem_addr_t page_addr, bool is_gmmu = false);
  void sort_valid_pages();

  // check whether the page to be accessed is already in pci-e write stage queue
  // being called on tlb hit or on tlb miss but no page fault
  void check_write_stage_queue(mem_addr_t page_num, bool refresh);

  void valid_pages_erase(mem_addr_t pagenum);
  void valid_pages_clear();

  void register_prefetch(mem_addr_t m_device_addr, mem_addr_t m_device_allocation_ptr, size_t m_cnt, struct CUstream_st *m_stream);
  void activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt, struct CUstream_st *m_stream);

  struct lp_tree_node* build_lp_tree(mem_addr_t addr, size_t size);
  void reset_large_page_info(struct lp_tree_node* node);
  void reset_lp_tree_node(struct lp_tree_node* node);
  struct lp_tree_node* get_lp_node(mem_addr_t addr);
  void evict_whole_tree(struct lp_tree_node *root);
  mem_addr_t update_basic_block(struct lp_tree_node *root, mem_addr_t addr, size_t size, bool prefetch);
  mem_addr_t get_basic_block(struct lp_tree_node *root, mem_addr_t addr);

  void fill_lp_tree(struct lp_tree_node* node, std::set<mem_addr_t>& scheduled_basic_blocks);
  void remove_lp_tree(struct lp_tree_node* node, std::set<mem_addr_t>& scheduled_basic_blocks);
  void traverse_and_fill_lp_tree(struct lp_tree_node* node, std::set<mem_addr_t>& scheduled_basic_blocks);
  void traverse_and_remove_lp_tree(struct lp_tree_node* node, std::set<mem_addr_t>& scheduled_basic_blocks);

  bool pcie_transfers_completed();

  void initialize_large_page(mem_addr_t start_addr, size_t size);

  unsigned long long get_ready_cycle(unsigned num_pages);
  unsigned long long get_ready_cycle_dma(unsigned size);

  float get_pcie_utilization(unsigned num_pages);

  void do_hardware_prefetch (std::map<mem_addr_t, std::list<mem_fetch*> > &page_fault_this_turn);

  void reserve_pages_insert(mem_addr_t addr, unsigned mem_access_uid);
  void reserve_pages_remove(mem_addr_t addr, unsigned mem_access_uid);
  //bool reserve_pages_check(mem_addr_t addr);
  bool is_page_reserved(mem_addr_t addr);
  void check_reserved_pages();

  //std::map<mem_addr_t, std::list<unsigned> > reserve_pages;
  std::map<mem_addr_t, std::unordered_set<unsigned>> reserve_pages;

  void update_hardware_prefetcher_oversubscribed();

  // update paging, pinning, and eviction decision based on memory access pattern under oversubscription
  void update_memory_management_policy();
  void log_kernel_info(unsigned kernel_id, unsigned long long time, bool finish);

  void reset_large_page_info();

  mem_addr_t get_eviction_base_addr(mem_addr_t page_addr);
  size_t get_eviction_granularity(mem_addr_t page_addr);

  int get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr);
  int get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr);
  void inc_bb_access_counter(mem_addr_t addr);
  void inc_bb_round_trip(struct lp_tree_node *root);
  void traverse_and_reset_access_counter(struct lp_tree_node *root);
  void reset_bb_access_counter();
  void traverse_and_reset_round_trip(struct lp_tree_node *root);
  void reset_bb_round_trip();
  void update_access_type(mem_addr_t addr, int type);

  bool should_cause_page_migration(mem_addr_t addr, bool is_write);

  // If cuda runtime API is supported, API generates large pages depending
  // on the size of cudaMallocManaged
  // However, accelsim does not support it, and thus we manage large pages here,
  // and they are initialized when the first page touch occurs
  std::set<unsigned long> initialized_large_page;
  std::set<mem_addr_t> pending_wb;

  // validate queue
  void push_pcie_gmmu_queue(mem_addr_t page_num) {
    m_pcie_gmmu_queue.push_back(page_num);
  }
  bool is_pcie_gmmu_queue_empty() {
    return m_pcie_gmmu_queue.empty();
  }
  mem_addr_t front_pcie_gmmu_queue() {
    assert(!is_pcie_gmmu_queue_empty());
    return m_pcie_gmmu_queue.front();
  }
  void pop_pcie_gmmu_queue() {
    assert(!is_pcie_gmmu_queue_empty());
    m_pcie_gmmu_queue.pop_front();
  }

  // invalidate queue
  void push_pcie_gmmu_invalidate_queue(mem_addr_t page_num) {
    m_pcie_gmmu_invalidate_queue.push_back(page_num);
  }
  bool is_pcie_gmmu_invalidate_queue_empty() {
    return m_pcie_gmmu_invalidate_queue.empty();
  }
  mem_addr_t front_pcie_gmmu_invalidate_queue() {
    assert(!is_pcie_gmmu_invalidate_queue_empty());
    return m_pcie_gmmu_invalidate_queue.front();
  }
  void pop_pcie_gmmu_invalidate_queue() {
    assert(!is_pcie_gmmu_invalidate_queue_empty());
    m_pcie_gmmu_invalidate_queue.pop_front();
  }

  unsigned long long get_host_memory_access_bytes() {
    unsigned long long current_host_memory_access_bytes = num_host_memory_access_bytes - prev_num_host_memory_access_bytes;
    prev_num_host_memory_access_bytes = num_host_memory_access_bytes;

    return current_host_memory_access_bytes;
  }
  memory_space* gmem;
  unsigned long num_pcie_read_transfer;
  unsigned long num_pcie_write_transfer;
  
  unsigned long long num_host_memory_access_bytes;
  unsigned long long prev_num_host_memory_access_bytes;
private:
  std::list<mem_addr_t> m_pcie_gmmu_queue;
  std::list<mem_addr_t> m_pcie_gmmu_invalidate_queue;
  // data structure to wrap memory fetch and page table walk delay
  struct page_table_walk_latency_t {
    mem_fetch* mf;
    unsigned long long ready_cycle;
  };

  //page table walk delay queue
  std::list<page_table_walk_latency_t> page_table_walk_queue;  

  enum class latency_type { PCIE_READ, PCIE_WRITE_BACK, INVALIDATE, PAGE_FAULT, DMA };

  // data structure to wrap a memory page and delay to transfer over PCI-E
  struct pcie_latency_t {
    mem_addr_t start_addr;
    unsigned long long size;
    std::list<mem_addr_t> page_list;
    unsigned long long ready_cycle;
    bool is_touched;
    bool is_last;

    mem_fetch* mf;
    latency_type type;
  }; 
  // staging queue to hold the PCI-E requests waiting for scheduling
  std::list<pcie_latency_t*>       pcie_read_stage_queue;
  std::list<pcie_latency_t*>       pcie_write_stage_queue;

  // read queue for fetching the page from host side 
  // the request may be global memory's read (load)/ write (store)
  pcie_latency_t *pcie_read_latency_queue;
  
  // write back queue for page eviction requests over PCI-E
  pcie_latency_t *pcie_write_latency_queue;
  // loosely represent MSHRs to hold all memory fetches 
  // corresponding to a PCI-E read requests, i.e., a common page number
  // to replay the memory fetch back upon completion
  std::map<mem_addr_t, std::list<mem_fetch*> > req_info;
  // need the gpu to do address translation, validate page
  class gpgpu_sim* m_gpu;   

  // config file
  const gpgpu_sim_config &m_config;
  const struct shader_core_config *m_shader_config; 

  // callback functions to invalidate the tlb in ldst unit
  std::list<std::function<void(mem_addr_t)> > callback_tlb_flush;

  // list of valid pages (valid = 1, accessed = 1/0, dirty = 1/0) ordered as LRU
  std::list<eviction_t *> valid_pages;

  // page eviction policy
  enum class eviction_policy { LRU, TBN, SEQUENTIAL_LOCAL, RANDOM, LFU, LRU4K }; 

  // types of hardware prefetcher
  enum class hardware_prefetcher { DISABLED, TBN, SEQUENTIAL_LOCAL, RANDOM }; 

  // types of hardware prefetcher under over-subscription
  enum class hardware_prefetcher_oversub { DISABLED, TBN, SEQUENTIAL_LOCAL, RANDOM };

  // type of DMA
  enum class dma_type { DISABLED, ADAPTIVE, ALWAYS, OVERSUB };

  // type of memory access pattern per data structure
  enum class ds_pattern { UNDECIDED, RANDOM, LINEAR, MIXED, RANDOM_REUSE, LINEAR_REUSE, MIXED_REUSE };

  // list of scheduled basic blocks by their timestamps
  std::list<std::pair<unsigned long long, mem_addr_t> > block_access_list;

  // list of launch and finish cycle of kernels keyed by id 
  std::map<unsigned, std::pair<unsigned long long, unsigned long long> > kernel_info;

  eviction_policy evict_policy;    
  hardware_prefetcher prefetcher;
  hardware_prefetcher_oversub oversub_prefetcher;

  dma_type dma_mode;

  struct prefetch_req {
    // starting address (rolled up and down for page alignment) for the prefetch
    mem_addr_t start_addr;

    // current address from the start up to which PCI-e has already processed
    mem_addr_t cur_addr;

    // starting address of the current variable allocation
    mem_addr_t allocation_addr;

    // total size (rolled up and down for page alignment) for the prefetch
    size_t size;

    // stream associated to the prefetch
    CUstream_st *m_stream;

    // memory fetches, which are created upon page fault and are depending on current prefetch,
    // aggreagted before the prefetch is actually scheduled
    std::map<mem_addr_t, std::list<mem_fetch*> > incoming_replayable_nacks;

    // memory fetches that are finished PCI-e transfer are aggregated to be replayed together
    // upon completion of the prefetch
    std::map<mem_addr_t, std::list<mem_fetch*> > outgoing_replayable_nacks; 

    // list of pages (max upto 2MB) from the current prefetch request which are being served by PCI-e
    std::list<mem_addr_t> pending_prefetch;

    // stream manager upon reaching to this entry of the queue sets it to active 
    bool active;
  };

  std::list<prefetch_req> prefetch_req_buffer;

  std::list<struct lp_tree_node*> large_page_info;
  size_t total_allocation_size;

  bool over_sub;


  unsigned long long page_fault_latency;
  void send_pcie_requests(mem_addr_t page_num, bool is_last, bool is_validate);
  unsigned long long get_pcie_page_ready_cycle();
  unsigned num_sent = 0;
  unsigned num_pages_to_be_sent = 0;
  std::unordered_set<mem_addr_t> valid_page_to_be_evicted;

  std::unordered_map<mem_addr_t, int> num_alloc;
};


struct lp_tree_node {
  mem_addr_t addr;
  size_t size;
  size_t valid_size;
  struct lp_tree_node *left;
  struct lp_tree_node *right;
  uint32_t access_counter;
  uint8_t  RW;
};

#endif
