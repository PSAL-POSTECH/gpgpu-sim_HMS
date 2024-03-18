#include "gmmu.h"
#include "l2cache.h"
#include "../abstract_hardware_model.h"

#include <bitset>

gmmu_t::gmmu_t(class gpgpu_sim* gpu, const gpgpu_sim_config &config)
  : m_gpu(gpu), m_config(config), gmem(gpu->get_global_memory()) {
  m_shader_config = &m_config.m_shader_config;

  num_pcie_read_transfer = 0;
  num_pcie_write_transfer = 0;
  num_host_memory_access_bytes = 0;
  prev_num_host_memory_access_bytes = 0;

  if ( m_config.enable_dma == 0 ) {
     dma_mode = dma_type::DISABLED;
  } else if ( m_config.enable_dma == 1 ) {
     dma_mode = dma_type::ADAPTIVE;
  } else if ( m_config.enable_dma == 2 ) {
     dma_mode = dma_type::ALWAYS;
  } else if ( m_config.enable_dma == 3 ) {
     dma_mode = dma_type::OVERSUB;
  } else {
     printf("Unknown DMA mode\n");
     exit(1);
  }

  if( m_config.eviction_policy == 0 ) {
     evict_policy = eviction_policy::LRU; 
  } else if (	m_config.eviction_policy == 1 ) {
     evict_policy = eviction_policy::TBN;
  } else if ( m_config.eviction_policy == 2 ) {
     evict_policy = eviction_policy::SEQUENTIAL_LOCAL;
  } else if ( m_config.eviction_policy == 3 ) {
     evict_policy = eviction_policy::RANDOM;
  } else if ( m_config.eviction_policy == 4 ) {
     evict_policy = eviction_policy::LFU;
  } else if ( m_config.eviction_policy == 5 ) {
     evict_policy = eviction_policy::LRU4K;
  } else {
      printf("Unknown eviction policy"); 
      exit(1);
  }

  if ( m_config.hardware_prefetch == 0 ) {
      prefetcher = hardware_prefetcher::DISABLED;
  } else if ( m_config.hardware_prefetch == 1 ) {
      prefetcher = hardware_prefetcher::TBN;
  } else if ( m_config.hardware_prefetch == 2 ) {
      prefetcher = hardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if ( m_config.hardware_prefetch == 3 ) {
      prefetcher = hardware_prefetcher::RANDOM;
  } else {
      printf("Unknown hardware prefeching policy");
      exit(1);
  }

  if ( m_config.hwprefetch_oversub == 0 ) {
      oversub_prefetcher = hardware_prefetcher_oversub::DISABLED;
  } else if ( m_config.hwprefetch_oversub == 1 ) {
      oversub_prefetcher = hardware_prefetcher_oversub::TBN;
  } else if ( m_config.hwprefetch_oversub == 2 ) {
      oversub_prefetcher = hardware_prefetcher_oversub::SEQUENTIAL_LOCAL;
  } else if ( m_config.hwprefetch_oversub == 3 ) {
      oversub_prefetcher = hardware_prefetcher_oversub::RANDOM;
  } else {
      printf("Unknown hardware prefeching policy under over-subscription");
      exit(1);
  }

  pcie_read_latency_queue = NULL;
  pcie_write_latency_queue = NULL;

  total_allocation_size = 0;

  over_sub = false;

  page_fault_latency = m_config.page_fault_latency;
  printf("page fault latency: %lu\n", page_fault_latency);
}

unsigned long long gmmu_t::calculate_transfer_time(size_t data_size) {
  float speed = 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(data_size) / (float)(1024)));

  if (data_size>= 2*1024*1024) {
    speed /= 2;
  }

  return  (unsigned long long) ( (float)(data_size) * m_config.core_freq / speed / (1024.0*1024.0*1024.0)) ;
}

void gmmu_t::calculate_devicesync_time(size_t data_size) {
  unsigned cur_turn = 0;
  unsigned cur_size = 0;

  float speed;

  while (data_size != 0) {
    unsigned long long cur_time = 0;

    if ( cur_turn == 0 ) {
      cur_size = MIN_PREFETCH_SIZE;
    } else {
      cur_size = MIN_PREFETCH_SIZE * pow(2, cur_turn - 1);
    }
    if (data_size < 4096) {
	    speed = 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(data_size) / (float)(1024)) );
	    cur_time = (unsigned long long) ( (float)(data_size) * m_config.core_freq / speed / (1024.0*1024.0*1024.0));
	   
	    //if (sim_prof_enable) {
	      //event_stats* d_sync  = new memory_stats(device_sync, cur_cycle, cur_cycle+cur_time, 0, data_size, 0);
        //sim_prof[cur_cycle].push_back(d_sync);		
      //}
      m_gpu->gpu_tot_sim_cycle += cur_time;
	    return;
    } else {
	    cur_size -= 4096;
	    data_size -= 4096;
	    speed = 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(4096) / (float)(1024)) );
      cur_time = (unsigned long long) ( (float)(4096) * m_config.core_freq / speed / (1024.0*1024.0*1024.0));

      //if (sim_prof_enable) {
        //event_stats* d_sync  = new memory_stats(device_sync, cur_cycle, cur_cycle+cur_time, 0, 4096, 0);
        //sim_prof[cur_cycle].push_back(d_sync);            
      //}
      m_gpu->gpu_tot_sim_cycle += cur_time;
    }

    if ( data_size < cur_size) {
      speed = 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(data_size) / (float)(1024)) );
      cur_time = (unsigned long long) ( (float)(data_size) * m_config.core_freq / speed / (1024.0*1024.0*1024.0));

      //if (sim_prof_enable) {
        //event_stats* d_sync  = new memory_stats(device_sync, cur_cycle, cur_cycle+cur_time, 0, data_size, 0);
        //sim_prof[cur_cycle].push_back(d_sync);            
      //}

	    m_gpu->gpu_tot_sim_cycle += cur_time;

      return;
    } else {
	    data_size -= cur_size;
	    speed = 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(cur_size) / (float)(1024)) );
      cur_time = (unsigned long long) ( (float)(cur_size) * m_config.core_freq / speed / (1024.0*1024.0*1024.0));

      //if (sim_prof_enable) {
        //event_stats* d_sync  = new memory_stats(device_sync, cur_cycle, cur_cycle+cur_time, 0, cur_size, 0);
        //sim_prof[cur_cycle].push_back(d_sync);            
      //}
	
      m_gpu->gpu_tot_sim_cycle += cur_time;
    }

    cur_turn++;
    if (cur_turn == 6) {
      cur_turn = 0;
    }
  }
  return;
}

bool gmmu_t::pcie_transfers_completed() {
  return pcie_write_stage_queue.empty() && 
         pcie_write_latency_queue == NULL && 
         pcie_read_stage_queue.empty() && 
         pcie_read_latency_queue == NULL;
}

void gmmu_t::register_tlbflush_callback(std::function<void(mem_addr_t)> cb_tlb) {
  callback_tlb_flush.push_back(cb_tlb);
}

void gmmu_t::tlb_flush(mem_addr_t page_num) {
  for (std::list<std::function<void(mem_addr_t)> >::iterator iter = callback_tlb_flush.begin();
      iter != callback_tlb_flush.end(); iter++) {
    (*iter)(page_num);
  }
}

void gmmu_t::check_write_stage_queue(mem_addr_t page_num, bool refresh) {
  // the page, about to be accessed, was selected for eviction earlier 
  // so don't evict that page
  for (std::list<pcie_latency_t*>::iterator iter = pcie_write_stage_queue.begin(); 
      iter != pcie_write_stage_queue.end(); iter++) {
    // already issued to be evicted from the GPU
    if (!(*iter)->is_touched) {
      if (std::find((*iter)->page_list.begin(), (*iter)->page_list.end(), page_num) != (*iter)->page_list.end()) {
        // on tlb hit refresh position of pages in the valid page list
        for (std::list<mem_addr_t>::iterator pg_iter = (*iter)->page_list.begin(); 
            pg_iter != (*iter)->page_list.end(); pg_iter++) {
          assert(pending_wb.find(*pg_iter) != pending_wb.end());
          pending_wb.erase(*pg_iter);
          m_gpu->get_global_memory()->set_page_access(*pg_iter);

          m_gpu->get_global_memory()->set_page_dirty(*pg_iter);

          // reclaim valid size in large page tree for unique basic blocks corresponding to all pages
          mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(*pg_iter);
          struct lp_tree_node* root = get_lp_node(page_addr);
          update_basic_block(root, page_addr, m_config.page_size, true);

          refresh_valid_pages(page_addr);
        }

        pcie_write_stage_queue.erase( iter );
        break;
      } 
    }
  }    
}

// check if the block is already scheduled for eviction or is not valid at all
bool gmmu_t::is_block_evictable(mem_addr_t addr, size_t size) {
  for (mem_addr_t start = addr; start != addr + size; start += m_config.page_size) {
    if (!m_gpu->get_global_memory()->is_valid( m_gpu->get_global_memory()->get_page_num(start))) {
      return false;
    }
  }

  for (std::list<pcie_latency_t*>::iterator iter = pcie_write_stage_queue.begin();
        iter != pcie_write_stage_queue.end(); iter++) {
    if ((addr >= (*iter)->start_addr) && (addr < (*iter)->start_addr + (*iter)->size)) {
      return false;
    }
  }

  for (mem_addr_t start = addr; start != addr + size; start += m_config.page_size) {
    if (is_page_reserved(start)) {
      return false;
    }
  }
  return true;
}

void gmmu_t::page_eviction_procedure() {
  memory_space* gmem = m_gpu->get_global_memory();
  sort_valid_pages();

  std::list<std::pair<mem_addr_t, size_t>> evicted_pages;

  int eviction_start = (int) (valid_pages.size() * m_config.reserve_accessed_page_percent / 100);

  if (evict_policy == eviction_policy::LRU4K) {
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance( iter, eviction_start );

    while (iter != valid_pages.end() && !is_block_evictable( (*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, false);

      evicted_pages.push_back(std::make_pair( page_addr, m_config.page_size));
    }
  } else if (evict_policy == eviction_policy::LRU || 
      evict_policy == eviction_policy::LFU || 
      m_config.page_size == MAX_PREFETCH_SIZE ) {
      // in lru, only evict the least recently used pages at the front of accessed pages queue
      // in lfu, only evict the page accessed least number of times from the front of accessed pages queue
      std::list<eviction_t *>::iterator iter = valid_pages.begin();
      std::advance(iter, eviction_start);

      while (iter != valid_pages.end() && !is_block_evictable( (*iter)->addr, (*iter)->size)) {
        iter++;
      }

      if (iter != valid_pages.end()) {
        mem_addr_t page_addr = (*iter)->addr;
        struct lp_tree_node *root = get_lp_node(page_addr);
        evict_whole_tree(root);

        evicted_pages.push_back(std::make_pair( root->addr, root->size));
      }
  } else if (evict_policy == eviction_policy::RANDOM) {
    // in random eviction, select a random page
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start + (rand() % (int)(valid_pages.size() * (1 - m_config.reserve_accessed_page_percent / 100))));

    while (iter != valid_pages.end() && !is_block_evictable( (*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, false);

      evicted_pages.push_back(std::make_pair(page_addr, m_config.page_size));
    }
  } else if (evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    // we evict sixteen 4KB pages in the 2 MB allocation where this evictable belong to
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance( iter, eviction_start );

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;

      root = get_lp_node(page_addr);

      bb_addr = get_basic_block(root, page_addr);

      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        break;
      }
    }
    if (iter != valid_pages.end()) {
      evicted_pages.push_back( std::make_pair( bb_addr, MIN_PREFETCH_SIZE));
    }
  } else if (evict_policy == eviction_policy::TBN) {
    // we evict multiple 64KB pages in the 2 MB allocation where this evictable belong
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    // find all basic blocks which are not staged/scheduled for write back or not invalid or not in ld/st unit
    std::set<mem_addr_t> all_basic_blocks;

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;

      root = get_lp_node(page_addr);

      bb_addr = get_basic_block(root, page_addr);

      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        break;
      }
    }

    if (iter != valid_pages.end()) {
      all_basic_blocks.insert( bb_addr );
      traverse_and_remove_lp_tree(root, all_basic_blocks);
    }

    // group all contiguous basic blocks if possible
    std::set<mem_addr_t>::iterator bb = all_basic_blocks.begin();

    while (bb != all_basic_blocks.end()) {
      std::set<mem_addr_t>::iterator next_bb = bb;
      size_t cur_num = 0;

      do {
        next_bb++;
        cur_num++;
      } while (next_bb != all_basic_blocks.end() && ((*next_bb) == ((*bb) + cur_num * MIN_PREFETCH_SIZE)));

      evicted_pages.push_back( std::make_pair((*bb), (cur_num * MIN_PREFETCH_SIZE)));

      bb = next_bb;
    }
  }

  // always write back the chunk no matter what it has not dirty pages or dirty pages
  for (std::list<std::pair<mem_addr_t, size_t> >::iterator iter = evicted_pages.begin(); 
      iter != evicted_pages.end(); iter++) {
    pcie_latency_t *p_t = new pcie_latency_t();

    p_t->start_addr = iter->first;
    p_t->size = iter->second;

    latency_type ltype = latency_type::PCIE_WRITE_BACK;

    for (std::list<eviction_t *>::iterator it = valid_pages.begin(); 
        it != valid_pages.end(); it++) {
      if ((*it)->addr <= iter->first && iter->first < (*it)->addr + (*it)->size) {
        if ((*it)->RW == 1) {
          ltype = latency_type::INVALIDATE;
          break;
        }
      }
    }

    p_t->type = ltype;

    if (m_config.page_size == MAX_PREFETCH_SIZE) {
      mem_addr_t page_num = gmem->get_page_num(iter->first);

      p_t->page_list.push_back( page_num );

      valid_pages_erase(page_num);
    } else {
      mem_addr_t page_num = gmem->get_page_num(iter->first);
      for (int i = 0; i < (int)(iter->second / m_config.page_size); i++) {
        if (pending_wb.find(page_num + i) == pending_wb.end()) {
          pending_wb.insert(page_num + i);
          p_t->page_list.push_back( page_num + i );

          valid_pages_erase(page_num + i);
        }
      }
    }

    if (p_t->page_list.size() != 0) {
      p_t->is_touched = false;
      pcie_write_stage_queue.push_back( p_t );
    } else {
      delete p_t;
    }
  }
}

void gmmu_t::valid_pages_erase(mem_addr_t page_num) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  for (std::list<eviction_t *>::iterator it = valid_pages.begin(); 
      it != valid_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      valid_pages.erase(it);
      break;
    }
  }
}

void gmmu_t::valid_pages_clear() {
  valid_pages.clear();
}

void gmmu_t::refresh_valid_pages(mem_addr_t page_addr, bool is_gmmu) {
  bool valid = false;
  for (std::list<eviction_t *>::iterator it = valid_pages.begin(); 
      it != valid_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      (*it)->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      valid = true;
      break;
    }
  }

  if (!valid) {
    eviction_t *item = new eviction_t();
    item->addr = get_eviction_base_addr(page_addr);
    assert(item->addr % MIN_PREFETCH_SIZE == 0);
    item->size = get_eviction_granularity(page_addr);
    item->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
    valid_pages.push_back(item);
  }
}

void gmmu_t::sort_valid_pages() {
  for (std::list<eviction_t *>::iterator vp_iter = valid_pages.begin(); 
      vp_iter != valid_pages.end(); vp_iter++) {
    for (std::list<struct lp_tree_node*>::iterator lp_iter = large_page_info.begin(); 
        lp_iter != large_page_info.end(); lp_iter++) {
      if ((*vp_iter)->addr == (*lp_iter)->addr) {
        (*vp_iter)->access_counter = (*lp_iter)->access_counter;
        (*vp_iter)->RW = (*lp_iter)->RW;
        break;
      }
    }
  }

  if (evict_policy == eviction_policy::LFU) {
    valid_pages.sort([](const eviction_t* i, const eviction_t* j) { 
        return (i->access_counter < j->access_counter) || 
               ((i->access_counter == j->access_counter) && (i->RW < j->RW)) || 
               ((i->access_counter == j->access_counter) && (i->RW == j->RW) && (i->cycle < j->cycle)); 
    });
  } else {
    if (evict_policy == eviction_policy::TBN || evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
      std::map<mem_addr_t, std::list<eviction_t *> > tempMap;

      for (std::list<eviction_t *>::iterator it = valid_pages.begin(); 
          it != valid_pages.end(); it++) {
        struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node((*it)->addr);
        tempMap[root->addr].push_back(*it);
      }

      for (std::map<mem_addr_t, std::list<eviction_t *> >::iterator it = tempMap.begin(); 
          it != tempMap.end(); it++) {
        it->second.sort([](const eviction_t* i, const eviction_t* j) { return i->cycle > j->cycle; });
      }

      std::list<std::pair< mem_addr_t, std::list<eviction_t *>>> tempList;

      for (std::map<mem_addr_t, std::list<eviction_t *> >::iterator it = tempMap.begin(); 
          it != tempMap.end(); it++) {
        tempList.push_back(make_pair(it->first, it->second));
      }

      tempList.sort([](
          const std::pair< mem_addr_t, std::list<eviction_t *>> i, 
          const std::pair< mem_addr_t, std::list<eviction_t *>> j) { 
            return i.second.front()->cycle < j.second.front()->cycle; 
          }
      );

      std::list<eviction_t *> new_valid_pages;

      for (std::list<std::pair< mem_addr_t, std::list<eviction_t *> > >::iterator it = tempList.begin(); 
          it != tempList.end(); it++) {
        (*it).second.sort([](const eviction_t* i, const eviction_t* j) { return i->cycle < j->cycle; });
        new_valid_pages.insert(new_valid_pages.end(), it->second.begin(), it->second.end());
      }

      valid_pages = new_valid_pages;
    } else {
      valid_pages.sort([](const eviction_t* i, const eviction_t* j) { return i->cycle < j->cycle; });
    }
  }
}

unsigned long long gmmu_t::get_ready_cycle(unsigned num_pages) {
  float speed = 2.0 * m_config.curve_a / M_PI * atan(m_config.curve_b * ((float)(num_pages*m_config.page_size)/1024.0));
  
  return  m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle + 
          (unsigned long long) 
          ((float)(m_config.page_size * num_pages) * m_config.core_freq / speed / (1024.0*1024.0*1024.0));
}

unsigned long long gmmu_t::get_ready_cycle_dma(unsigned size) {
   float speed = 2.0 * m_config.curve_a / M_PI * atan(m_config.curve_b * ((float)(size)/1024.0));
   return  m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle + 200;
}

float gmmu_t::get_pcie_utilization(unsigned num_pages) {
  return 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(num_pages*m_config.page_size)/1024.0) ) / m_config.pcie_bandwidth;
}

void gmmu_t::activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt, struct CUstream_st *m_stream) {
  for (std::list<prefetch_req>::iterator iter = prefetch_req_buffer.begin(); 
      iter!=prefetch_req_buffer.end(); iter++) {
    if (iter->start_addr == m_device_addr && 
        iter->size == m_cnt && 
        iter->m_stream->get_uid() == m_stream->get_uid()) {
      assert(iter->cur_addr == m_device_addr);
      iter->active = true;
      return;
    }
  }
}

void gmmu_t::register_prefetch(mem_addr_t m_device_addr, 
                               mem_addr_t m_device_allocation_ptr, 
                               size_t m_cnt, struct CUstream_st *m_stream) {
  struct prefetch_req pre_q;

  pre_q.start_addr = m_device_addr;
  pre_q.cur_addr = m_device_addr;
  pre_q.allocation_addr = m_device_allocation_ptr;
  pre_q.size = m_cnt;
  pre_q.active = false;
  pre_q.m_stream = m_stream;

  prefetch_req_buffer.push_back(pre_q);
}

struct lp_tree_node* gmmu_t::build_lp_tree(mem_addr_t addr, size_t size) {
	struct lp_tree_node *node = new lp_tree_node();
	node->addr = addr;
	node->size = size;
	node->valid_size = 0;
	node->access_counter = 0;
	node->RW = 0;
	
	if (size == MIN_PREFETCH_SIZE) {
		node->left = NULL;
		node->right = NULL;
	} else {
		node->left = build_lp_tree(addr, size/2);
		node->right = build_lp_tree(addr + size/2, size/2);
	}
	return node;
}

void gmmu_t::initialize_large_page(mem_addr_t start_addr, size_t size) {
  struct lp_tree_node * root = build_lp_tree(start_addr, size);
  large_page_info.push_back(root);
  total_allocation_size += size;
} 

struct lp_tree_node* gmmu_t::get_lp_node(mem_addr_t addr) {
  for (std::list<struct lp_tree_node*>::iterator iter = large_page_info.begin(); 
      iter != large_page_info.end(); iter++) {
    if ((*iter)->addr <= addr && addr < (*iter)->addr + (*iter)->size) {
      return *iter;
    }
  }
  return NULL;
}

mem_addr_t gmmu_t::get_basic_block(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr && addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return node->addr;
}

void gmmu_t::evict_whole_tree(struct lp_tree_node *node) {
  if (node != NULL) {
    node->valid_size = 0;
    evict_whole_tree(node->left);
    evict_whole_tree(node->right);
  }
}

mem_addr_t gmmu_t::update_basic_block(struct lp_tree_node *node, 
                                      mem_addr_t addr, 
                                      size_t size,  bool prefetch) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (prefetch) {
      if (node->valid_size != node->size) {
        node->valid_size += size;
      }
    } else {
      if ( node->valid_size != 0 ) {
        node->valid_size -= size;
      }
    }

    if (node->left->addr <= addr && addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  if (prefetch) {
    if (node->valid_size != node->size) {
      node->valid_size += size;
    }
  } else {
    if ( node->valid_size != 0 ) {
      node->valid_size -= size;
    }
  }

  return node->addr;
}

void gmmu_t::fill_lp_tree(struct lp_tree_node* node, 
                          std::set<mem_addr_t>& scheduled_basic_blocks) {
  if (node->size == MIN_PREFETCH_SIZE) {
    if (node->valid_size == 0) {
			node->valid_size = MIN_PREFETCH_SIZE;
			scheduled_basic_blocks.insert(node->addr);
		}
	} else {
		fill_lp_tree(node->left, scheduled_basic_blocks);
		fill_lp_tree(node->right, scheduled_basic_blocks);
		node->valid_size = node->left->valid_size + node->right->valid_size;
	}
}

void gmmu_t::remove_lp_tree(struct lp_tree_node* node, 
                            std::set<mem_addr_t>& scheduled_basic_blocks) {
	if (node->size == MIN_PREFETCH_SIZE) {
		if (node->valid_size == MIN_PREFETCH_SIZE && is_block_evictable(node->addr, MIN_PREFETCH_SIZE)) {
			node->valid_size = 0;
			scheduled_basic_blocks.insert(node->addr);
		}
	} else {
		remove_lp_tree(node->left, scheduled_basic_blocks);
		remove_lp_tree(node->right, scheduled_basic_blocks);
		node->valid_size = node->left->valid_size + node->right->valid_size;
	}
}

void gmmu_t::traverse_and_fill_lp_tree(struct lp_tree_node* node, 
                                       std::set<mem_addr_t>& scheduled_basic_blocks) {
	if (node->size != MIN_PREFETCH_SIZE) {
		traverse_and_fill_lp_tree(node->left, scheduled_basic_blocks);
		traverse_and_fill_lp_tree(node->right, scheduled_basic_blocks);
		node->valid_size = node->left->valid_size + node->right->valid_size;
	
		if (node->valid_size != node->size && node->valid_size > node->size / 2) {
			fill_lp_tree(node, scheduled_basic_blocks);
		}
	}
}

void gmmu_t::traverse_and_remove_lp_tree(struct lp_tree_node* node, 
                                         std::set<mem_addr_t>& scheduled_basic_blocks) {
	if (node->size != MIN_PREFETCH_SIZE) {
		traverse_and_remove_lp_tree(node->left, scheduled_basic_blocks);
		traverse_and_remove_lp_tree(node->right, scheduled_basic_blocks);
		node->valid_size = node->left->valid_size + node->right->valid_size;
	
		if (node->valid_size != 0 && node->valid_size < node->size / 2) {
			remove_lp_tree(node, scheduled_basic_blocks);
		}
	}
}

void gmmu_t::reserve_pages_insert(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  if (reserve_pages[page_num].find(ma_uid) == reserve_pages[page_num].end())
    reserve_pages[page_num].insert(ma_uid);

  //if (find(reserve_pages[page_num].begin(), reserve_pages[page_num].end(), ma_uid) == reserve_pages[page_num].end()) { 
    //reserve_pages[page_num].push_back(ma_uid);
  //}
}

void gmmu_t::reserve_pages_remove(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);
  assert(reserve_pages.find(page_num) != reserve_pages.end());
  assert(reserve_pages[page_num].find(ma_uid) != reserve_pages[page_num].end());
  reserve_pages[page_num].erase(ma_uid);
  if (reserve_pages[page_num].empty()) {
    reserve_pages.erase(page_num);
  }


  //std::list<unsigned>::iterator iter = std::find( reserve_pages[page_num].begin(), reserve_pages[page_num].end(), ma_uid );

  //assert(iter !=  reserve_pages[page_num].end());

  //reserve_pages[page_num].erase(iter);

  //if (reserve_pages[page_num].empty()) {
    //reserve_pages.erase(page_num);
  //}
}

bool gmmu_t::is_page_reserved(mem_addr_t addr) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  return reserve_pages.find(page_num) != reserve_pages.end();
}

void gmmu_t::check_reserved_pages() {
  std::cout << "num reserved pages: " << reserve_pages.size() << std::endl;
}



void gmmu_t::update_hardware_prefetcher_oversubscribed() {
  if (oversub_prefetcher == hardware_prefetcher_oversub::DISABLED) {
    prefetcher = hardware_prefetcher::DISABLED;
  } else if (oversub_prefetcher == hardware_prefetcher_oversub::TBN) {
    prefetcher = hardware_prefetcher::TBN;
  } else if (oversub_prefetcher == hardware_prefetcher_oversub::SEQUENTIAL_LOCAL) {
    prefetcher = hardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if (oversub_prefetcher == hardware_prefetcher_oversub::RANDOM) {
    prefetcher = hardware_prefetcher::RANDOM;
  }
}

void gmmu_t::update_memory_management_policy() {
  std::map<std::string, ds_pattern> accessPatterns;

  int i = 1;
  std::map<std::pair<mem_addr_t, size_t>, std::string> dataStructures;
  std::map<std::string, std::list<mem_addr_t> > dsUniqueBlocks;

  // get the managed allocations 
  const std::map<uint64_t, struct allocation_info*>& managedAllocations = 
    m_gpu->gpu_get_managed_allocations();

  // loop over managed allocations to create three maps
  // 1. data structures - key: pair of start addr and size; value: ds_i
  // 2. access pattern: key: ds_i; value: UNDECIDED pattern
  // 3. unique accessed blocks for reuse: key: ds_i; value: empty list of block start address 
  for (std::map<uint64_t, struct allocation_info*>::const_iterator iter = managedAllocations.begin(); 
      iter != managedAllocations.end(); iter++) {
    dataStructures.insert(
        std::make_pair(std::make_pair(iter->second->gpu_mem_addr, 
        iter->second->allocation_size), 
        std::string("ds" + std::to_string(i))));

    accessPatterns.insert(
        std::make_pair(std::string("ds" + std::to_string(i)), ds_pattern::UNDECIDED));
    dsUniqueBlocks.insert(
        std::make_pair(std::string("ds" + std::to_string(i)), std::list<mem_addr_t>()));
    i++;
  }

  // create three level hierarchy for kernel-wise then data-structure wise block address
  // first level: name of kernel (k_i); second level: ds_i; third level: block addresses ordered by access time
  std::map<unsigned, std::map<std::string, std::list<mem_addr_t>>> kernel_pattern;

  for (std::map<unsigned, std::pair<unsigned long long, unsigned long long> >::iterator k_iter = kernel_info.begin(); 
      k_iter != kernel_info.end(); k_iter++) {

    unsigned long long start = k_iter->second.first;
    unsigned long long end = k_iter->second.second;

    std::map<std::string, std::list<mem_addr_t> > dsAccess;

    for (std::list<std::pair<unsigned long long , mem_addr_t> >::iterator acc_iter = block_access_list.begin(); 
        acc_iter != block_access_list.end(); acc_iter++ ) {

      unsigned long long access_cycle = acc_iter->first;
      mem_addr_t block_addr = acc_iter->second;

      if (access_cycle >= start && ((end == 0) || (access_cycle <= end))) {

        for (std::map<std::pair<mem_addr_t, size_t>, std::string>::iterator ds_iter = dataStructures.begin(); 
            ds_iter != dataStructures.end(); ds_iter++ ) {

          if (block_addr >= ds_iter->first.first && block_addr < ds_iter->first.first + ds_iter->first.second) {
            dsAccess[ds_iter->second].push_back(block_addr);
          }
        }
      }
    }

    kernel_pattern.insert(std::make_pair(k_iter->first, dsAccess));
  }

  // determine pattern per data structure 
  // first loop on kernel level then on data structures accessed in that kernel
  for (std::map<unsigned, std::map<std::string, std::list<mem_addr_t> > >::iterator k_iter = kernel_pattern.begin(); 
      k_iter != kernel_pattern.end(); k_iter++) {

    for (std::map<std::string, std::list<mem_addr_t> >::iterator da_iter = k_iter->second.begin(); 
        da_iter != k_iter->second.end(); da_iter++) {

      // get the sorted list of block addresses belonging to the current data-structure in current kernel
      std::list<mem_addr_t> curBlocks = std::list<mem_addr_t>(da_iter->second);
      curBlocks.sort();
      curBlocks.unique();

      // check for data reuse
      bool reuse = false;

      // first within this kernel
      // if the number of unique blocks accessed and total number of blocks accessed are not same then there is repetition
      if (curBlocks.size() != da_iter->second.size()) {
        reuse = true;
      }

      // second check if the current accessed blocks are already seen or not
      std::map<std::string, std::list<mem_addr_t> >::iterator ub_it = dsUniqueBlocks.find(da_iter->first);

      // check for intersection between unique blocks accessed in current kernel and the previous kernels is null set or not
      std::list<int> intersection;
      std::set_intersection(
          curBlocks.begin(), 
          curBlocks.end(), ub_it->second.begin(), 
          ub_it->second.end(), 
          std::back_inserter(intersection));

      if (intersection.size() != 0) {
        reuse = true;
      }

      // add the current blocks to the seen set per data structure
      ub_it->second.merge(curBlocks);
      ub_it->second.sort();
      ub_it->second.unique();

      // now update the pattern
      std::map<std::string, ds_pattern>::iterator dsp_it = accessPatterns.find(da_iter->first);
      ds_pattern curPattern;

      // check for linearity or randomness in current kernel
      if (std::is_sorted(da_iter->second.begin(), da_iter->second.end())) {
        if (reuse) {
          curPattern = ds_pattern::LINEAR_REUSE;
        } else {
          curPattern = ds_pattern::LINEAR;
        }
      } else {
        if (reuse) {
          curPattern = ds_pattern::RANDOM_REUSE;
        } else {
          curPattern = ds_pattern::RANDOM;
        }
      }

      // determine the pattern
      if (dsp_it->second == ds_pattern::UNDECIDED) {
        dsp_it->second = curPattern;
      } else if (dsp_it->second == ds_pattern::LINEAR) { 
        if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::LINEAR_REUSE;
        } else if (curPattern == ds_pattern::RANDOM) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::LINEAR_REUSE) { 
        if (curPattern == ds_pattern::RANDOM || curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM) { 
        if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::RANDOM_REUSE;
        } else if (curPattern == ds_pattern::LINEAR) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM_REUSE) { 
        if (curPattern == ds_pattern::LINEAR || curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      }
    }
  }

  bool is_random = false;
  bool is_random_reuse = false;
  bool is_linear = false; 
  bool is_linear_reuse = false;
  bool is_mixed = false;
  bool is_mixed_reuse = false;

  for (std::map<std::string, ds_pattern>::iterator ap_iter = accessPatterns.begin(); 
      ap_iter != accessPatterns.end(); ap_iter++) {
    if (ap_iter->second == ds_pattern::RANDOM) {
      is_random = true;
    } else if (ap_iter->second == ds_pattern::RANDOM_REUSE) {
      is_random_reuse = true;
    } else if (ap_iter->second == ds_pattern::LINEAR) {
      is_linear = true;
    } else if (ap_iter->second == ds_pattern::LINEAR_REUSE) {
      is_linear_reuse = true;
    } else if (ap_iter->second == ds_pattern::MIXED) {
      is_mixed = true;
    } else if (ap_iter->second == ds_pattern::MIXED_REUSE) {
      is_mixed_reuse = true;
    }
  }

  if (is_random || is_random_reuse || is_mixed || is_mixed_reuse) {
    dma_mode = dma_type::OVERSUB;
    evict_policy = eviction_policy::TBN;
  } else if (is_linear_reuse) {
    evict_policy = eviction_policy::TBN;
  }
}

void gmmu_t::reset_lp_tree_node(struct lp_tree_node* node) {
  node->valid_size = 0;
  node->access_counter = 0;
  node->RW = 0;

  if (node->size != MIN_PREFETCH_SIZE) {
    reset_lp_tree_node(node->left);
    reset_lp_tree_node(node->right);
  }
}

void gmmu_t::reset_large_page_info() {
  for (std::list<struct lp_tree_node*>::iterator iter = large_page_info.begin(); 
      iter != large_page_info.end(); iter++) {
    reset_lp_tree_node(*iter);
  }    

  over_sub = false;
}

mem_addr_t gmmu_t::get_eviction_base_addr(mem_addr_t page_addr) {
  mem_addr_t lru_addr;

  struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(page_addr);

  if (evict_policy == eviction_policy::TBN || evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_addr = m_gpu->getGmmu()->get_basic_block(root, page_addr);
  } else if (evict_policy == eviction_policy::LRU4K || evict_policy == eviction_policy::RANDOM) {
    lru_addr = page_addr;
  } else {
    lru_addr = root->addr;
  }

  return lru_addr;
}

size_t gmmu_t::get_eviction_granularity(mem_addr_t page_addr) {
  size_t lru_size;

  struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(page_addr);

  if (evict_policy == eviction_policy::TBN || evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_size = MIN_PREFETCH_SIZE;
  } else if (evict_policy == eviction_policy::LRU4K || evict_policy == eviction_policy::RANDOM) {
    lru_size = m_config.page_size;
  } else {
    lru_size = root->size;
  }
  
  return lru_size;
}

void gmmu_t::update_access_type(mem_addr_t addr, int type) {
  struct lp_tree_node *node = m_gpu->getGmmu()->get_lp_node(addr);
  assert(node != NULL);
  while (node->size != MIN_PREFETCH_SIZE) {
    node->RW |= type;

    if (node->left->addr <= addr && addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  node->RW |= type;
}

int gmmu_t::get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr) {  
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr && addr < node->left->addr + node->left->size) {
      node = node->left;
    } else { 
      node = node->right;
    }
  }

  return node->access_counter & ((1 << 27) - 1);
}

int gmmu_t::get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr) {  
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr && addr < node->left->addr + node->left->size) {
      node = node->left;
    } else { 
      node = node->right;
    }
  }   

  return (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
}

void gmmu_t::inc_bb_access_counter(mem_addr_t addr) {
  struct lp_tree_node *node = m_gpu->getGmmu()->get_lp_node(addr);

  while (node->size != MIN_PREFETCH_SIZE) {
    node->access_counter++;

    if (node->left->addr <= addr && addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  if (node->access_counter == ((1 << 27) - 1)) {
    reset_bb_access_counter();
  }

  node->access_counter++;
}

void gmmu_t::inc_bb_round_trip(struct lp_tree_node *node) {
  if (node->size != MIN_PREFETCH_SIZE) {
    inc_bb_round_trip(node->left);
    inc_bb_round_trip(node->right);
  } else {
    uint16_t round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;

    if (round_trip == ((1 << 6) - 1)) {
      reset_bb_round_trip();
    }

    round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    round_trip++;

    node->access_counter = (round_trip << 27) | (node->access_counter & ((1 << 27) - 1));
  }
}

void gmmu_t::traverse_and_reset_access_counter(struct lp_tree_node *node) {
  if (node->size == MIN_PREFETCH_SIZE) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    int access_counter = (node->access_counter & ((1 << 27) - 1)) >> 1;

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
    node->access_counter = node->access_counter >> 1;
  }
}

void gmmu_t::reset_bb_access_counter() {
  for (std::list<struct lp_tree_node*>::iterator iter = large_page_info.begin(); 
      iter != large_page_info.end(); iter++) {
    traverse_and_reset_access_counter(*iter);
  }
}

void gmmu_t::traverse_and_reset_round_trip(struct lp_tree_node *node) {
  if (node->size == MIN_PREFETCH_SIZE) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 28;
    int access_counter = node->access_counter & ((1 << 27) - 1);

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
  }
}

void gmmu_t::reset_bb_round_trip() {
  for (std::list<struct lp_tree_node*>::iterator iter = large_page_info.begin(); 
      iter != large_page_info.end(); iter++) {
    traverse_and_reset_round_trip(*iter);
  }
}

bool gmmu_t::should_cause_page_migration(mem_addr_t addr, bool is_write) {
  if (dma_mode == dma_type::DISABLED) {
    return true;
  } else if (dma_mode == dma_type::ALWAYS) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);
      if (get_bb_access_counter(root, addr) < m_config.migrate_threshold) {
        return false;
      } else {
        return true;
      }
    }
  } else if (dma_mode == dma_type::OVERSUB) {
    if (over_sub) {
      if (is_write) {
        return true;
      } else {
        struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

        if (get_bb_access_counter(root, addr) < m_config.migrate_threshold) {
          return false;
        } else {
          return true;
        }
      }
    } else {
      return true;
    }
  } else if (dma_mode == dma_type::ADAPTIVE) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

      int derived_threshold;

      if (over_sub) {
        derived_threshold = m_config.migrate_threshold * m_config.multiply_dma_penalty * (get_bb_round_trip(root, addr) + 1);
      } else {
        size_t num_read_stage_queue = 0;

        for (std::list<pcie_latency_t*>::iterator iter = pcie_read_stage_queue.begin(); 
            iter != pcie_read_stage_queue.end(); iter++) {
          num_read_stage_queue += (*iter)->page_list.size();
        }

        size_t num_write_stage_queue = 0;

        for (std::list<pcie_latency_t*>::iterator iter = pcie_write_stage_queue.begin(); 
            iter != pcie_write_stage_queue.end(); iter++) {
          num_write_stage_queue += (*iter)->page_list.size();
        }

        derived_threshold = 
          (int)(1.0 + m_config.migrate_threshold * m_gpu->get_global_memory()->get_projected_occupancy(
                num_read_stage_queue, num_write_stage_queue, m_config.free_page_buffer_percentage));
      }

      if (get_bb_access_counter(root, addr) < derived_threshold) {
        return false;
      } else {
        return true;
      }
    }
  } else {
    assert(false);
    return false;
  }
}

void gmmu_t::cycle() {
  memory_space* gmem = m_gpu->get_global_memory();
  unsigned long long gpu_total_cycle = m_gpu->get_total_cycle();
  int simt_cluster_id = 0;

  size_t num_read_stage_queue = 0;
  
  for (std::list<pcie_latency_t*>::iterator iter = pcie_read_stage_queue.begin(); 
      iter != pcie_read_stage_queue.end(); iter++) {
    if (!(*iter)->is_touched)
      num_read_stage_queue += (*iter)->page_list.size();
    unsigned threshold = ((m_config.free_page_buffer_percentage * 2) / 100) * gmem->get_total_pages();
    if (num_read_stage_queue > threshold)
      break;
  }

  size_t num_write_stage_queue = 0;
  for (std::list<pcie_latency_t*>::iterator iter = pcie_write_stage_queue.begin(); 
      iter != pcie_write_stage_queue.end(); iter++) {
    if (!(*iter)->is_touched)
      num_write_stage_queue += (*iter)->page_list.size();
  }
  num_write_stage_queue += num_sent;

  //num_write_stage_queue += (pcie_write_latency_queue != NULL) ? 
                           //pcie_write_latency_queue->page_list.size() : 0;

  if (gmem->should_evict_page(num_read_stage_queue, num_write_stage_queue, m_config.free_page_buffer_percentage)) {
    if (m_config.enable_smart_runtime) {
      update_memory_management_policy();
    }
    page_eviction_procedure();
  }
  // check whether the current transfer in the pcie write latency queue is
  // finished
  /*
  if (pcie_write_latency_queue != NULL && 
      (gpu_total_cycle >= pcie_write_latency_queue->ready_cycle)) {
    delete pcie_write_latency_queue;
    pcie_write_latency_queue = NULL;
  }
  */

  if (pcie_write_latency_queue != NULL &&
      (gpu_total_cycle >= pcie_write_latency_queue->ready_cycle)) {
    //mem_addr_t page_num = pcie_write_latency_queue->page_list.front();

    //std::cout << "write issue decrease: " << page_num << std::endl;
    //std::cout << "num sent: " << num_sent << std::endl;
    assert(num_sent > 0);
    delete pcie_write_latency_queue;
    pcie_write_latency_queue = NULL;

    
    write_issue_count -= 1;
    if (write_issue_count == 0) {
      num_sent = 0;
      write_issued = false;
    }
    
    num_host_memory_access_bytes += 4096; //4KB page write
  }
  
  // Page to be evicted is read from the memory
  if (!is_pcie_gmmu_invalidate_queue_empty()) {
    if (pcie_write_latency_queue == NULL) {
      mem_addr_t page_num = front_pcie_gmmu_invalidate_queue();

      pcie_latency_t *per_pg_latency = new pcie_latency_t();
      per_pg_latency->page_list.push_back(page_num);
      per_pg_latency->ready_cycle = get_pcie_page_ready_cycle();

      pcie_write_latency_queue = per_pg_latency;

      // Modify the page table
      pop_pcie_gmmu_invalidate_queue();
    }
  }

  // schedule a write back transfer if there is a write back request in staging queue and a free lane
  if (!pcie_write_stage_queue.empty() && !write_issued) {
    //pcie_write_latency_queue = pcie_write_stage_queue.front();
    pcie_latency_t* p_t = pcie_write_stage_queue.front();
    std::list<mem_addr_t> &page_list = p_t->page_list;
    assert(page_list.size() != 0);
    // Read each page from the memory
    if (page_list.size() != 0) {
      // If the page list is not touched yet, invalidate all pages
      if (!p_t->is_touched) {
        unsigned num_evicted = 0;
        assert(valid_page_to_be_evicted.size() == 0);
        for (mem_addr_t &page_num : page_list) {
          num_pcie_write_transfer += 1;
          gmem->invalidate_page(page_num);
          gmem->clear_page_dirty(page_num);
          gmem->clear_page_access(page_num);

          if (m_gpu->is_in_addr_range(page_num)) {
            gmem->free_pages(1);
            num_evicted += 1;
            valid_page_to_be_evicted.insert(page_num);
          }

          tlb_flush(page_num);

          assert(pending_wb.find(page_num) != pending_wb.end());
          pending_wb.erase(page_num);
          //std::cout << page_num << " ";
        }
        //std::cout << std::endl;
        p_t->is_touched = true;  // All pages in the page list is invalidated. Now disable to invalidate pages.
        num_sent = num_evicted;  // Used for pre eviction
        //std::cout << "set num sent: " << num_sent << std::endl;

        write_issue_count = num_evicted;
      }
      // If this is the last page in the list, true is sent.
      mem_addr_t cur_page_num = page_list.front();
      page_list.pop_front();
      if (valid_page_to_be_evicted.find(cur_page_num) !=
          valid_page_to_be_evicted.end()) {
        valid_page_to_be_evicted.erase(cur_page_num);
        send_pcie_requests(cur_page_num, valid_page_to_be_evicted.size() == 0, false);
      }

      if (page_list.size() == 0) {
        assert(valid_page_to_be_evicted.size() == 0);
        if (num_sent > 0) {
          write_issued = true;
        } else {
          assert(num_sent == 0);
          write_issued = false;
        }
        pcie_write_stage_queue.pop_front();
        delete p_t;
      }
    }
  }

  // If the PCIE request is done
  while (!is_pcie_gmmu_queue_empty()) {
    mem_addr_t page_num = front_pcie_gmmu_queue();
    // validate page
    gmem->validate_page(page_num);
    refresh_valid_pages(gmem->get_mem_addr(page_num), true);
    // Send request to the cores
    if (req_info.find(page_num) != req_info.end()) {
      for (auto it = req_info[page_num].begin(); it != req_info[page_num].end(); ++it) {
        mem_fetch* mf = *it;
        simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
        m_gpu->getSIMTCluster(simt_cluster_id)->push_gmmu_cu_queue(mf);
      }
      req_info.erase(page_num);
    }
    pop_pcie_gmmu_queue();
  }

  // check whether the current transfer in the pcie latency queue is finished
  // send single page
  if ((pcie_read_latency_queue != NULL) && 
      (gpu_total_cycle >= pcie_read_latency_queue->ready_cycle)) {
    pcie_latency_t *p_t = pcie_read_latency_queue;
    assert(p_t->page_list.size() == 1);
    
    send_pcie_requests(p_t->page_list.front(), p_t->is_last, true);
    delete pcie_read_latency_queue;
    pcie_read_latency_queue = NULL;

    num_host_memory_access_bytes += 4096; //4KB page read
  }


  // schedule a transfer if there is a pending item in staging queue and
  // nothing is being served at the read latency queue and we have available
  // free pages
  if (!pcie_read_stage_queue.empty() && pcie_read_latency_queue == NULL) {
    pcie_latency_t* p_t = pcie_read_stage_queue.front();
    std::list<mem_addr_t> &page_list = p_t->page_list;
    latency_type latency_type = p_t->type;
    unsigned long num_free_pages = gmem->get_free_pages();
    assert(page_list.size() != 0);
    if (num_free_pages >= page_list.size()) {
      if (page_list.size() != 0) {
        pcie_latency_t *per_pg_latency = new pcie_latency_t();
        per_pg_latency->page_list.push_back(page_list.front());
        
        per_pg_latency->type = latency_type::PCIE_READ;
        per_pg_latency->ready_cycle = get_pcie_page_ready_cycle();
        per_pg_latency->is_last = false;
        if (!p_t->is_touched) {
          per_pg_latency->ready_cycle += page_fault_latency;
          p_t->is_touched = true;
        }

        page_list.pop_front();

        if (page_list.size() == 0) {
          per_pg_latency->is_last = true;
          pcie_read_stage_queue.pop_front();
          delete p_t;
        }

        num_pcie_read_transfer += 1;
        pcie_read_latency_queue = per_pg_latency;

        gmem->alloc_pages(1);
        
      }
    }
  }

  std::map<mem_addr_t, std::list<mem_fetch*> > page_fault_this_turn;  

  // check the page_table_walk_delay_queue
  while (!page_table_walk_queue.empty() &&
         ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >= 
          page_table_walk_queue.front().ready_cycle)) {
    mem_fetch* mf = page_table_walk_queue.front().mf;
    mem_addr_t mem_addr = mf->get_addr();
    if (m_config.first_in_gpu) {
      std::list<mem_addr_t> first_in_gpu_validated_page_list = gmem->get_first_in_gpu_pages(mem_addr, mf->get_access_size());//page_num
      if (!first_in_gpu_validated_page_list.empty()) {
        for (std::list<mem_addr_t>::iterator pg_iter = first_in_gpu_validated_page_list.begin(); pg_iter != first_in_gpu_validated_page_list.end(); pg_iter++) {
          refresh_valid_pages(gmem->get_mem_addr(*pg_iter), true);
        }
      }
    }

    std::list<mem_addr_t> page_list = 
      gmem->get_faulty_pages(mem_addr, mf->get_access_size());
    //assert(m_gpu->is_in_addr_range(page_num));

    simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
    // If there is no page fault, directly return to the upward queue of cluster
    // Even though the page list is empty, eviction list can contain the target page
    if (page_list.empty()) {
      mem_addr_t page_num = gmem->get_page_num(mem_addr);
      check_write_stage_queue(page_num, false);
      (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
    } else {
      assert(page_list.size() == 1);
      mem_addr_t faulty_page_num = *(page_list.begin());

      if (req_info.find(faulty_page_num) != req_info.end()) {
        // Merge request
        req_info[faulty_page_num].push_back(mf);
      } else {
        page_fault_this_turn[faulty_page_num].push_back(mf);
      }
    }
    page_table_walk_queue.pop_front();
  }

  // call hardware prefetcher based on the current page faults
  do_hardware_prefetch(page_fault_this_turn);

  // fetch from cluster's cu to gmmu queue and push it into the page table walk delay queue
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
    if (!(m_gpu->getSIMTCluster(i))->empty_cu_gmmu_queue()) {
      mem_fetch *mf = (m_gpu->getSIMTCluster(i))->front_cu_gmmu_queue();

      struct page_table_walk_latency_t pt_t;
      pt_t.mf = mf;
      pt_t.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config.page_table_walk_latency;

      page_table_walk_queue.push_back(pt_t);

      m_gpu->getSIMTCluster(i)->pop_cu_gmmu_queue();
    }
  }
}

void gmmu_t::do_hardware_prefetch (std::map<mem_addr_t, std::list<mem_fetch*> > &page_fault_this_turn) {
  // now decide on transfers as a group of page faults and prefetches
  if (!page_fault_this_turn.empty()) {
    unsigned long long num_pages_read_stage_queue = 0;

    for (std::list<pcie_latency_t*>::iterator iter = pcie_read_stage_queue.begin();
       iter != pcie_read_stage_queue.end(); iter++) {
       num_pages_read_stage_queue += (*iter)->page_list.size();
    }

    std::list< std::list<mem_addr_t> > all_transfer_all_page;
    std::list< std::list<mem_addr_t> > all_transfer_faulty_pages;
    std::map<mem_addr_t, std::list<mem_fetch*> > temp_req_info;

    // create a tree structure large page -> basic blocks -> faulty pages
    std::map<mem_addr_t, std::map<mem_addr_t, std::list<mem_addr_t>>> block_tree;

    if (prefetcher == hardware_prefetcher::DISABLED || prefetcher == hardware_prefetcher::RANDOM) {
      for (std::map<mem_addr_t, std::list<mem_fetch*> >::iterator it = page_fault_this_turn.begin(); 
          it != page_fault_this_turn.end(); it++) {
        std::list<mem_addr_t> temp_pages;
        temp_pages.push_back(it->first);

        mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(it->first);
        struct lp_tree_node* root = get_lp_node(page_addr);
        update_basic_block(root, page_addr, m_config.page_size, true);

        all_transfer_all_page.push_back(temp_pages);
        all_transfer_faulty_pages.push_back(temp_pages);

        temp_req_info[it->first];

        if (prefetcher == hardware_prefetcher::RANDOM) {
          struct lp_tree_node* root = 
            get_lp_node(m_gpu->get_global_memory()->get_mem_addr(it->first));

          size_t random_size = ( rand() % (root->size / m_config.page_size) ) * m_config.page_size;

          if (random_size > root->size) {
            random_size -= root->size;
          }

          mem_addr_t prefetch_addr = root->addr + random_size;

          mem_addr_t prefetch_page_num = m_gpu->get_global_memory()->get_page_num(prefetch_addr);
		     
          if (!m_gpu->get_global_memory()->is_valid(prefetch_page_num) && 
              page_fault_this_turn.find(prefetch_addr) == page_fault_this_turn.end() &&
              temp_req_info.find(prefetch_page_num) == temp_req_info.end() &&
              req_info.find(prefetch_page_num) == req_info.end()) {
            mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(prefetch_page_num);
            struct lp_tree_node* root = get_lp_node(page_addr);
            update_basic_block(root, page_addr, m_config.page_size, true);

            all_transfer_all_page.back().push_back( prefetch_page_num );

            temp_req_info[prefetch_page_num];
          }
        }
      }
    } else {
      // 2MB node addr => 4KB page addr
      std::map<mem_addr_t, std::set<mem_addr_t> > lp_pf_groups;

      for (std::map<mem_addr_t, std::list<mem_fetch*> >::iterator it = page_fault_this_turn.begin(); 
          it != page_fault_this_turn.end(); it++) {
        mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(it->first);
          
        struct lp_tree_node* root = get_lp_node(page_addr);

        lp_pf_groups[root->addr].insert(page_addr);
      }

      // lp_pf_iter->first: 2MB page addr
      for (std::map<mem_addr_t, std::set<mem_addr_t> >::iterator lp_pf_iter = lp_pf_groups.begin(); 
          lp_pf_iter != lp_pf_groups.end(); lp_pf_iter++ ) {
        // contain 64KB page addrs
        std::set<mem_addr_t> schedulable_basic_blocks;

        // list of all invalid pages and pages with fault from all basic blocks to satisfy current transfer size
        std::list<mem_addr_t> cur_transfer_all_pages;
        // contain 4KB page num
        std::list<mem_addr_t> cur_transfer_faulty_pages;

        // lp_pf_iter->second: 4KB pages within 2MB page addr
        for (std::set<mem_addr_t>::iterator pf_iter = lp_pf_iter->second.begin(); 
            pf_iter != lp_pf_iter->second.end(); pf_iter++) {
          // 4KB page addr
          mem_addr_t page_addr = *pf_iter;

          struct lp_tree_node* root = get_lp_node(page_addr);

          // 64KB page addr
          mem_addr_t bb_addr = update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, true);

          schedulable_basic_blocks.insert(bb_addr);

          // real fauly pages
          cur_transfer_faulty_pages.push_back(m_gpu->get_global_memory()->get_page_num(page_addr));
        }

        if (prefetcher == hardware_prefetcher::TBN) {
          struct lp_tree_node* root = get_lp_node(lp_pf_iter->first);
          traverse_and_fill_lp_tree(root, schedulable_basic_blocks);
        }

        // for each 64 KB page addr
        for (std::set<mem_addr_t>::iterator bb = schedulable_basic_blocks.begin(); 
            bb != schedulable_basic_blocks.end(); bb++) {
          block_access_list.push_back(std::make_pair(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, *bb));

          if (m_config.first_in_gpu) {
            std::list<mem_addr_t> first_in_gpu_validated_page_list = m_gpu->get_global_memory()->get_first_in_gpu_pages(*bb, MIN_PREFETCH_SIZE);//page_num
            if (!first_in_gpu_validated_page_list.empty()) {
              for (std::list<mem_addr_t>::iterator pg_iter = first_in_gpu_validated_page_list.begin(); pg_iter != first_in_gpu_validated_page_list.end(); pg_iter++) {
                refresh_valid_pages(gmem->get_mem_addr(*pg_iter), true);
              }
            }
          }

          // all the invalid pages in the current 64 K basic block of transfer
          std::list<mem_addr_t> all_block_pages = 
            m_gpu->get_global_memory()->get_faulty_pages( *bb, MIN_PREFETCH_SIZE );

          // for each 4KB page within 64KB page addr
          for (std::list<mem_addr_t>::iterator pg_iter = all_block_pages.begin(); 
              pg_iter != all_block_pages.end(); pg_iter++) {
            // Chedk address range to prevent unrelated pages from being
            // fetched.
            if (m_gpu->is_in_addr_range(*pg_iter)) {
              if (temp_req_info.find(*pg_iter) == temp_req_info.end()) {
                // mark entry into mshr for all pages in the current basic block
                temp_req_info[*pg_iter];
                // faulty pages including real faulty page
                cur_transfer_all_pages.push_back(*pg_iter);
              }
            } else {
              // just validate page now
              // This is important because for correct write eviction
              gmem->validate_page(*pg_iter);
              assert(req_info.find(*pg_iter) == req_info.end());
            }
          } 
          //assert(!cur_transfer_all_pages.empty());
        }
        all_transfer_all_page.push_back(cur_transfer_all_pages);
        all_transfer_faulty_pages.push_back(cur_transfer_faulty_pages);
      }
    }

    
    for (std::map<mem_addr_t, std::list<mem_fetch*> >::iterator iter = temp_req_info.begin(); 
        iter != temp_req_info.end(); iter++) {
      req_info[iter->first];
      req_info[iter->first].merge(iter->second);
    }

    // Each pcie latency request corresponds to the basic block
    for (auto &bb : all_transfer_all_page) {
      pcie_latency_t* p_t = new pcie_latency_t();
      p_t->page_list = bb;
      p_t->size = p_t->page_list.size() * m_config.page_size;
      p_t->type = latency_type::PCIE_READ;
      p_t->is_touched = false;
      pcie_read_stage_queue.push_back(p_t);
    }
    
    // adding statistics for prefetch
    for (std::map<mem_addr_t, std::list<mem_fetch*> >::iterator iter2 = page_fault_this_turn.begin(); 
        iter2 != page_fault_this_turn.end(); iter2++) {
	    assert(req_info[iter2->first].size() == 0);
           
      // add the pending prefecthes to the MSHR entry
      req_info[iter2->first] = iter2->second;
    }

    if (!over_sub && m_gpu->get_global_memory()->should_evict_page(
          num_pages_read_stage_queue + temp_req_info.size(), 0, m_config.free_page_buffer_percentage)) {
      if (m_config.enable_smart_runtime) {
        update_memory_management_policy();
      } else {
		    update_hardware_prefetcher_oversubscribed();
      }
      over_sub = true;
    }
  }
}

void gmmu_t::send_pcie_requests(mem_addr_t page_num, bool is_last, bool is_validate) {
  std::map<unsigned, std::list<mem_addr_t>> ch_to_addrs;
  // Because each row buffer contains 256B page data
  unsigned range = 256;
  for (unsigned i = 0; i < 4096; i += range) {
    mem_addr_t page_addr = gmem->get_mem_addr(page_num) | i;
    addrdec_t raw_addr;
    m_gpu->getMemoryConfig()->m_address_mapping.addrdec_tlx(page_addr, &raw_addr);
    ch_to_addrs[raw_addr.chip].push_back(page_addr);
  }

  for (auto it = ch_to_addrs.begin(); it != ch_to_addrs.end(); ++it) {
    unsigned ch = it->first;
    auto page_addr_list = it->second;
    int page_data_per_ch = page_addr_list.size();
   
    memory_partition_unit* mp = m_gpu->get_mem_partition(ch);
    for (int i = 0; i < page_data_per_ch - 1; ++i) {
      mp->push_gmmu_pcie_queue(page_addr_list.front(), false, is_validate);
      page_addr_list.pop_front();
    }
    mp->push_gmmu_pcie_queue(page_addr_list.front(), true, is_validate);
    page_addr_list.pop_front();

  }
}


// send 4096 bytes
// 
unsigned long long gmmu_t::get_pcie_page_ready_cycle() {
  float bw = m_config.pcie_bandwidth * pow(1024.0, 3);
  float transfer_size = 4096;
  double core_period = m_config.core_period;
  unsigned long long time = (transfer_size / bw) / core_period;
  return m_gpu->get_total_cycle() + time;
}

