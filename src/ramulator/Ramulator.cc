#include <string>

#include "Ramulator.h"

#include "HBM.h"
#include "PCM.h"
#include "HMS.h"

#include "../cuda-sim/memory.h"
#include "../gpgpu-sim/l2cache.h"
#include "../gpgpu-sim/mem_latency_stat.h"

using namespace ramulator;

static map<string, 
           function<MemoryBase *(const Config&, int, int, 
                                 const memory_config*,
                                 class memory_partition_unit*,
                                 class gpgpu_sim*)>> name_to_func = {
    {"HBM", &MemoryFactory<HBM>::create},
    {"PCM", &MemoryFactory<PCM>::create},
    {"HMS", &MemoryFactory<HMS>::create},
};

Ramulator::Ramulator(unsigned int mem_partition_id,
                     const memory_config *m_config,
                     class memory_stats_t *stats,
                     class memory_partition_unit *mp,
                     class gpgpu_sim *gpu) :
    dram_t(mem_partition_id, m_config, stats, mp, gpu),
    read_cb_func(std::bind(&Ramulator::readComplete, this, std::placeholders::_1)),
    write_cb_func(std::bind(&Ramulator::writeComplete, this, std::placeholders::_1)),
    pcie_cb_func(std::bind(&Ramulator::pcieComplete, this, std::placeholders::_1)),
    ramulator_configs("./" + std::string(m_config->ramulator_mem_type) + "-config.cfg"),
    num_cores(gpu->get_config().num_shader() + 1) {

  assert(m_config->use_ramulator);
  // + 1 is added to consider core ID -1 (WRITE BACK)
  ramulator_configs.set_core_num(num_cores);
  std_name = ramulator_configs["standard"];
  assert(name_to_func.find(std_name) != name_to_func.end() &&
         "unrecognized standard name");

  memory = 
    name_to_func[std_name](ramulator_configs, (int)SECTOR_SIZE, 
                           mem_partition_id, m_config,
                           mp, gpu);

  tCK = memory->clk_ns(gpu->get_config().get_dram_freq());

  // Interface between ramulator and gpgpusim
  from_gpgpusim = new fifo_pipeline<mem_fetch>("fromgpgpusim", 0, 2);
  r_rwq = new fifo_pipeline<mem_fetch>("rwq", 0, 1024);
  r_returnq = new fifo_pipeline<mem_fetch>(
      "dramreturnq", 0,
      m_config->gpgpu_dram_return_queue_size == 0
          ? 1024
          : m_config->gpgpu_dram_return_queue_size);

  // set ramulator output files
  if (!Stats_ramulator::statlist.is_open())
    Stats_ramulator::statlist.output("./output/ramulator.stats");
}

Ramulator::~Ramulator() {}

bool Ramulator::full(bool is_write) const {
  return memory->full(is_write);
}

void Ramulator::cycle() {
  if (!returnq_full()) {
    mem_fetch* mf = r_rwq->pop();
    if (mf) {
      if (mf->get_type() == WRITE_REQUEST) {
        num_write_callback += 1;
      } else {
        num_read_callback += 1;
      }
      mf->set_status(IN_PARTITION_MC_RETURNQ, 
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      if (mf->get_access_type() != L1_WRBK_ACC && mf->get_access_type() != L2_WRBK_ACC) {
        mf->set_reply();
        r_returnq->push(mf);
      } else {
        m_memory_partition_unit->set_done(mf);
        delete mf;
      }
    }
  }

  bool accepted = false;
  while (!from_gpgpusim->empty()) {
    mem_fetch* mf = from_gpgpusim->pop();
    mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(mf->get_addr());
    if (!m_gpu->get_global_memory()->is_valid(page_num)) {
      if (mf->get_type() == READ_REQUEST) {
        num_read_req += 1;
      } else {
        num_write_req += 1;
      }
      r_rwq->push(mf);
    } else {
      if (mf->get_type() == READ_REQUEST) {
        Request req(mf->get_addr(), request::Type::R_READ, read_cb_func, mf->get_sid(), mf);
        accepted = send(req);
        num_read_req += 1;
      } else if (mf->get_type() == WRITE_REQUEST) {        
        if ((unsigned)mf->get_sid() == -1) { // WRITE BACK request
          Request req(mf->get_addr(), request::Type::R_WRITE, write_cb_func, num_cores - 1, mf);
          accepted = send(req);
          num_write_req += 1;
        } else { // NORMAL WRITE from the core
          Request req(mf->get_addr(), request::Type::R_WRITE, write_cb_func, mf->get_sid(), mf);
          accepted = send(req);
          num_write_req += 1;
        }
      }

      assert(accepted);
    }
  }

  // PCI-E validattion request (from HOST to GPU) => write
  if (!m_memory_partition_unit->is_gmmu_pcie_validate_queue_empty()) {
    auto pcie_req = m_memory_partition_unit->front_gmmu_pcie_validate_queue();
    mem_addr_t page_addr = pcie_req.first;
    bool is_last = pcie_req.second;
    pcie_req_list.push_back(page_addr);

    unsigned req_bytes = 256;
    unsigned line_size = m_config->dram_cache_line_size;

    if (is_last) {
      for (mem_addr_t &pg_addr : pcie_req_list) {
        Request req(pg_addr, request::Type::R_WRITE, pcie_cb_func, 0);
        req.num_bytes = req_bytes;
        req.original_num_bytes = req_bytes;
        req.qtype = request::QueueType::PCIE_VALIDATE;
        send_pcie(req);
      }
      pcie_req_list.clear();
    }
    m_memory_partition_unit->pop_gmmu_pcie_validate_queue();
  }

  // PCI-E invalidation request (from GPU to HOST) => read
  if (!m_memory_partition_unit->is_gmmu_pcie_invalidate_queue_empty()) {
    auto pcie_req = m_memory_partition_unit->front_gmmu_pcie_invalidate_queue();
    mem_addr_t page_addr = pcie_req.first;
    bool is_last = pcie_req.second;

    pcie_invalidate_req_list.push_back(page_addr);

    unsigned req_bytes = 256;
    unsigned line_size = m_config->dram_cache_line_size;

    if (is_last) {
      for (mem_addr_t &pg_addr : pcie_invalidate_req_list) {
        //count += 1;
        Request req(pg_addr, request::Type::R_READ, pcie_cb_func, 0);
        req.num_bytes = req_bytes;
        req.original_num_bytes = req_bytes;
        req.qtype = request::QueueType::PCIE_INVALIDATE;
        send_pcie(req);
      }
      pcie_invalidate_req_list.clear();
    }
    m_memory_partition_unit->pop_gmmu_pcie_invalidate_queue();
  }  
  memory->tick();
}

bool Ramulator::send(Request& req) {
  return memory->send(req);
}
void Ramulator::send_pcie(Request& req) {
  memory->send_pcie(req);
}

void Ramulator::push(class mem_fetch* mf) {
  mf->set_status(IN_PARTITION_MC_INTERFACE_QUEUE, 
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  from_gpgpusim->push(mf);

  // this function is related to produce average latency stat
  m_stats->memlatstat_dram_access(mf);
}

bool Ramulator::returnq_full() const {
  return r_returnq->full();
}
mem_fetch* Ramulator::return_queue_top() {
  return r_returnq->top();
}
mem_fetch* Ramulator::return_queue_pop() {
  return r_returnq->pop();
}

void Ramulator::readComplete(Request& req) {
  assert(req.mf != nullptr);
  assert(!r_rwq->full());
  r_rwq->push(req.mf);
}

void Ramulator::writeComplete(Request& req) {
  assert(req.mf != nullptr);
  assert(!r_rwq->full());
  r_rwq->push(req.mf);
}

void Ramulator::pcieComplete(Request& req) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(req.addr);
  // Remove synthetic mem_fetch object
  //delete req.mf;
  // Because the page is distributed across multiple channels,
  // we need to wait until all data of the page are read from each channel.
  // validation
  auto *gmem = m_gpu->get_global_memory();
  if (req.qtype == request::QueueType::PCIE_VALIDATE) {
    bool all_done = 
      gmem->finish_pcie_request(page_num, req.original_num_bytes);
    if (all_done) {
      m_gpu->getGmmu()->push_pcie_gmmu_queue(page_num);
    }
  } else {  // invalidation
    bool all_done =
      gmem->finish_pcie_invalidate_request(page_num, req.original_num_bytes);
    if (all_done) {
      m_gpu->getGmmu()->push_pcie_gmmu_invalidate_queue(page_num);
    }
  }
}

void Ramulator::print(FILE* fp) const {
  std::vector<unsigned> k_uid = m_gpu->get_executed_kernel_uid();
  memory->finish(k_uid[0]);
}
