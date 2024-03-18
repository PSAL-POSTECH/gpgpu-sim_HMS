#include "Memory.h"

#include "HMS.h"

namespace ramulator {

template<>
Memory<HMS, Controller>::Memory(const Config& configs, vector<Controller<HMS>*> ctrls,
                                const memory_config* m_config)
    : ctrls(ctrls),
      spec(ctrls[0]->channel->spec),
      dram_addr_bits(int(HMS::Level::MAX)),
      scm_addr_bits(int(HMS::Level::MAX)),
      m_config(m_config) {
    // make sure 2^N channels/ranks
  int *dram_sz = spec->dram_org_entry.count;
  int *scm_sz = spec->scm_org_entry.count;
  assert((dram_sz[0] & (dram_sz[0] - 1)) == 0);
  assert((dram_sz[1] & (dram_sz[1] - 1)) == 0);
  assert((scm_sz[0] & (scm_sz[0] - 1)) == 0);
  assert((scm_sz[1] & (scm_sz[1] - 1)) == 0);
  // validate size of one transaction
  int tx = (spec->prefetch_size * spec->channel_width / 8);
  tx_bits = calc_log2(tx);
  assert((1<<tx_bits) == tx);
  
  // Parsing mapping file and initialize mapping table
  use_mapping_file = false;
  dump_mapping = false;

  dram_max_address = spec->channel_width / 8;
  scm_max_address = spec->channel_width / 8;

  // DRAM addr bits
  for (unsigned int lev = 0; lev < dram_addr_bits.size(); lev++) {
    dram_addr_bits[lev] = calc_log2(dram_sz[lev]);
      dram_max_address *= dram_sz[lev];
  }
  for (unsigned int lev = 0; lev < scm_addr_bits.size(); lev++) {
    scm_addr_bits[lev] = calc_log2(scm_sz[lev]);
      scm_max_address *= scm_sz[lev];
  }

  dram_addr_bits[int(HMS::Level::MAX) - 1] -= calc_log2(spec->prefetch_size);
  scm_addr_bits[int(HMS::Level::MAX) - 1] -= calc_log2(spec->prefetch_size);

  // Initiating translation
  if (configs.contains("translation")) {
    translation = name_to_translation[configs["translation"]];
  }
  if (translation != Translation::None) {
    // construct a list of available pages
    free_physical_pages_remaining = max_address >> 12;

    free_physical_pages.resize(free_physical_pages_remaining, -1);
  }
}
template<>
void Memory<HMS, Controller>::send_pcie(Request& req) {
  addrdec_t raw_addr;
  m_config->m_address_mapping.addrdec_tlx(req.addr, &raw_addr);
  assert(raw_addr.chip == ctrls[0]->ch_id &&
         "request arrives at the wrong memory partition");
  req.addr_vec.resize(int(HMS::Level::MAX));
  req.addr_vec[int(HMS::Level::Channel)] = 0;
  req.addr_vec[int(HMS::Level::BankGroup)] = raw_addr.bk & (4 - 1);
  req.addr_vec[int(HMS::Level::Bank)] = raw_addr.bk >> 2;
  assert(req.addr_vec[int(HMS::Level::BankGroup)] < 4 &&
         req.addr_vec[int(HMS::Level::Bank)] < 4);
  req.addr_vec[int(HMS::Level::Rank)] = raw_addr.rank;
  req.addr_vec[int(HMS::Level::Row)] = raw_addr.row;
  req.addr_vec[int(HMS::Level::Column)] = raw_addr.col >> 5;
  ctrls[0]->enqueue_pcie(req);
}

template<>
bool Memory<HMS, Controller>::send(Request& req) {
  req.addr_vec.resize(int(HMS::Level::MAX));
  assert(req.mf->get_tlx_addr().chip == ctrls[0]->ch_id &&
         "request arrives at the wrong memory partition");
  req.addr_vec[int(HMS::Level::Channel)] = 0;
  req.addr_vec[int(HMS::Level::BankGroup)] = req.mf->get_tlx_addr().bk & (4 - 1);
  req.addr_vec[int(HMS::Level::Bank)] = req.mf->get_tlx_addr().bk >> 2;
  assert(req.addr_vec[int(HMS::Level::BankGroup)] < 4 &&
         req.addr_vec[int(HMS::Level::Bank)] < 4);
  req.addr_vec[int(HMS::Level::Rank)] = req.mf->get_tlx_addr().rank;
  req.addr_vec[int(HMS::Level::Row)] = req.mf->get_tlx_addr().row;
  req.addr_vec[int(HMS::Level::Column)] = req.mf->get_tlx_addr().col >> 5;

  if(ctrls[0]->enqueue(req)) {
      return true;
  }
  return false;
}

template<>
void Memory<HMS, Controller>::finish(unsigned kernel_id) {
  for (auto ctrl : ctrls) {
    //long read_req = long(incoming_read_reqs_per_channel[ctrl->channel->id].value());
    long read_req = 0;
    long dram_cycles = num_dram_cycles.value();
    ctrl->finish(read_req, dram_cycles, kernel_id);
  }
}

} /*namespace ramulator*/
