#include "MemoryFactory.h"
#include "HBM.h"
#include "HMS.h"
#include "HMSController.h"

using namespace ramulator;

namespace ramulator
{

template <>
void MemoryFactory<HBM>::validate(int channels, int ranks, const Config& configs) {
  assert(channels == 1 && "HBM comes with 1 channel for each ramulator object");
}

template<>
Memory<HMS>* MemoryFactory<HMS>::populate_memory(const Config& configs, 
                                                 HMS *spec,
                                                 int channels, int ranks, 
                                                 int channel_id, //string app_name,
                                                 const memory_config *m_config,
                                                 class memory_partition_unit* mp,
                                                 class gpgpu_sim* gpu) {
  int& dram_default_ranks = spec->dram_org_entry.count[int(HMS::Level::Rank)];
  int& dram_default_channels = spec->dram_org_entry.count[int(HMS::Level::Channel)];
  if (dram_default_channels == 0) dram_default_channels = channels;
  if (dram_default_ranks == 0) dram_default_ranks = ranks;

  int& scm_default_ranks = spec->scm_org_entry.count[int(HMS::Level::Rank)];
  int& scm_default_channels = spec->scm_org_entry.count[int(HMS::Level::Channel)];
  if (scm_default_channels == 0) scm_default_channels = channels;
  if (scm_default_ranks == 0) scm_default_ranks = ranks;

  vector<Controller<HMS>*> ctrls;
  int end_of_dram_ch = m_config->m_n_mem;
  assert(end_of_dram_ch >= 0 && end_of_dram_ch <= m_config->m_n_mem);
  for (int c = 0; c < channels; c++) { //channels is always 1
    HMS::Type type;
    if (0 <= mp->get_mpid() && mp->get_mpid() < end_of_dram_ch) {
      std::cout<<"This channel ("<<mp->get_mpid()<<") is SCMDRAM"<<std::endl;
      type = HMS::Type::SCMDRAM;
    }
    else {
      std::cout<<"This channel ("<<mp->get_mpid()<<") is SCM"<<std::endl;
      type = HMS::Type::SCM;
    }
    DRAM<HMS>* channel = new DRAM<HMS>(spec, HMS::Level::Channel, NULL, c, type); //mp->get_mpid() == channel number
    // Here, the channel->id is used for indexing controller in DRAM object
    channel->id = c;
    channel->regStats("");
    ctrls.push_back(
      new HMSController(configs, channel, channel_id, m_config, mp, gpu));
  }
  // only one controller exists per channel
  assert(ctrls.size() == 1);
  return new Memory<HMS>(configs, ctrls, m_config);
}

template <>
MemoryBase* MemoryFactory<HMS>::create(const Config& configs, 
                                       int cacheline, int channel_id,
                                       const memory_config* m_config,
                                       class memory_partition_unit* mp,
                                       class gpgpu_sim* gpu) {
  int channels = stoi(configs["channels"], NULL, 0);
  int ranks = stoi(configs["ranks"], NULL, 0);
  const string& std_name = configs["standard"];

  assert(std_name == "HMS");
  assert(channels == 1 && "Each memory handles one channel");
  assert(ranks == 2 && "The number of HMS ranks must be 2");
  assert(configs.contains("DRAM_org") && configs.contains("DRAM_speed") &&
         "need to specify dram spec");
  assert(configs.contains("PCM_org") && configs.contains("PCM_speed") &&
         "need to specify scm spec");

  const string& DRAM_org_name = configs["DRAM_org"];
  const string& DRAM_speed_name = configs["DRAM_speed"];
  const string& PCM_org_name = configs["PCM_org"];
  const string& PCM_speed_name = configs["PCM_speed"];

  // we assume that current HMS consists of two different types of memory, DRAM and PCM
  HMS* spec = new HMS(DRAM_org_name, DRAM_speed_name, 
                      PCM_org_name, PCM_speed_name,
                      m_config->busW * 8, m_config->BL);

  return (MemoryBase *)populate_memory(configs, spec, 
                                       channels, ranks, 
                                       channel_id, m_config,
                                       mp, gpu);
}

}  // end namespace

// This function can be used by autoconf AC_CHECK_LIB since
// apparently it can't detect C++ functions.
// Basically just an entry in the symbol table
extern "C"
{
    void libramulator_is_present(void)
    {
        ;
    }
}
