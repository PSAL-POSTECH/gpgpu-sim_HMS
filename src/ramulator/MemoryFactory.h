#ifndef __MEMORY_FACTORY_H
#define __MEMORY_FACTORY_H

#include <map>
#include <string>
#include <cassert>

#include "Config.h"
#include "Memory.h"

#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/l2cache.h"

using namespace std;

namespace ramulator
{

template <typename T>
class MemoryFactory {
public:
    static void extend_channel_width(T* spec, int cacheline)
    {
        int channel_unit = spec->prefetch_size * spec->channel_width / 8;
        int gang_number = cacheline / channel_unit;
        
        assert(gang_number >= 1 && 
            "cacheline size must be greater or equal to minimum channel width");
        
        assert(cacheline == gang_number * channel_unit &&
            "cacheline size must be a multiple of minimum channel width");
        
        spec->channel_width *= gang_number;
    }

    static Memory<T> *populate_memory(const Config& configs, T *spec, 
                                      int channels, int ranks,
                                      int channel_id,
                                      const memory_config* m_config,
                                      class memory_partition_unit* mp,
                                      class gpgpu_sim* gpu) {
        int& default_ranks = spec->org_entry.count[int(T::Level::Rank)];
        int& default_channels = spec->org_entry.count[int(T::Level::Channel)];

        if (default_channels == 0) default_channels = channels;
        if (default_ranks == 0) default_ranks = ranks;

        assert(m_config->gpu_n_mem_per_ctrlr == 1);
        assert(channel_id < m_config->m_n_mem);

        vector<Controller<T> *> ctrls;
        for (int c = 0; c < channels; c++){
          DRAM<T>* channel = new DRAM<T>(spec, T::Level::Channel, NULL, c, T::Type::SINGLE);//jeongmin - add type of current DRAM node
          // Here, the channel->id is used for indexing controller in DRAM object
          channel->id = c;
          channel->regStats("");
          ctrls.push_back(
            new Controller<T>(configs, channel, channel_id, m_config, mp, gpu));
        }
        // only one controller exists per channel
        assert(ctrls.size() == 1);
        return new Memory<T>(configs, ctrls, m_config);
    }

    static void validate(int channels, int ranks, const Config& configs) {
        assert(channels > 0 && ranks > 0);
    }

    static MemoryBase *create(const Config& configs, int cacheline, 
                              int channel_id, const memory_config* m_config,
                              class memory_partition_unit* mp,
                              class gpgpu_sim* gpu)
    {
        int channels = stoi(configs["channels"], NULL, 0);
        int ranks = stoi(configs["ranks"], NULL, 0);
        
        assert(channels == 1 && "Each memory handles one channel");
        validate(channels, ranks, configs);

        const string& org_name = configs["org"];
        const string& speed_name = configs["speed"];

        T *spec = new T(org_name, speed_name,
                        m_config->busW * 8, m_config->BL);

        //extend_channel_width(spec, cacheline);

        return (MemoryBase *)populate_memory(configs, spec, 
                                             channels, ranks, 
                                             channel_id, m_config,
                                             mp, gpu);
    }
};

template<>
Memory<HMS>* MemoryFactory<HMS>::populate_memory(const Config& configs, 
                                                 HMS *spec,
                                                 int channels, int ranks, 
                                                 int channel_id,
                                                 const memory_config* m_config,
                                                 class memory_partition_unit* mp,
                                                 class gpgpu_sim* gpu);
template<>
MemoryBase* MemoryFactory<HMS>::create(const Config& configs, 
                                       int cacheline, int channel_id,
                                       const memory_config* m_config,
                                       class memory_partition_unit* mp,
                                       class gpgpu_sim* gpu);
//template <>
//MemoryBase *MemoryFactory<WideIO2>::create(const Config& configs, int cacheline);
//template <>
//MemoryBase *MemoryFactory<SALP>::create(const Config& configs, int cacheline);

} /*namespace ramulator*/

#endif /*__MEMORY_FACTORY_H*/
