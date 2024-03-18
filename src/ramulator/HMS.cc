#include "HMS.h"
#include "DRAM.h"

#include <stdio.h>
#include <vector>
#include <functional>
#include <cassert>

using namespace std;
using namespace ramulator;

string HMS::standard_name = "HMS";
int HMS::DRAM_RANK_ID = 0;
int HMS::SCM_RANK_ID = 1;

map<string, enum HMS::DRAM_Org> HMS::dram_org_map = {
    {"HMS_1Gb", HMS::DRAM_Org::HMS_1Gb},
    {"HMS_2Gb", HMS::DRAM_Org::HMS_2Gb},
    {"HMS_4Gb", HMS::DRAM_Org::HMS_4Gb},
};
map<string, enum HMS::DRAM_Speed> HMS::dram_speed_map = {
    {"HMS_1Gbps", HMS::DRAM_Speed::HMS_1Gbps},
    {"HMS_2Gbps", HMS::DRAM_Speed::HMS_2Gbps},
    {"HMS_V100", HMS::DRAM_Speed::HMS_V100},
    {"HMS_A100", HMS::DRAM_Speed::HMS_A100},
};

map<string, enum HMS::SCM_Org> HMS::scm_org_map = {
  {"PCM_SLC", HMS::SCM_Org::PCM_SLC}, 
  {"PCM_MLC", HMS::SCM_Org::PCM_MLC}, 
  {"PCM_TLC", HMS::SCM_Org::PCM_TLC},
  {"VRRAM", HMS::SCM_Org::VRRAM},
};

map<string, enum HMS::SCM_Speed> HMS::scm_speed_map = {
  {"PCM_SLC", HMS::SCM_Speed::PCM_SLC},
  {"PCM_MLC", HMS::SCM_Speed::PCM_MLC},
  {"PCM_TLC", HMS::SCM_Speed::PCM_TLC},
  {"VRRAM", HMS::SCM_Speed::VRRAM},
  {"PCM_SLC_A100", HMS::SCM_Speed::PCM_SLC_A100},
  {"PCM_MLC_A100", HMS::SCM_Speed::PCM_MLC_A100},
  {"PCM_TLC_A100", HMS::SCM_Speed::PCM_TLC_A100},
  {"VRRAM_A100", HMS::SCM_Speed::VRRAM_A100},
};

HMS::HMS(DRAM_Org dram_org, DRAM_Speed dram_speed,
         SCM_Org scm_org, SCM_Speed scm_speed,
         int channel_width, int prefetch_size) :
    dram_org_entry(dram_org_table[int(dram_org)]),
    dram_speed_entry(dram_speed_table[int(dram_speed)]),

    scm_org_entry(scm_org_table[int(scm_org)]),
    scm_speed_entry(scm_speed_table[int(scm_speed)]),
    speed_entry(speed_table[int(dram_speed)]),
    // scm and dram have same read and write latency
    read_latency(dram_speed_entry.nCL + dram_speed_entry.nBL),
    write_latency(dram_speed_entry.nCWL + dram_speed_entry.nBL),
    channel_width(channel_width), prefetch_size(prefetch_size)
{
    // This is the spec of HMS
    assert(channel_width == 128 && "The data bus width must be 128 for HMS");
    assert(prefetch_size == 2 && "The prefetch size must be 2 for HMS");
    init_speed();
    init_prereq();
    init_rowhit(); // SAUGATA: added row hit function
    init_rowopen();
    init_lambda();
    init_dram_timing();
    init_scm_timing();

}

HMS::HMS(const string& dram_org_str, const string& dram_speed_str,
         const string& scm_org_str, const string& scm_speed_str,
         int channel_width, int prefetch_size) :
    HMS(dram_org_map[dram_org_str], dram_speed_map[dram_speed_str],
        scm_org_map[scm_org_str], scm_speed_map[scm_speed_str],
        channel_width, prefetch_size)
{
  assert(dram_org_map.find(dram_org_str) != dram_org_map.end());
  assert(scm_org_map.find(scm_org_str) != scm_org_map.end());
  assert(dram_speed_map.find(dram_speed_str) != dram_speed_map.end());
  assert(scm_speed_map.find(scm_speed_str) != scm_speed_map.end());
}

void HMS::set_channel_number(int channel) {
  dram_org_entry.count[int(Level::Channel)] = channel;
  scm_org_entry.count[int(Level::Channel)] = channel;
}

void HMS::set_rank_number(int rank) {
  dram_org_entry.count[int(Level::Rank)] = rank;
  scm_org_entry.count[int(Level::Rank)] = rank;
}


void HMS::init_speed()
{
    const static int RFC_TABLE[int(DRAM_Speed::MAX)][int(DRAM_Org::MAX)] = {
        {55, 80, 130},
        {110, 160, 260},
        {96, 140, 228}
    };
    const static int REFI1B_TABLE[int(DRAM_Speed::MAX)][int(DRAM_Org::MAX)] = {
        {64, 128, 256},
        {128, 256, 512},
        {112, 224, 449}
    };
    const static int XS_TABLE[int(DRAM_Speed::MAX)][int(DRAM_Org::MAX)] = {
        {60, 85, 135}
    };

    int speed = 0, density = 0;
    switch (dram_speed_entry.rate) {
        case 1000: speed = 0; break;
        case 2000: speed = 1; break;
        case 1752: speed = 2; break;
        //default: assert(false);
        default: break;
    };
    switch (dram_org_entry.size >> 10){
        case 1: density = 0; break;
        case 2: density = 1; break;
        case 4: density = 2; break;
        //default: assert(false);
        default: break;

    }
    dram_speed_entry.nRFC = RFC_TABLE[speed][density];
    dram_speed_entry.nREFI1B = REFI1B_TABLE[speed][density];
    dram_speed_entry.nXS = XS_TABLE[speed][density];

    // max (TR command cycles, dram CCDL, scm CCDL)
    nTR_CCD = std::max({2, dram_speed_entry.nCCDL, scm_speed_entry.nCCDL});
}


void HMS::init_prereq()
{
    // RD
    prereq[int(Level::Rank)][int(Command::RD)] = [] (DRAM<HMS>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::MAX;
            case int(State::ActPowerDown): return Command::PDX;
            case int(State::PrePowerDown): return Command::PDX;
            case int(State::SelfRefresh): return Command::SRX;
            default: assert(false);
        }};
    prereq[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HMS>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return Command::ACT;
            case int(State::Opened):
                if (node->row_state.find(id) != node->row_state.end())
                    return cmd;
                else return Command::PRE;
            default: assert(false);
        }};

    // WR
    prereq[int(Level::Rank)][int(Command::WR)] = prereq[int(Level::Rank)][int(Command::RD)];
    prereq[int(Level::Bank)][int(Command::WR)] = prereq[int(Level::Bank)][int(Command::RD)];

    // REF
    prereq[int(Level::Rank)][int(Command::REF)] = [] (DRAM<HMS>* node, Command cmd, int id) {
        for (auto bg : node->children)

            for (auto bank: bg->children) {
                if (bank->state == State::Closed)
                    continue;
                return Command::PREA;
            }
        return Command::REF;};

    // REFSB
    prereq[int(Level::Bank)][int(Command::REFSB)] = [] (DRAM<HMS>* node, Command cmd, int id) {
        if (node->state == State::Closed) return Command::REFSB;
        return Command::PRE;};

    // PD
    prereq[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<HMS>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::PowerUp): return Command::PDE;
            case int(State::ActPowerDown): return Command::PDE;
            case int(State::PrePowerDown): return Command::PDE;
            case int(State::SelfRefresh): return Command::SRX;
            default: assert(false);
        }};

    // SR
    prereq[int(Level::Rank)][int(Command::SRE)] = [] (DRAM<HMS>* node, Command cmd, int id) {
      switch (int(node->state)) {
          case int(State::PowerUp): return Command::SRE;
          case int(State::ActPowerDown): return Command::PDX;
          case int(State::PrePowerDown): return Command::PDX;
          case int(State::SelfRefresh): return Command::SRE;
          default: assert(false);
      }
    };
}

// SAUGATA: added row hit check functions to see if the desired location is currently open
void HMS::init_rowhit()
{
    // RD
    rowhit[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HMS>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return false;
            case int(State::Opened):
                if (node->row_state.find(id) != node->row_state.end())
                    return true;
                return false;
            default: assert(false);
        }};

    // WR
    rowhit[int(Level::Bank)][int(Command::WR)] = rowhit[int(Level::Bank)][int(Command::RD)];
}

void HMS::init_rowopen()
{
    // RD
    rowopen[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HMS>* node, Command cmd, int id) {
        switch (int(node->state)) {
            case int(State::Closed): return false;
            case int(State::Opened): return true;
            default: assert(false);
        }};

    // WR
    rowopen[int(Level::Bank)][int(Command::WR)] = rowopen[int(Level::Bank)][int(Command::RD)];
}

void HMS::init_lambda()
{
    lambda[int(Level::Bank)][int(Command::ACT)] = [] (DRAM<HMS>* node, int id) {
        node->state = State::Opened;
        node->row_state[id] = State::Opened;};
    lambda[int(Level::Bank)][int(Command::PRE)] = [] (DRAM<HMS>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Rank)][int(Command::PREA)] = [] (DRAM<HMS>* node, int id) {
        for (auto bg : node->children)
            for (auto bank : bg->children) {
                bank->state = State::Closed;
                bank->row_state.clear();
            }};
    lambda[int(Level::Rank)][int(Command::REF)] = [] (DRAM<HMS>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::RD)] = [] (DRAM<HMS>* node, int id) {};
    lambda[int(Level::Bank)][int(Command::WR)] = [] (DRAM<HMS>* node, int id) {};

    lambda[int(Level::Bank)][int(Command::RDA)] = [] (DRAM<HMS>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Bank)][int(Command::WRA)] = [] (DRAM<HMS>* node, int id) {
        node->state = State::Closed;
        node->row_state.clear();};
    lambda[int(Level::Rank)][int(Command::PDE)] = [] (DRAM<HMS>* node, int id) {
        for (auto bg : node->children)
            for (auto bank : bg->children) {
                if (bank->state == State::Closed)
                    continue;
                node->state = State::ActPowerDown;
                return;
            }
        node->state = State::PrePowerDown;};
    lambda[int(Level::Rank)][int(Command::PDX)] = [] (DRAM<HMS>* node, int id) {
        node->state = State::PowerUp;};
    lambda[int(Level::Rank)][int(Command::SRE)] = [] (DRAM<HMS>* node, int id) {
        node->state = State::SelfRefresh;};
    lambda[int(Level::Rank)][int(Command::SRX)] = [] (DRAM<HMS>* node, int id) {
        node->state = State::PowerUp;};
}

void HMS::init_dram_timing()
{
  DRAM_SpeedEntry& s = dram_speed_entry;
  vector<TimingEntry> *t;

  /*** Channel ***/
  t = dram_timing[int(Level::Channel)];

  // CAS <-> CAS
  t[int(Command::RD)].push_back({Command::RD, 1, s.nBL});
  t[int(Command::RD)].push_back({Command::RDA, 1, s.nBL});
  t[int(Command::RDA)].push_back({Command::RD, 1, s.nBL});
  t[int(Command::RDA)].push_back({Command::RDA, 1, s.nBL});
  t[int(Command::WR)].push_back({Command::WR, 1, s.nBL});
  t[int(Command::WR)].push_back({Command::WRA, 1, s.nBL});
  t[int(Command::WRA)].push_back({Command::WR, 1, s.nBL});
  t[int(Command::WRA)].push_back({Command::WRA, 1, s.nBL});

  /*** Rank ***/
  t = dram_timing[int(Level::Rank)];

  // CAS <-> CAS
  t[int(Command::RD)].push_back({Command::RD, 1, s.nCCDS});
  t[int(Command::RD)].push_back({Command::RDA, 1, s.nCCDS});
  t[int(Command::RDA)].push_back({Command::RD, 1, s.nCCDS});
  t[int(Command::RDA)].push_back({Command::RDA, 1, s.nCCDS});
  t[int(Command::WR)].push_back({Command::WR, 1, s.nCCDS});
  t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCDS});
  t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCDS});
  t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCDS});
  t[int(Command::RD)].push_back({Command::WR, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::RD)].push_back({Command::WRA, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::RDA)].push_back({Command::WR, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::RDA)].push_back({Command::WRA, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRS});
  t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRS});
  t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRS});
  t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRS});

  // CAS <-> CAS (between sibling ranks)
  t[int(Command::RD)].push_back({Command::RD, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::RDA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::RD, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::RDA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::WR, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::WRA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::WR, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::WRA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::WR, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::RD)].push_back({Command::WRA, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::RDA)].push_back({Command::WR, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::RDA)].push_back({Command::WRA, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});
  t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});
  t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});
  t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});

  t[int(Command::RD)].push_back({Command::PREA, 1, s.nRTPL});
  t[int(Command::WR)].push_back({Command::PREA, 1, s.nCWL + s.nBL + s.nWR});

  // CAS <-> PD
  t[int(Command::RD)].push_back({Command::PDE, 1, s.nCL + s.nBL + 1});
  t[int(Command::RDA)].push_back({Command::PDE, 1, s.nCL + s.nBL + 1});
  t[int(Command::WR)].push_back({Command::PDE, 1, s.nCWL + s.nBL + s.nWR});
  t[int(Command::WRA)].push_back({Command::PDE, 1, s.nCWL + s.nBL + s.nWR + 1}); // +1 for pre
  t[int(Command::PDX)].push_back({Command::RD, 1, s.nXP});
  t[int(Command::PDX)].push_back({Command::RDA, 1, s.nXP});
  t[int(Command::PDX)].push_back({Command::WR, 1, s.nXP});
  t[int(Command::PDX)].push_back({Command::WRA, 1, s.nXP});

  // CAS <-> SR: none (all banks have to be precharged)

  // RAS <-> RAS
  t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRRDS});
  t[int(Command::ACT)].push_back({Command::ACT, 4, s.nFAW});
  t[int(Command::ACT)].push_back({Command::PREA, 1, s.nRAS});
  t[int(Command::PREA)].push_back({Command::ACT, 1, s.nRP});

  // RAS <-> REF
  t[int(Command::PRE)].push_back({Command::REF, 1, s.nRP});
  t[int(Command::PREA)].push_back({Command::REF, 1, s.nRP});
  t[int(Command::REF)].push_back({Command::ACT, 1, s.nRFC});

  // RAS <-> PD
  t[int(Command::ACT)].push_back({Command::PDE, 1, 1});
  t[int(Command::PDX)].push_back({Command::ACT, 1, s.nXP});
  t[int(Command::PDX)].push_back({Command::PRE, 1, s.nXP});
  t[int(Command::PDX)].push_back({Command::PREA, 1, s.nXP});

  // RAS <-> SR
  t[int(Command::PRE)].push_back({Command::SRE, 1, s.nRP});
  t[int(Command::PREA)].push_back({Command::SRE, 1, s.nRP});
  t[int(Command::SRX)].push_back({Command::ACT, 1, s.nXS});

  // REF <-> REF
  t[int(Command::REF)].push_back({Command::REF, 1, s.nRFC});

  // REF <-> PD
  t[int(Command::REF)].push_back({Command::PDE, 1, 1});
  t[int(Command::PDX)].push_back({Command::REF, 1, s.nXP});
  t[int(Command::PDX)].push_back({Command::SRE, 1, s.nXP});
  t[int(Command::SRX)].push_back({Command::PDE, 1, s.nXS});

  // REF <-> SR
  t[int(Command::SRX)].push_back({Command::REF, 1, s.nXS});

  // PD <-> PD
  t[int(Command::PDE)].push_back({Command::PDX, 1, s.nPD});
  t[int(Command::PDX)].push_back({Command::PDE, 1, s.nXP});

  // PD <-> SR
  t[int(Command::PDX)].push_back({Command::SRE, 1, s.nXP});
  t[int(Command::SRX)].push_back({Command::PDE, 1, s.nXS});

  // SR <-> SR
  t[int(Command::SRE)].push_back({Command::SRX, 1, s.nCKESR});
  t[int(Command::SRX)].push_back({Command::SRE, 1, s.nXS});

  /*** Bank Group ***/
  t = dram_timing[int(Level::BankGroup)];
  // CAS <-> CAS
  t[int(Command::RD)].push_back({Command::RD, 1, s.nCCDL});
  t[int(Command::RD)].push_back({Command::RDA, 1, s.nCCDL});
  t[int(Command::RDA)].push_back({Command::RD, 1, s.nCCDL});
  t[int(Command::RDA)].push_back({Command::RDA, 1, s.nCCDL});
  t[int(Command::WR)].push_back({Command::WR, 1, s.nCCDL});
  t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCDL});
  t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCDL});
  t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCDL});
  t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRL});
  t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRL});
  t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRL});
  t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRL});


  // RAS <-> RAS
  t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRRDL});

  /*** Bank ***/
  t = dram_timing[int(Level::Bank)];

  // CAS <-> RAS
  t[int(Command::ACT)].push_back({Command::RD, 1, s.nRCDR});
  t[int(Command::ACT)].push_back({Command::RDA, 1, s.nRCDR});
  t[int(Command::ACT)].push_back({Command::WR, 1, s.nRCDW});
  t[int(Command::ACT)].push_back({Command::WRA, 1, s.nRCDW});

  t[int(Command::RD)].push_back({Command::PRE, 1, s.nRTPL}); // 3
  t[int(Command::WR)].push_back({Command::PRE, 1, s.nCWL + s.nBL + s.nWR}); // 2 + 1 + 16

  t[int(Command::RDA)].push_back({Command::ACT, 1, s.nRTPL + s.nRP});
  t[int(Command::WRA)].push_back({Command::ACT, 1, s.nCWL + s.nBL + s.nWR + s.nRP});

  // RAS <-> RAS
  t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRC});
  t[int(Command::ACT)].push_back({Command::PRE, 1, s.nRAS});
  t[int(Command::PRE)].push_back({Command::ACT, 1, s.nRP}); // 14

  // REFSB
  t[int(Command::PRE)].push_back({Command::REFSB, 1, s.nRP});
  t[int(Command::REFSB)].push_back({Command::REFSB, 1, s.nRFC});
  t[int(Command::REFSB)].push_back({Command::ACT, 1, s.nRFC});
}

void HMS::init_scm_timing() {
  SCM_SpeedEntry& s = scm_speed_entry;
  vector<TimingEntry> *t;

  /* Rank */
  t = scm_timing[int(Level::Rank)];

  // CAS <-> CAS
  t[int(Command::RD)].push_back({Command::RD, 1, s.nCCDS});
  t[int(Command::RD)].push_back({Command::RDA, 1, s.nCCDS});
  t[int(Command::RDA)].push_back({Command::RD, 1, s.nCCDS});
  t[int(Command::RDA)].push_back({Command::RDA, 1, s.nCCDS});
  t[int(Command::WR)].push_back({Command::WR, 1, s.nCCDS});
  t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCDS});
  t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCDS});
  t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCDS});
  t[int(Command::RD)].push_back({Command::WR, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::RD)].push_back({Command::WRA, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::RDA)].push_back({Command::WR, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::RDA)].push_back({Command::WRA, 1, s.nCL + s.nCCDS + 2 - s.nCWL});
  t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRS});
  t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRS});
  t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRS});
  t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRS});

  // CAS <-> CAS (between sibling ranks)
  t[int(Command::RD)].push_back({Command::RD, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::RDA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::RD, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::RDA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::WR, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::WRA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::WR, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RDA)].push_back({Command::WRA, 1, s.nBL + s.nRTRS, true});
  t[int(Command::RD)].push_back({Command::WR, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::RD)].push_back({Command::WRA, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::RDA)].push_back({Command::WR, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::RDA)].push_back({Command::WRA, 1, s.nCL + s.nBL + s.nRTRS - s.nCWL, true});
  t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});
  t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});
  t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});
  t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nRTRS - s.nCL, true});
  
  t[int(Command::RD)].push_back({Command::PREA, 1, s.nRTPL});
  t[int(Command::WR)].push_back({Command::PREA, 1, s.nCWL + s.nBL + s.nWR});
  
  // RAS <-> RAS
  t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRRDS});
  t[int(Command::ACT)].push_back({Command::ACT, 4, s.nFAW});
  t[int(Command::ACT)].push_back({Command::PREA, 1, s.nRAS});
  t[int(Command::PREA)].push_back({Command::ACT, 1, s.nRP});

  /* Bank Group */
  t = scm_timing[int(Level::BankGroup)];

  // CAS <-> CAS
  t[int(Command::RD)].push_back({Command::RD, 1, s.nCCDL});
  t[int(Command::RD)].push_back({Command::RDA, 1, s.nCCDL});
  t[int(Command::RDA)].push_back({Command::RD, 1, s.nCCDL});
  t[int(Command::RDA)].push_back({Command::RDA, 1, s.nCCDL});
  t[int(Command::WR)].push_back({Command::WR, 1, s.nCCDL});
  t[int(Command::WR)].push_back({Command::WRA, 1, s.nCCDL});
  t[int(Command::WRA)].push_back({Command::WR, 1, s.nCCDL});
  t[int(Command::WRA)].push_back({Command::WRA, 1, s.nCCDL});
  t[int(Command::WR)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRL});
  t[int(Command::WR)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRL});
  t[int(Command::WRA)].push_back({Command::RD, 1, s.nCWL + s.nBL + s.nWTRL});
  t[int(Command::WRA)].push_back({Command::RDA, 1, s.nCWL + s.nBL + s.nWTRL});

  // RAS <-> RAS
  t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRRDL});

  /* Bank */
  t = scm_timing[int(Level::Bank)];

  // CAS <-> RAS
  t[int(Command::ACT)].push_back({Command::RD, 1, s.nRCDR});
  t[int(Command::ACT)].push_back({Command::RDA, 1, s.nRCDR});
  t[int(Command::ACT)].push_back({Command::WR, 1, s.nRCDW});
  t[int(Command::ACT)].push_back({Command::WRA, 1, s.nRCDW});

  t[int(Command::RD)].push_back({Command::PRE, 1, s.nRTPL}); // 3
  t[int(Command::WR)].push_back({Command::PRE, 1, s.nCWL + s.nBL + s.nWR}); // 2 + 1 + 16

  t[int(Command::RDA)].push_back({Command::ACT, 1, s.nRTPL + s.nRP});
  t[int(Command::WRA)].push_back({Command::ACT, 1, s.nCWL + s.nBL + s.nWR + s.nRP});

  t[int(Command::ACT)].push_back({Command::ACT, 1, s.nRC});
  t[int(Command::ACT)].push_back({Command::PRE, 1, s.nRAS});
  t[int(Command::PRE)].push_back({Command::ACT, 1, s.nRP});
}
