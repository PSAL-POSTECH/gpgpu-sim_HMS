#ifndef __HBM_H
#define __HBM_H

//#include "DRAM.h"
#include "Request.h"
#include <vector>
#include <functional>

using namespace std;

namespace ramulator
{
template <typename T>
class DRAM;

class HBM
{
public:
    static string standard_name;
    enum class Org;
    enum class Speed;
    HBM(Org org, Speed speed, int channel_width, int prefetch_size);
    HBM(const string& org_str, const string& speed_str,
        int channel_width, int prefetch_size);

    static map<string, enum Org> org_map;
    static map<string, enum Speed> speed_map;

    /* Level */
    enum class Level : int
    {
        Channel, Rank, BankGroup, Bank, Row, Column, MAX
    };

    static std::string level_str [int(Level::MAX)];

    /* Command */
    enum class Command : int
    {
        ACT, PRE,   PREA,
        RD,  WR,    RDA, WRA,
        REF, REFSB, PDE, PDX,  SRE, SRX,
        MAX
    };

    enum class Type : int
    {
        SINGLE, MAX //jeongmin - HBM has only DRAM type
    };

    // REFSB and REF is not compatible, choose one or the other.
    // REFSB can be issued to banks in any order, as long as REFI1B
    // is satisfied for all banks

    string command_name[int(Command::MAX)] = {
        "ACT", "PRE",   "PREA",
        "RD",  "WR",    "RDA",  "WRA",
        "REF", "REFSB", "PDE",  "PDX",  "SRE", "SRX"
    };

    Level scope[int(Command::MAX)] = {
        Level::Row,    Level::Bank,   Level::Rank,
        Level::Column, Level::Column, Level::Column, Level::Column,
        Level::Rank,   Level::Bank,   Level::Rank,   Level::Rank,   Level::Rank,   Level::Rank
    };

    bool is_opening(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::ACT):
                return true;
            default:
                return false;
        }
    }

    bool is_accessing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::RD):
            case int(Command::WR):
            case int(Command::RDA):
            case int(Command::WRA):
                return true;
            default:
                return false;
        }
    }

    bool is_closing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::RDA):
            case int(Command::WRA):
            case int(Command::PRE):
            case int(Command::PREA):
                return true;
            default:
                return false;
        }
    }

    bool is_refreshing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::REF):
            case int(Command::REFSB):
                return true;
            default:
                return false;
        }
    }

    /* State */
    enum class State : int
    {
        Opened, Closed, PowerUp, ActPowerDown, PrePowerDown, SelfRefresh, MAX
    } start[int(Level::MAX)] = {
        State::MAX, State::PowerUp, State::MAX, State::Closed, State::Closed, State::MAX
    };

    /* Translate */
    Command translate[int(request::Type::MAX)] = {
        Command::RD,  Command::WR,
        Command::REF, Command::PDE, Command::SRE
    };

    /* Prereq */
    function<Command(DRAM<HBM>*, Command cmd, int)> prereq[int(Level::MAX)][int(Command::MAX)];

    // SAUGATA: added function object container for row hit status
    /* Row hit */
    function<bool(DRAM<HBM>*, Command cmd, int)> rowhit[int(Level::MAX)][int(Command::MAX)];
    function<bool(DRAM<HBM>*, Command cmd, int)> rowopen[int(Level::MAX)][int(Command::MAX)];

    /* Timing */
    struct TimingEntry
    {
        Command cmd;
        int dist;
        int val;
        bool sibling;
    };
    vector<TimingEntry> timing[int(Level::MAX)][int(Command::MAX)];

    /* Lambda */
    function<void(DRAM<HBM>*, int)> lambda[int(Level::MAX)][int(Command::MAX)];

    /* Organization */
    enum class Org : int
    { // per channel density here. Each stack comes with 8 channels
      // TITAN V has 3 stacks (total 12GBs)
      // each stack (4GBs) owns four dram dies with 2 channels each
      // each dram die owns 1GB -> 1GB/2 channels = 4Gb (0.5 GB) per channel
        HBM_1Gb,
        HBM_2Gb,
        HBM_4Gb, // we will use this , 4 bank groups, 4 banks, page size: 2KB
        HBM_1Gb_2rank,
        HBM_2Gb_2rank,
        HBM_4Gb_2rank, // we will use this , 4 bank groups, 4 banks, page size: 2KB
        MAX
    };

    struct OrgEntry {
        int size;
        int dq;
        int count[int(Level::MAX)];
    } org_table[int(Org::MAX)] = {
        {1<<10, 128, {0, 0, 4, 2, 1<<13, 1<<(6+1)}},
        {2<<10, 128, {0, 0, 4, 2, 1<<14, 1<<(6+1)}},
        {4<<10, 128, {0, 0, 4, 4, 1<<14, 1<<(6+1)}},
        {1<<10, 128, {0, 2, 4, 2, 1<<13, 1<<(6+1)}},
        {2<<10, 128, {0, 2, 4, 2, 1<<14, 1<<(6+1)}},
        {4<<10, 128, {0, 2, 4, 4, 1<<14, 1<<(6+1)}},
    }, org_entry;

    struct AddrMask {
      string mask;
    } addr_mask_list[int(Org::MAX)] = {
      {"00000000.00000000.00000000.00000000.00000RRR.RRRRRRRR.RRBGGCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RRBGGCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.000RRRRR.RRRRRRRR.RBBGGCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBGGHCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.000RRRRR.RRRRRRRR.RBGGHCCC.CCCSSSSS"},
      {"00000000.00000000.00000000.00000000.00RRRRRR.RRRRRRRR.BBGGHCCC.CCCSSSSS"},
    }, addr_mask;


    void set_channel_number(int channel);
    void set_rank_number(int rank);

    /* Speed */
    enum class Speed : int
    {
        HBM_1Gbps,
        HBM_2Gbps,
        HBM_V100,
        HBM_A100,
        MAX
    };

    //int prefetch_size = 4; // burst length could be 2 and 4 (choose 4 here), 2n prefetch
    int prefetch_size = 2; // burst length could be 2 and 4 (choose 4 here), 2n prefetch
    int channel_width = 128;  // 16 bytes channel width

    struct SpeedEntry {
        // JESD 325 (page 68) describes that RAS and RP are decided by vendors.
        // However, RC = RAS + RP must be achieved.
        // RFC: 1Gb (110ns), 2Gb (160ns), 4Gb (260ns)
        //
        // JESD 325 (page 72) describes refresh timings
        // REFI: 1Gb, 2Gb, 4Gb (3.9us -> 3900ns)
        int rate;
        double freq, tCK;
        int nRTRS;
        int nBL, nCCDS, nCCDL;
        int nCL, nRCDR, nRCDW, nRP, nCWL;
        int nRAS, nRC;
        int nRTP, nRTPL, nWTRS, nWTRL, nWR;
        int nRRDS, nRRDL, nFAW;
        int nRFC, nREFI, nREFI1B;
        int nPD, nXP;
        int nCKESR, nXS;

        int nRCD_TR_RD;
    } speed_table[int(Speed::MAX)] = {
        {1000, 
         500, 2.0, 
         0,
         2, 2, 3, 
         7, 7, 6, 7, 4, 
         17, 24, 
         7, 2, 4, 8, 
         4, 5, 20, 
         0, 1950, 0, 
         5, 5, 
         5, 0,
         0},

        {2000, 
         1000, 1.0,
         0,
         1, 1, 2,
         14, 14, 14, 14, 2,
         33, 47,
         3, 4, 3, 8, 16,
         4, 6, 16,
         0, 3900, 0,
         0, 0,
         0, 0,
         0},

        // down-scaled from HBM_2Gbps
        {1752,
         876, 1.1415,
         0,
         1, 1, 2,
         12, 12, 12, 12, 2,
         29, 41,
         3, 4, 3, 7, 14,
         4, 5, 14,
         0, 3417, 0,
         0, 0,
         0, 0,
         0},

        //A100
         {2000, 
         1000, 1.0,
         0,
         1, 1, 2,
         14, 14, 14, 14, 2,
         33, 47,
         3, 4, 3, 8, 16,
         4, 6, 16,
         0, 3900, 0,
         0, 0,
         0, 0,
         0},

    }, speed_entry;

    int read_latency;
    int write_latency;

    bool refresh = true;

private:
    void init_speed();
    void init_lambda();
    void init_prereq();
    void init_rowhit();  // SAUGATA: added function to check for row hits
    void init_rowopen();
    void init_timing();
};

} /*namespace ramulator*/

#endif /*__HBM_H*/
