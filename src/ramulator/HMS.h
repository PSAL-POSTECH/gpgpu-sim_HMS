#ifndef __HMS_H
#define __HMS_H

//#include "DRAM.h"
#include "Request.h"
#include <vector>
#include <functional>

using namespace std;

namespace ramulator
{

template <typename T> class DRAM;

class HMS
{
public:   
    static int DRAM_RANK_ID;
    static int SCM_RANK_ID;

    static string standard_name;

    enum class DRAM_Org;
    enum class DRAM_Speed;
    enum class SCM_Org;
    enum class SCM_Speed;

    HMS(DRAM_Org dram_org, DRAM_Speed dram_speed,
        SCM_Org scm_org, SCM_Speed scm_speed,
        int channel_width, int prefetch_size);

    HMS(const string& dram_org_str, const string& dram_speed_str,
        const string& scm_org_str, const string& scm_speed_str,
        int channel_width, int prefetch_size);

    static map<string, enum DRAM_Org> dram_org_map;
    static map<string, enum DRAM_Speed> dram_speed_map;

    static map<string, enum SCM_Org> scm_org_map;
    static map<string, enum SCM_Speed> scm_speed_map;

    /* Level */
    enum class Level : int
    {
        Channel, Rank, BankGroup, Bank, Row, Column, MAX  // dram and scm have same memory structure
    };

    enum class Type : int
    {
        SCM, DRAM, SCMDRAM, MAX 
    };

    string level_name[int(Level::MAX)] = {
      "Channel", "Rank", "BankGroup", "Bank", "Row", "Column"
    };

    string type_name[int(Type::MAX)] = {
      "SCM", "DRAM", "SCMDRAM"
    };

    /* Command */
    enum class Command : int
    {
        ACT, PRE, PREA, // common commands for dram and scm

        RD, WR, RDA, WRA, // dram commands for read and write

        REF, REFSB, PDE, PDX,  SRE, SRX,
        MAX
    };

    // REFSB and REF is not compatible, choose one or the other.
    // REFSB can be issued to banks in any order, as long as REFI1B
    // is satisfied for all banks

    string command_name[int(Command::MAX)] = {
        "ACT", "PRE",   "PREA",

        "RD", "WR", "RDA", "WRA",

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
    bool is_reading(Command cmd) { return cmd == Command::RD; }
    bool is_writing(Command cmd) { return cmd == Command::WR; }
    bool is_normal_closing(Command cmd) { return cmd == HMS::Command::PRE; }
    bool is_refresh_closing(Command cmd) { return cmd == HMS::Command::PREA; }

    /* State */
    enum class State : int
    {
        Opened, Closed, PowerUp, ActPowerDown, PrePowerDown, SelfRefresh, MAX
    } start[int(Level::MAX)] = {
        State::MAX, State::PowerUp, State::MAX, State::Closed, State::Closed, State::MAX
    };

    /* Translate */
    //Request::Type::MAX
    Command translate[int(request::Type::MAX)] = {
      Command::RD,  Command::WR,
      Command::REF, Command::PDE, Command::SRE,
    };

    /* Prereq */
    function<Command(DRAM<HMS>*, Command cmd, int)> prereq[int(Level::MAX)][int(Command::MAX)];

    // SAUGATA: added function object container for row hit status
    /* Row hit */
    function<bool(DRAM<HMS>*, Command cmd, int)> rowhit[int(Level::MAX)][int(Command::MAX)];
    function<bool(DRAM<HMS>*, Command cmd, int)> rowopen[int(Level::MAX)][int(Command::MAX)];

    /* Timing */
    struct TimingEntry
    {
        Command cmd;
        int dist;
        int val;
        bool sibling;
    };
    vector<TimingEntry> dram_timing[int(Level::MAX)][int(Command::MAX)];
    vector<TimingEntry> scm_timing[int(Level::MAX)][int(Command::MAX)];

    /* Lambda */
    function<void(DRAM<HMS>*, int)> lambda[int(Level::MAX)][int(Command::MAX)];

    /* Organization */
    enum class DRAM_Org : int
    { // per channel density here. Each stack comes with 8 channels
      // TITAN V has 3 stacks (total 12GBs)
      // each stack (4GBs) owns four dram dies with 2 channels each
      // each dram die owns 1GB -> 1GB/2 channels = 4Gb (0.5 GB) per channel
        HMS_1Gb,
        HMS_2Gb,
        HMS_4Gb, // we will use this , 4 bank groups, 4 banks, page size: 2KB
        MAX
    };

    enum class SCM_Org : int
    {
        PCM_SLC,  // 16Gb
        PCM_MLC,  // 16Gb
        PCM_TLC,  // 16Gb
        VRRAM,  // 16Gb
        MAX
    };

    struct DRAM_OrgEntry {
        int size;
        int dq;
        int count[int(Level::MAX)]; //Channel, Rank, BankGroup, Bank, Row, Column
    } dram_org_table[int(DRAM_Org::MAX)] = {
        {1<<10, 128, {0, 0, 4, 2, 1<<13, 1<<(6+1)}},
        {2<<10, 128, {0, 0, 4, 2, 1<<14, 1<<(6+1)}},
        {4<<10, 128, {0, 0, 4, 4, 1<<14, 1<<(6+1)}},
    }, dram_org_entry;

    struct SCM_OrgEntry {
        int size;
        int dq;
        int count[int(Level::MAX)];
    } scm_org_table[int(SCM_Org::MAX)] = {
        {16<<10, 128, {0, 0, 4, 4, 1<<15, 1<< (6 + 1)}},  // SLC
        {16<<10, 128, {0, 0, 4, 4, 1<<16, 1<< (6 + 1)}},  // MLC
        {16<<10, 128, {0, 0, 4, 4, 1<<17, 1<< (6 + 1)}},  // TLC
        {16<<10, 128, {0, 0, 4, 4, 1<<17, 1<< (6 + 1)}},  // VRRAM
    }, scm_org_entry;

    void set_channel_number(int channel);
    void set_rank_number(int rank);

    /* Speed */
    enum class DRAM_Speed : int
    {
        HMS_1Gbps,
        HMS_2Gbps,  // we will use this
        HMS_V100,
        HMS_A100,
        MAX
    };

    enum class SCM_Speed : int
    {
        PCM_SLC,
        PCM_MLC,
        PCM_TLC,
        VRRAM,
        PCM_SLC_A100,
        PCM_MLC_A100,
        PCM_TLC_A100,
        VRRAM_A100,
        MAX
    };


    int prefetch_size = 2; // burst length could be 2 and 4 (choose 4 here), 2n prefetch
    int channel_width = 128;  // 16 bytes channel width

    // dram speed entry
    struct DRAM_SpeedEntry {
        int rate;
        double freq, tCK;
        int nBL, nCCDS, nCCDL;
        int nRTRS;
        int nCL, nRCDR, nRCDW, nRP, nCWL;
        int nRAS, nRC;
        int nRTPS, nRTPL, nWTRS, nWTRL, nWR;
        int nRRDS, nRRDL, nFAW;
        int nRFC, nREFI, nREFI1B;
        int nPD, nXP;
        int nCKESR, nXS;
    } dram_speed_table[int(DRAM_Speed::MAX)] = {
        {1000, 
         500, 2.0, 
         2, 2, 3, 
         0,
         7, 7, 6, 7, 4, 
         17, 24, 
         7, 2, 4, 8, 
         4, 5, 20, 
         0, 1950, 0, 
         5, 5, 
         5, 0},

        {2000, 
         1000, 1.0,
         1, 1, 2,
         0,
         14, 14, 14, 14, 2,
         33, 47,
         3, 4, 3, 8, 16,
         4, 6, 16,
         0, 3900, 0,
         0, 0,
         0, 0},

        // down-scaled from HMS_2Gbps
        {1752,
         876, 1.1415,
         1, 1, 2,
         0,
         12, 12, 12, 12, 2,
         29, 41,
         3, 4, 3, 7, 14,
         4, 5, 14,
         0, 3417, 0,
         0, 0,
         0, 0},

        // for A100
        {2000, 
         1000, 1.0,
         1, 1, 2,
         0,
         14, 14, 14, 14, 2,
         33, 47,
         3, 4, 3, 8, 16,
         4, 6, 16,
         0, 3900, 0,
         0, 0,
         0, 0},
    }, dram_speed_entry;

    struct SCM_SpeedEntry {
        int rate;
        double freq, tCK;
        int nBL, nCCDS, nCCDL;
        int nRTRS;
        int nCL, nRCDR, nRCDW, nRP, nCWL;
        int nRAS, nRC;
        int nRTPS, nRTPL, nWTRS, nWTRL, nWR;
        int nRRDS, nRRDL, nFAW;
    } scm_speed_table[int(SCM_Speed::MAX)] = {
      // 876 Mhz setting for tesla v100
      
      {1752, 876.0, 1.1415,   1, 1, 2,  0,  12, 52,  52,  12, 2,   52,  64,    3, 4, 3, 7, 131,   4, 5, 14}, // SLC
      {1752, 876.0, 1.1415,   1, 1, 2,  0,  12, 105, 105, 12, 2,   105, 117,   3, 4, 3, 7, 876,   4, 5, 14}, // MLC
      {1752, 876.0, 1.1415,   1, 1, 2,  0,  12, 219, 219, 12, 2,   219, 231,   3, 4, 3, 7, 2059,  4, 5, 14}, // TLC
      {1752, 876.0, 1.1415,   1, 1, 2,  0,  12, 263, 263, 12, 2,   263, 275,   3, 4, 3, 7, 88,  4, 5, 14}, // VRRAM

      // 1000 Mhz setting for A100
      
      {2000, 1000, 1.0,   1, 1, 2,  0,  14, 60,  60,  14, 2,   60,  74,    3, 4, 3, 8, 150,   4, 6, 16}, // SLC
      {2000, 1000, 1.0,   1, 1, 2,  0,  14, 120, 120, 14, 2,   120, 134,   3, 4, 3, 8, 1000,   4, 6, 16}, // MLC
      {2000, 1000, 1.0,   1, 1, 2,  0,  14, 250, 250, 14, 2,   250, 264,   3, 4, 3, 8, 2350,  4, 6, 16}, // TLC
      {2000, 1000, 1.0,   1, 1, 2,  0,  14, 301, 301, 14, 2,   300, 314,   3, 4, 3, 8, 100,  4, 6, 16}, // VRRAM

    }, scm_speed_entry;
    
    struct CommonSpeedEntry {
        int rate; 
        double freq, tCK;
    } speed_table[int(DRAM_Speed::MAX)] = { 
        {1000, 500, 2.0,}, //HMS_1Gbps
        {2000, 1000, 1.0}, //HMS_2Gbps
        {1752, 876.0, 1.1415}, //HMS_V100
        {2000, 1000, 1.0}, //HMS_A100
    }, speed_entry;

    int read_latency;
    int write_latency;

    int nTR_CCD;

    // Not used in HMS mode
    bool refresh = false;

private:
    void init_speed();
    void init_lambda();
    void init_prereq();
    void init_rowhit();  // SAUGATA: added function to check for row hits
    void init_rowopen();
    void init_dram_timing();
    void init_scm_timing();
};

} /*namespace ramulator*/

#endif /*__HMS_H*/

