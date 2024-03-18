#include "HMS.h"
#include "DRAM.h"

namespace ramulator {

// Recursively check whether my parent is SCM or not
bool isSCM(DRAM<HMS>* node) {
  bool isSCMChild = false;
  DRAM<HMS>* curNode = node;
  while (curNode->parent != NULL) {
    if (int(curNode->level) == int(HMS::Level::Rank)) {
      if (curNode->id == HMS::SCM_RANK_ID) {
        isSCMChild = true;
        break;
      } else {
        isSCMChild = false;
        break;
      }
    } else {
      curNode = curNode->parent;
    }
  }

  return isSCMChild;
}

template<>
DRAM<HMS>::DRAM(HMS* spec, HMS::Level level, DRAM<HMS>* parent, int id, HMS::Type parent_type) :
  spec(spec), level(level), id(id), parent(parent) {
  //spec(spec), level(level), id(0), parent(NULL) {

  state = spec->start[static_cast<int>(level)];
  prereq = spec->prereq[static_cast<int>(level)];
  rowhit = spec->rowhit[static_cast<int>(level)];
  rowopen = spec->rowopen[static_cast<int>(level)];
  lambda = spec->lambda[static_cast<int>(level)];

  
  //jeongmin - decide type of current DRAM
  if (level == HMS::Level::Rank) {
    if (parent_type == HMS::Type::SCMDRAM) {
      //channel type is SCMDRAM
      if (id == HMS::DRAM_RANK_ID) {
        //current rank is DRAM rank
        type = HMS::Type::DRAM;
      } else if (id == HMS::SCM_RANK_ID) {
        //current rank is SCM rank
        type = HMS::Type::SCM;
      }
    } else if (parent_type == HMS::Type::SCM) {
      //channel type is SCM 
      type = HMS::Type::SCM;// this channel only have SCM
    } else {
      assert("\nChannel cannot have DRAM type!!\n" && 0);
    }
  } else {
    type = parent_type;
  }

  //if (isSCM(this))
  if (type == HMS::Type::SCM)
    timing = spec->scm_timing[static_cast<int>(level)];
  else
    timing = spec->dram_timing[static_cast<int>(level)];
 

  fill_n(next, int(HMS::Command::MAX), -1);
  for (int cmd = 0; cmd < static_cast<int>(HMS::Command::MAX); ++cmd) {
    int dist = 0;
    for (auto &t : timing[cmd])
      dist = max(dist, t.dist);

    if (dist)
      prev[cmd].resize(dist, -1); // initialize history
  }

  // try to recursively construct my children
  int child_level = int(level) + 1; 
  if (child_level == int(HMS::Level::Row))
    return;

  int child_max = 0;
  //if (isSCM(this))
  if (type == HMS::Type::SCM)
    child_max = spec->scm_org_entry.count[child_level];
  else
    child_max = spec->dram_org_entry.count[child_level];

  if (!child_max)
      return; // stop recursion: the number of children is unspecified

  for (int i = 0; i < child_max; i++) {
    DRAM<HMS>* child = 
      new DRAM<HMS>(spec, HMS::Level(child_level), this, i, type);
      //new DRAM<HMS>(spec, HMS::Level(child_level));
    //child->parent = this;
    //child->id = i;
    children.push_back(child);
  }
}

} // end namespace
