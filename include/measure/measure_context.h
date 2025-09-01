#pragma once

#include "utils/temporary_pool.h"

// forward declarations
namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace DQMC {
class Walker;
}
namespace Measure {
class MeasureHandler;
}

namespace Measure {

using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using Walker = DQMC::Walker;

struct MeasureContext {
  const MeasureHandler& handler;
  const Walker& walker;
  const ModelBase& model;
  const LatticeBase& lattice;
  Utils::TemporaryPool& pool;

  MeasureContext(const MeasureHandler& h, const Walker& w, const ModelBase& m, const LatticeBase& l,
                 Utils::TemporaryPool& p)
      : handler(h), walker(w), model(m), lattice(l), pool(p) {}
};
}  // namespace Measure
