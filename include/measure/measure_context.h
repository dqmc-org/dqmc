#pragma once

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

  MeasureContext(const MeasureHandler& h, const Walker& w, const ModelBase& m, const LatticeBase& l)
      : handler(h), walker(w), model(m), lattice(l) {}
};
}  // namespace Measure
