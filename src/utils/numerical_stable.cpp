#include "utils/numerical_stable.hpp"

namespace Utils {
void GreensWorkspace::resize(int n) {
  if (ndim == n) return;
  ndim = n;

  dlmax.resize(n);
  dlmin.resize(n);
  drmax.resize(n);
  drmin.resize(n);

  Atmp.resize(n, n);
  Btmp.resize(n, n);
  Xtmp.resize(n, n);
  Ytmp.resize(n, n);
  tmp.resize(n, n);
  B_for_solve.resize(n, n);
}
}  // namespace Utils
