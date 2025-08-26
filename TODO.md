# TODO LIST -- [ ] for undone, [~] for in-progress, [X] for done

## Configuration System
- [X] Remove TOML dependency and refactor to use Boost.Program_options for CLI and config file parsing

## Code Quality
- [ ] Investigate [[maybe_unused]] parameters in measurement methods - Review if unused parameters in consistent interfaces can be utilized or if the interface can be refactored

## Lattice System
- [ ] Support rectangular (non-square) 2D lattices - Currently crashes with assertion failure when using uneven dimensions like 4x2 due to hardcoded square-only assumption in Square lattice class

