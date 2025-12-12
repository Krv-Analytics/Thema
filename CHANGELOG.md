# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2025-12-11

### Added

- **Cosmic Graph**: New ensemble graph representation that collapses the multiverse of Mapper graphs into a single unified graph where nodes represent individual data items
  - Opt-in via YAML configuration with `cosmic_graph.enabled: true`
  - Configurable neighborhood types (`cc` or `node`) and edge thresholds
  - Computes pseudo-Laplacian matrices across all stars to measure co-occurrence stability
- **New Star variants**: Alternative graph construction methods for generating Mapper-style graphs
  - `gudhiStar`: Leverages GUDHI's MapperComplex with auto-tuned parameters
  - `pyballStar`: Simplified BallMapper-based approach with single `EPS` parameter
- **Pseudo-Laplacian computation**: Added `get_pseudoLaplacian()` method to all Star classes for matrix representations of graph neighborhoods

### Changed

- Python CI updated to 3.13
- Refactored shared star utilities into `starHelpers.py` module
- `starGraph.components` now cached for performance

### Dependencies

- Added: `gudhi>=3.11.0`
- Added: `pyballmapper>=0.3.7`

### Documentation

- New Cosmic Graph user guide with configuration examples and API reference
- Enhanced Stars documentation with comparison table and usage recommendations

## [0.1.3] - 2025-10-23

### Added

- **New `Thema` class**: Automated end-to-end pipeline orchestration with `genesis()` method
- **Comprehensive logging system**: Debug-level logging throughout the multiverse components
- **Graph filtering capabilities**: YAML-configurable filter functions for star model selection
- **Analytics module `expansion`**: Domain-specific analysis tools for real estate and geographic data
- **Automated PyPI publishing workflow**: GitHub Actions for seamless package releases
- **Star filtering utilities**: Min-coverage filters and custom filter function support
- **Galaxy coordinates**: MDS embedding of model space for analysis
- **Robust path handling**: Support for Path-like objects throughout the codebase

### Changed

- **BREAKING**: Removed visualization support (observatory module and Telescope entry point)
- Bump `curvature-filtrations` dependency to >=0.1.2
- **Default curvature measure**: Switched from "forman_curvature" to "ollivier_ricci_curvature"
- **Improved error handling**: Better exception management and user feedback
- **Enhanced documentation**: Comprehensive README with badges, contact links, and usage examples
- **Code formatting**: Applied Black formatting across the entire codebase
- **Dependency management**: Reorganized optional dependencies in pyproject.toml
- **Test optimization**: Split high-compute tests for local-only execution

### Fixed

- **Galaxy collapse logic**: Fixed filter handling when `filter=None`
- **Planet YAML creation**: Robust handling when YAML file doesn't exist
- **Import error handling**: Graceful `tqdm` import failures
- **Path safety**: Type-safe casts and consistent path handling
- **Model file tracking**: Proper initialization and population of `selected_model_files`

### Removed

- **Observatory module**: Complete removal of visualization components
- **Telescope class**: Removed legacy main entry point
- **UMAP support**: Purged UMAP mentions from source, tests, and documentation
- **Plotly dependencies**: Removed visualization-related imports

### Performance

- **Multiprocessing improvements**: Galaxy instances made pickle-friendly
- **Timing instrumentation**: Added duration measurement for major operations
- **Progress reporting**: Enhanced logging with file deltas and worker counts

### Documentation

- **Sphinx documentation**: Complete documentation build with Furo theme
- **API reference**: Comprehensive module documentation
- **User guides**: Beginner, intermediate, and advanced tutorials
- **Configuration guide**: Detailed parameter reference

### Dependencies

- Added: black>=25.9.0 (development)
- Added: furo>=2025.7.19 (docs)
- Added: sphinx>=8.1.3 (docs)
- Updated: curvature-filtrations>=0.1.2
- Removed: UMAP-related dependencies

---

### Contributing

When adding entries to this changelog:

1. Add new entries under `[Unreleased]` section
2. Use the following categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
3. Follow the existing format and style
4. Include relevant issue/PR numbers when applicable
5. Move entries from `[Unreleased]` to a new version section when releasing
