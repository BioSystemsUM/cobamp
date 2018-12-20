# Change log
Major changes between versions will be documented on this file.

## [0.0.2] - 2018-12-20
### Added
 - Basic metabolic model classes for when an SBML file is not available or necessary
 - Elementary flux pattern enumeration using the K-Shortest algorithm (as a wrapper)
 
### Changed
 - Entire package structure to accomodate other types of algorithms beyond the K-Shortest one

### Removed
 - Some modules that were too specific for the changes above
## [0.0.1] - 2018-12-04
### Added

- Base code for K-Shortest enumeration of EFMs and MCSs
- Core framework architecture (linear systems, algorithms and wrappers)
- COBRApy and framed model object readers