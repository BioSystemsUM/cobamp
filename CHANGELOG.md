# Change log
Major changes between versions will be documented on this file.
## [0.1.2] - 2019-09-23
### Added
 - Gene protein reaction rule evaluation module (cobamp.gpr.evaluators). Still a work in
 progress but the aim is to eventually move all GPR code into this module
 upon releasing a major revision
 - New dependency (boolean.py)
### Changed
 - Fixed errors in the readthedocs quickstart example
 - Corrected some assorted bugs on the ConstraintBasedModel and KShortest classes

## [0.1.1] - 2019-06-07
### Changed
 - Several bug fixes
 - KShortest algorithm now adds constraints iteratively to avoid memory errors
 - Several improvements to the ConstraintBasedModel class 

## [0.1.0] - 2019-04-29
### Added
 - Big-M indicators for solvers without a dedicated indicator constraint abstraction

## [0.1.0rc1] - 2019-04-18
### Added
 - SCIPY MAT format model reader
 - Gene-protein-rule support with gene expression data integration functions
 - Analysis functions (mainly frequency and graphs) with some plotting capability
 - Transformer classes for algorithms that alter a metabolic network, 
 guaranteeing mapping between the reactions of both
 - Higher-level classes (linear_systems module) for semi-efficient definition of LP problems based on Optlang
 - Classes for evaluating and converting into postfix type arithmetic and boolean expressions

### Changed
 - Major code refactor. Module structure drastically changed and reorganized.
 - Many bugfixes

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