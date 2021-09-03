#pragma once

namespace glmnetpp {
namespace core {

/* 
 * The main driver of an elastic net solver.
 * This class will merely act as a delegator to the implementation (ImplType).
 * The extra layer of abstraction is to separate the implementation detail
 * from the export to higher-level languages (R/Python), as these exports
 * put additional constraints on what can be exported.
 */
template <class ImplType>
struct ElasticNet;

} // namespace core
} // namespace glmnetpp