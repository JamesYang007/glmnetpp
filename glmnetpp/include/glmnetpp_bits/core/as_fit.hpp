#pragma once
#include <glmnetpp_bits/util/typedefs.hpp>

namespace glmnetpp {
namespace core {

/*
 * This class is responsible for implementing an interface
 * that is compatible as a type parameter to ElasticNetImplDefault.
 * It fits an elastic net problem at a given lambda using the 
 * active/strong set principle.
 */
template <util::method_type method
        , class ResourceType>
struct ASFit;

} // namespace core
} // namespace glmnetpp