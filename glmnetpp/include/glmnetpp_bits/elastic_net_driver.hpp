#pragma once

namespace glmnetpp {

template <class ElnetImpl>
struct ElasticNetDriver
{
    // TODO: fix some API to communicate from R.

private:
    ElnetImpl impl_;
};

} // namespace glmnetpp