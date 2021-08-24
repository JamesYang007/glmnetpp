#include <Eigen/Core>

namespace glmnetpp {

struct ElasticNetConfig
{
    uint32_t max_iter = 100000;
    uint32_t nlambda = 100;
    double thresh = 1e-7;
    double alpha = 1;
};

template <class XDerived, class YDerived>
inline void lasso_path(const Eigen::MatrixBase<XDerived>& X,
                       const Eigen::MatrixBase<YDerived>& y,
                       const ElasticNetConfig& config)
{
    auto n = X.rows();
    auto p = X.cols();
    
    
}

} // namespace glmnetpp
