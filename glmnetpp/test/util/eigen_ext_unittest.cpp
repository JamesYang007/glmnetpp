#include <testutil/base_fixture.hpp>
#include <glmnetpp_bits/util/eigen_ext.hpp>

namespace glmnetpp {
namespace util {

struct eigen_ext_fixture : base_fixture
{
protected:
    Eigen::VectorXd dest;
    Eigen::VectorXd expected_dest;
};

TEST_F(eigen_ext_fixture,
        geomspace_ascending_pow2)
{
    geomspace(dest, 1., 16., 5);

    expected_dest.resize(5);
    expected_dest << 1, 2, 4, 8, 16;

    expect_double_eq_vec(dest, expected_dest);
}

TEST_F(eigen_ext_fixture,
        geomspace_descending_pow2)
{
    geomspace(dest, 16., 1., 5);

    expected_dest.resize(5);
    expected_dest << 16, 8, 4, 2, 1;

    expect_double_eq_vec(dest, expected_dest);
}

TEST_F(eigen_ext_fixture,
        geomspace_ascending_arbitrary)
{
    geomspace(dest, 0.2, 14.23, 6);

    expected_dest.resize(6);
    expected_dest << 0.2, 0.46931559, 1.10128559, 2.58425246, 6.06414978, 14.23;

    expect_near_vec(dest, expected_dest, 1e-8);
}

TEST_F(eigen_ext_fixture,
        geomspace_descending_arbitrary)
{
    geomspace(dest, 9.23, 0.52, 6);

    expected_dest.resize(6);
    expected_dest << 9.23, 5.19232693, 2.92093813, 1.64317072, 0.92436398, 0.52;

    expect_near_vec(dest, expected_dest, 1e-8);
}

} // namespace util
} // namespace glmnetpp
