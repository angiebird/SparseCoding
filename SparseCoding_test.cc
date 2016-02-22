#include <limits.h>
#include "SparseCoding.h"
#include <gtest/gtest.h>

TEST(SparseCodingTest, multiply) {
  Mat A = random_matrix(5, 3);
  hash_map_if x_map;
  x_map[0] = 0.2;
  x_map[2] = 0.1;
  Mat Ax = multiply(A, x_map);
  EXPECT_EQ(Ax.size().height, 5);
  EXPECT_EQ(Ax.size().width, 1);

  Mat Ax_ref = A.col(0) * 0.2 + A.col(2) * 0.1;
  for(int r = 0; r < 5; r++) {
    EXPECT_EQ(Ax_ref.at<double>(r, 0), Ax.at<double>(r, 0));
  }
}

TEST(SparseCodingTest, partial_differential) {
  Mat A = random_matrix(5, 3);
  Mat y = random_matrix(5, 1);
  hash_map_if x_map;
  x_map[0] = 0.2;
  x_map[2] = 0.1;
  int x_idx = 2;
  double df = partial_differential(x_map, A, y, x_idx);
  double df_ref = partial_differential_ref(x_map, A, y, x_idx);
  EXPECT_LT(fabs(df - df_ref), 0.001);
}

TEST(SparseCodingTest, get_theta_map) {
  hash_map_if x_map;
  x_map[0] = 0.2;
  x_map[2] = -0.1;
  hash_map_ii theta_map = get_theta_map(x_map);
  EXPECT_EQ(theta_map.find(0)->second, 1);
  EXPECT_EQ(theta_map.find(2)->second, -1);
}

TEST(SparseCodingTest, pick_theta_map) {
  Mat A = random_matrix(5, 3);
  Mat y = random_matrix(5, 1);
  hash_map_if x_map;
  x_map[0] = 0.2;
  x_map[2] = 0.1;
  hash_map_ii theta_map = get_theta_map(x_map);
  int x_idx = 1;
  double df = partial_differential(x_map, A, y, x_idx);
  double r = fabs(df)/2;
  pick_theta_map(x_map, A, y, r, theta_map);
  auto x = x_map.find(x_idx);
  auto theta = theta_map.find(x_idx);
  EXPECT_NE(x, x_map.end());
  EXPECT_EQ(x->second, 0);
  EXPECT_NE(theta, theta_map.end());
  EXPECT_EQ(theta->second, -sign(df));
}
TEST(SparseCodingTest, check_nonzero_opt_condition) {
  Mat A = random_matrix(5, 3);
  Mat y = random_matrix(5, 1);
  hash_map_if x_map;
  int x_idx = 2;
  x_map[x_idx] = 0.1;
  hash_map_ii theta_map = get_theta_map(x_map);
  double df = partial_differential(x_map, A, y, x_idx);
  double r = -df / sign(0.1);
  EXPECT_EQ(check_nonzero_opt_condition(x_map, A, y, r, theta_map), 1);
  r = fabs(df) + EPSILON;
  EXPECT_EQ(check_nonzero_opt_condition(x_map, A, y, r, theta_map), 0);
}
