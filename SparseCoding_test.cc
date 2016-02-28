#include <limits.h>
#include "SparseCoding.h"
#include <gtest/gtest.h>

using namespace std;

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

TEST(SparseCodingTest, check_zero_opt_condition) {
  Mat A = random_matrix(5, 3);
  Mat y = random_matrix(5, 1);
  hash_map_if x_map;
  x_map[0] = 0.1;
  x_map[2] = 0.2;
  hash_map_ii theta_map = get_theta_map(x_map);
  double df = partial_differential(x_map, A, y, 1);
  double r = fabs(df)/2;
  EXPECT_EQ(check_zero_opt_condition(x_map, A, y, r, theta_map), 0);
  r = 2 * fabs(df);
  EXPECT_EQ(check_zero_opt_condition(x_map, A, y, r, theta_map), 1);
}

TEST(SparseCodingTest, QP_solution) {
  hash_map_ii theta_map;
  theta_map[0] = 1;
  theta_map[1] = -1;
  theta_map[2] = 1;
  Mat A = random_matrix(3, 3);
  Mat y = random_matrix(3, 1);
  double r = 0.0;
  hash_map_if x_map = QP_solution(A, y, r, theta_map);
  if(fabs(determinant(A)) > EPSILON) {
    // When r = 0, the QP_error should is near to zero too
    double error = QP_error(x_map, A, y, r, theta_map);
    EXPECT_LT(error, EPSILON);
  }

  r = 0.1;
  x_map = QP_solution(A, y, r, theta_map);
  if(fabs(determinant(A)) > EPSILON) {
    hash_map_if df = QP_partial_differential(x_map, A, y, r, theta_map);
    for(auto& it : df) {
      EXPECT_LT(fabs(it.second), EPSILON);
    }
  }
}

TEST(SparseCodingTest, interpolate) {
  hash_map_if x1_map;
  hash_map_if x2_map;

  x1_map[0] = 1;
  x1_map[1] = 2;
  x1_map[2] = 3;

  x2_map[0] = 4;
  x2_map[1] = 5;
  x2_map[2] = 6;

  double a = 0.5;
  hash_map_if x3_map = interpolate(x1_map, x2_map, a);
  for(auto& it : x3_map) {
    int idx = it.first;
    double xv3 = it.second;
    double xv1 = x1_map.find(idx)->second;
    double xv2 = x2_map.find(idx)->second;
    EXPECT_LT(fabs(xv3 - ((1-a)*xv1 + a*xv2)), EPSILON);
  }
}

TEST(SparseCodingTest, one_norm_line_search) {
  double r = 1;
  hash_map_if x_map;
  x_map[0] = 1;
  x_map[1] = 1;
  x_map[2] = 1;
  hash_map_ii theta_map = get_theta_map(x_map);
  Mat A = random_matrix(3, 6);
  Mat y = random_matrix(3, 1);
  hash_map_if x_map_new = QP_solution(A, y, r, theta_map);
  hash_map_if x_map_best;
  double a = one_norm_line_search(A, y, r, x_map, x_map_new, x_map_best);

  EXPECT_LE(a, 1);
  EXPECT_GE(a, 0);

  for(auto& it : x_map) {
    int idx = it.first;
    double xv = it.second;
    double xv_new = x_map_new.find(idx)->second;
    double xv_best = 0;
    auto it_best = x_map_best.find(idx);
    if(it_best != x_map_best.end()) {
      xv_best = it_best->second;
    }
    double xv_interp = (1 - a) * xv + a * xv_new;
    EXPECT_LT(fabs(xv_best - xv_interp), EPSILON);
  }
}
