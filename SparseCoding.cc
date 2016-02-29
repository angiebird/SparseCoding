#include <stdio.h>
#include <vector>
#include <algorithm>

#include "SparseCoding.h"

#define DEBUG 0

typedef std::pair<double, int> pair_fi;
typedef std::pair<int, double> pair_if;

int sign(double x) {
  return x >= 0 ? 1 : -1;
}

void show_theta_map(const hash_map_ii& theta_map) {
  printf("theta_map:\n");
  for(auto& it : theta_map) {
    printf("  idx: %d coeff: %d\n", it.first, it.second);
  }
}

void show_x_map(const hash_map_if& x_map) {
  printf("x_map:\n");
  for(auto& it : x_map) {
    printf("  idx: %d coeff: %f\n", it.first, it.second);
  }
}

void show_matrix(String name, Mat A) {
  int rows = A.size().height;
  int cols= A.size().width;
  printf("%s=\n", name.c_str());
  printf("[\n");
  for(int r = 0; r < rows; r++) {
    for(int c = 0; c < cols; c++) {
      printf("%f ", A.at<double>(r, c));
    }
    printf("\n");
  }
  printf("]\n");
}

Mat random_matrix(int rows, int cols) {
  Mat A(rows, cols, CV_64F);
  for(int r = 0; r < rows; r++) {
    for(int c = 0; c < cols; c++) {
      A.at<double>(r, c) = (rand() % (1<<8)) * 1. / (1<<8);
      //A.at<double>(r, c) = r+1;
    }
  }
  return A;
}

Mat get_Ax(Mat x, Mat A, hash_map_ii theta_map) {
  Mat Ax = Mat::zeros(A.size().height, 1, CV_64F);
  int xi = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    Ax += x.at<double>(xi) * A.col(idx);
    xi ++;
  }
  return Ax;
}

Mat multiply(const Mat& A, const hash_map_if& x_map) {
  Mat Ax = Mat::zeros(A.size().height, 1, CV_64F);
  for(auto& x : x_map) {
    int x_idx = x.first;
    double x_v = x.second;
    Ax += A.col(x_idx) * x_v;
  }
  return Ax;
}

// d||y - Ax ||^2 / dxi = 2 * Tr(y - Ax) A.col(i)
double partial_differential(hash_map_if x_map, Mat A, Mat y, int x_idx) {
  Mat Ax = multiply(A, x_map);
  Mat y_Ax = y - Ax;
  return -2 * y_Ax.dot(A.col(x_idx));
}

double partial_differential_ref(hash_map_if x_map, Mat A, Mat y, int x_idx) {
  // Ax
  Mat Ax = multiply(A, x_map);
  Mat y_Ax = y - Ax;
  double f_x = y_Ax.dot(y_Ax);

  double delta = 0.001;
  y_Ax -= delta * A.col(x_idx);
  double f_x_delta = y_Ax.dot(y_Ax);
  return (f_x_delta - f_x) / delta;
}

hash_map_if get_x_map(const hash_map_ii& theta_map, const Mat& x) {
  hash_map_if x_map;
  int xi = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    x_map[idx] = x.at<double>(xi, 0);
    xi++;
  }
  return x_map;
}

hash_map_ii get_theta_map(const hash_map_if& x_map) {
  hash_map_ii theta_map;
  for(auto& it : x_map) {
    theta_map[it.first] = sign(it.second);
  }
  return theta_map;
}

void get_theta_map_and_x(const hash_map_if& x_map, hash_map_ii& theta_map, Mat& x) {
  theta_map.clear();
  for(auto& it : x_map) {
    int idx = it.first;
    theta_map[idx] = sign(it.second);
  }
  x = Mat(theta_map.size(), 1, CV_64F);
  int x_idx = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    auto itx = x_map.find(idx);
    if(itx != x_map.end()) {
      x.at<double>(x_idx, 0) = itx->second;
    }
    x_idx++;
  }
}

// 2a
// update theta_map 
// update x
// add entry with  partial_diferential > r into theta_map and x_map
void pick_theta_map(hash_map_if& x_map, const Mat& A, const Mat& y, const double r, hash_map_ii& theta_map) {

  // find the best partial differential of ||y - Ax||^2
  int w = A.size().width;
  double best_df = r;
  int best_i = -1;
  for(int i = 0; i < w; i++) {
    if(theta_map.find(i) == theta_map.end()) {
      double df = partial_differential(x_map, A, y, i);
      if(fabs(df) > fabs(best_df)) {
        best_df = df;
        best_i = i;
      }
    }
  }

  if(best_i >= 0) {
    theta_map[best_i] = -sign(best_df);
    x_map[best_i] = 0;
  }
}

double QP_error(const hash_map_if& x_map, const Mat& A, const Mat& y, const double& r, const hash_map_ii& theta_map) {
  Mat Ax = multiply(A, x_map);
  double theta_x = 0; 
  for(auto& it : theta_map) {
    int idx = it.first;
    int theta = it.second;
    double xv = x_map.find(idx)->second;
    theta_x += theta * xv;
  }
  Mat err_vec = y - Ax;
  return err_vec.dot(err_vec) + r * theta_x;
}

Mat get_theta(hash_map_ii theta_map) {
  Mat theta(theta_map.size(), 1, CV_64F);
  int theta_idx = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    int coeff = it.second;
    theta.at<double>(theta_idx, 0) = coeff;
    theta_idx++;
  }
  return theta;
}

hash_map_if QP_partial_differential(const hash_map_if& x_map, const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map) {
  // Ax
  Mat Ax = multiply(A, x_map);
  Mat y_Ax = y - Ax;

  // -2 * Tr(A)(y-Ax)
  Mat AT_y_Ax = Mat::zeros(theta_map.size(), 1, CV_64F);

  int AT_y_Ax_idx = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    AT_y_Ax.at<double>(AT_y_Ax_idx) = -2 * y_Ax.dot(A.col(idx));
    AT_y_Ax_idx++;
  }

  Mat theta = get_theta(theta_map);
  Mat df = AT_y_Ax + r*theta;
  hash_map_if df_map = get_x_map(theta_map, df);
  return df_map;
}

// 3a
// min_x ||y - Ax||^2 + r* Tr(theta)x
hash_map_if QP_solution(const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map) {
  int x_size = theta_map.size();

  // get ATA
  Mat ATA(x_size, x_size, CV_64F);
  int ri = 0;
  for(auto& it : theta_map) {
    // row
    int A_ri = it.first;
    int ci = 0;
    for(auto& it : theta_map) {
      //col
      int A_ci = it.first;
      ATA.at<double>(ri, ci) = A.col(A_ri).dot(A.col(A_ci));
      ci++;
    }
    ri++;
  }
  Mat ATA_inv = ATA.inv(DECOMP_SVD); 

  // get ATy
  Mat ATy(x_size, 1, CV_64F);
  int ATy_idx = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    ATy.at<double>(ATy_idx, 0) = A.col(idx).dot(y);
    ATy_idx++;
  }

  // get theta
  Mat theta = get_theta(theta_map);
  Mat x = ATA_inv * (ATy - theta * (r/2));
  return get_x_map(theta_map, x);
}

double one_norm_error(hash_map_if x_map, Mat A, Mat y, double r) {
  Mat Ax = multiply(A, x_map);

  double one_norm_x = 0; 
  for(auto& it : x_map) {
    int idx = it.first;
    one_norm_x += fabs(it.second);
  }

  Mat err_vec = y - Ax;
  return err_vec.dot(err_vec) + r * one_norm_x;
}

// x*(1-a) + x_new*a
hash_map_if interpolate(const hash_map_if& x_map, const hash_map_if& x_map_new, const double a) {
  hash_map_if x_map_interp;
  for(auto& it : x_map) {
    int idx = it.first;
    double xv = it.second;
    double xv_new = x_map_new.find(idx)->second;
    x_map_interp[idx] = (1-a) * xv + a * xv_new;
  }
  return x_map_interp;
}

// 3b
// argmin ||y - A(x + ap)||^2 + r*Tr(theta)(x + ap)
// update theta_map and x
double one_norm_line_search(const Mat& A, const Mat& y, const double r, const hash_map_if& x_map, const hash_map_if& x_map_new, hash_map_if& x_map_best) {

  // search sign-change points
  std::vector<pair_fi> separate_points;
  for(auto& it : x_map) {
    int idx = it.first;
    double xv = it.second;
    double xv_new = x_map_new.find(idx)->second;
    if(sign(xv) != sign(xv_new)) {
      separate_points.push_back(pair_fi(xv/(xv - xv_new), idx));
    }
  }

  std::sort(separate_points.begin(), separate_points.end());

  double err = one_norm_error(x_map, A, y, r);
  double err_new = one_norm_error(x_map_new, A, y, r);

  double a = 0;
  x_map_best = x_map;
  if(err_new < err) {
    err = err_new;
    x_map_best = x_map_new;
    a = 1;
  }

  // update x
  for(auto& pt : separate_points) {
    double tmp_a = pt.first;
    int theta_idx = pt.second;
    hash_map_if x_map_tmp = interpolate(x_map, x_map_new, tmp_a);
    double err_tmp = one_norm_error(x_map_tmp, A, y, r);
    if(err_tmp < err) {
      err = err_tmp;
      x_map_best = x_map_tmp;
      a = tmp_a;
    }
  }

  // erase near zero entry
  for(auto it = x_map_best.begin(); it != x_map_best.end();) {
    if(fabs(it->second) < EPSILON) {
      it = x_map_best.erase(it);
    } else {
      it++;
    }
  }

  return a;
}

// 4a
int check_nonzero_opt_condition(const hash_map_if& x_map, const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map) {
  for(auto& x : x_map) {
    int idx = x.first;
    double df = partial_differential(x_map, A, y, idx);
    double vx = x.second;
    if(fabs(df + r * sign(vx)) >= EPSILON) {
      return 0;
    }
  }
  return 1;
}

// 4b
int check_zero_opt_condition(const hash_map_if& x_map, const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map) {
  for(int i = 0; i < A.size().width; i++) {
    if(theta_map.find(i) == theta_map.end()) {
      double df = partial_differential(x_map, A, y, i);
      if(fabs(df) > r + EPSILON) {
        return 0;
      }
    }
  }
  return 1;
}

hash_map_if feature_sign_search(Mat A, Mat y, double r) {
  Mat x; 
  hash_map_ii theta_map;
  hash_map_if x_map;
  do {
    pick_theta_map(x_map, A, y, r, theta_map);
    do {
      hash_map_if x_map_new = QP_solution(A, y, r, theta_map);
      one_norm_line_search(A, y, r, x_map, x_map_new, x_map);
      theta_map = get_theta_map(x_map);
      double err = one_norm_error(x_map, A, y, r);
      printf("err: %f\n", err);
    } while(!check_nonzero_opt_condition(x_map, A, y, r, theta_map));
  } while(!check_zero_opt_condition(x_map, A, y, r, theta_map));
  return x_map; 
}
