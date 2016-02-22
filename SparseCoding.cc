#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <tr1/unordered_map>

#define EPSILON 0.01 
#define DEBUG 0

using namespace cv;
typedef std::tr1::unordered_map<int, int> hash_map_ii;
typedef std::tr1::unordered_map<int, double> hash_map_if;
typedef std::pair<double, int> pair_fi;
typedef std::pair<int, double> pair_if;

int sign(double x) {
  return x >= 0 ? 1 : -1;
}

Mat rgb_to_gray(Mat rgb_image) {
  Mat gray_image;
  cvtColor(rgb_image, gray_image, COLOR_RGB2GRAY);
  return gray_image;
}

std::vector<Mat> get_blocks(int size, int num, Mat image) {
  // return num of size*size blocks randomly sampled from gray_image
  std::vector<Mat> blocks;
  int rows = image.size().height;
  int cols = image.size().width;
  if(rows < size || cols < size)
    printf("rows < size || cols < size\n");
  rows -= size;
  cols -= size;

  for(int i = 0; i < num; i++) {
    int r = rand() % rows;
    int c = rand() % cols;
    blocks.push_back(image(Range(r, r + size), Range(c, c + size)));
  }

  return blocks;
}

void show_theta_map(hash_map_ii theta_map) {
  printf("theta_map:\n");
  for(auto& it : theta_map) {
    printf("  idx: %d coeff: %d\n", it.first, it.second);
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

// d||y - Ax ||^2 / dxi = 2 * Tr(y - Ax) A.col(i)
double partial_differential(Mat x, Mat A, Mat y, hash_map_ii theta_map, int x_idx) {
  // Ax
  Mat Ax = get_Ax(x, A, theta_map);
  Mat y_Ax = y - Ax;
  return -2 * y_Ax.dot(A.col(x_idx));
}

double partial_differential_slow(Mat x, Mat A, Mat y, hash_map_ii theta_map, int x_idx) {
  // Ax
  Mat Ax = get_Ax(x, A, theta_map);
  Mat y_Ax = y - Ax;
  double f_x = y_Ax.dot(y_Ax);

  double delta = 0.001;
  y_Ax -= delta * A.col(x_idx);
  double f_x_delta = y_Ax.dot(y_Ax);
  return (f_x_delta - f_x) / delta;
}

void test_partial_differential() {
  double r = 0.1;
  hash_map_ii theta_map;
  theta_map[0] = 1;
  theta_map[1] = 1;
  theta_map[2] = 1;
  Mat A = random_matrix(3, 6);
  Mat y = random_matrix(3, 1);
  Mat x = Mat::zeros(theta_map.size(), 1, CV_64F);

  int x_idx = 3;
  double df = partial_differential(x, A, y, theta_map, x_idx);
  double df_ref = partial_differential_slow(x, A, y, theta_map, x_idx);

  //printf("df: %f df_ref: %f\n", df, df_ref);
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
void pick_theta_map(Mat& x, Mat A, Mat y, double r, hash_map_ii& theta_map) {
  // x to x_map
  hash_map_if x_map = get_x_map(theta_map, x);

  // find the best partial differential of ||y - Ax||^2
  int w = A.size().width;
  double best_df = r;
  int best_i = -1;
  for(int i = 0; i < w; i++) {
    if(theta_map.find(i) == theta_map.end()) {
      double df = partial_differential(x, A, y, theta_map, i);
      if(fabs(df) > fabs(best_df)) {
        best_df = df;
        best_i = i;
      }
    }
  }

  if(best_i >= 0) {
    theta_map[best_i] = sign(best_df);
    x_map[best_i] = sign(best_df);
  }

  x = Mat::zeros(theta_map.size(), 1, CV_64F);
  int xi = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    x.at<double>(xi, 0) = x_map[idx];
    xi++;
  }
}

double QP_error(Mat x, Mat A, Mat y, double r, hash_map_ii theta_map) {
  int y_size = A.size().height;
  Mat Ax = Mat::zeros(y_size, 1, CV_64F);
  double theta_x = 0; 

  // Ax theta_x
  int x_idx = 0;
  for(auto& it : theta_map) {
    int A_idx = it.first;
    int theta = it.second;
    Ax += A.col(A_idx) * x.at<double>(x_idx, 0);
    theta_x += theta * x.at<double>(x_idx, 0);
    x_idx++;
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

Mat QP_partial_differential(Mat x, Mat A, Mat y, double r, hash_map_ii theta_map) {
  // Ax
  Mat Ax = get_Ax(x, A, theta_map);
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
  return AT_y_Ax + r*theta;
}

// 3a
// min_x ||y - Ax||^2 + r* Tr(theta)x
Mat QP_solution(Mat A, Mat y, double r, hash_map_ii theta_map) {
  //printf("QP_solution start ====\n");
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
#if DEBUG
  //if(determinant(ATA) > EPSILON) {
    Mat df = QP_partial_differential(x, A, y, r, theta_map);
    int df_size = df.size().height;
    for(int i = 0; i < df_size; i++) {
      if(fabs(df.at<double>(i, 0)) > EPSILON){
        printf("QP_solution error\n");
      }
    }
    show_theta_map(theta_map);
    show_matrix("x", x);
    show_matrix("df", df);
    printf("det ATA: %f\n", determinant(ATA));
 // }
#endif
  //printf("QP_solution end ====\n");
  return x;
}

void test_QP_solution() {
  double r = 0.1;
  std::tr1::unordered_map<int, int> theta_map;
  theta_map[0] = 1;
  theta_map[1] = -1;
  theta_map[2] = 1;
  Mat A = random_matrix(3, 6);
  Mat y = random_matrix(3, 1);
  Mat x = QP_solution(A, y, r, theta_map);
  double err = QP_error(x, A, y, r, theta_map);
  printf("QP_error %f\n", err);
}

double one_norm_error(Mat x, Mat A, Mat y, double r, hash_map_ii theta_map) {
  int y_size = A.size().height;
  Mat Ax = Mat::zeros(y_size, 1, CV_64F);
  double one_norm_x = 0; 

  // Ax, one_norm_x
  int x_idx = 0;
  for(auto& it : theta_map) {
    int A_idx = it.first;
    int theta = it.second;
    Ax += A.col(A_idx) * x.at<double>(x_idx, 0);
    one_norm_x += fabs(x.at<double>(x_idx, 0));
    x_idx++;
  }

  Mat err_vec = y - Ax;
  return err_vec.dot(err_vec) + r * one_norm_x;
}

// 3b
// argmin ||y - A(x + ap)||^2 + r*Tr(theta)(x + ap)
// update theta_map and x
void one_norm_line_search(Mat A, Mat y, double r, hash_map_ii& theta_map, Mat& x, Mat x_new) {
  printf("one_norm_line_search start ====\n");
  double a = 0;
  Mat p = x_new - x;
  int p_size = p.size().height;
  std::vector<pair_fi> separate_points;
  int pi = 0;


  // search sign-change points
  for(auto& it : theta_map) {
    int theta_idx = it.first;
    double pv = p.at<double>(pi);
    double xv = x.at<double>(pi);
    double xv_new = x_new.at<double>(pi);
    if(sign(xv) != sign(xv_new)) {
      separate_points.push_back(pair_fi(-xv/pv, theta_idx));
    }
    pi++;
  }

  std::sort(separate_points.begin(), separate_points.end());

  double err = one_norm_error(x, A, y, r, theta_map);
  double err_new = one_norm_error(x_new, A, y, r, theta_map);

  if(err_new < err) {
    err = err_new;
    x = x_new;
  }

  // update x
  for(auto& pt : separate_points) {
    double a = pt.first;
    int theta_idx = pt.second;
    Mat x_tmp = x + a * p;
    double err_tmp = one_norm_error(x_tmp, A, y, r, theta_map);
    if(err_tmp < err) {
      err = err_tmp;
      x = x_tmp;
    }
  }

  // from x update theta
  hash_map_if x_map = get_x_map(theta_map, x);
  for(auto it = x_map.begin(); it != x_map.end();) {
    if(fabs(it->second) < EPSILON) {
      it = x_map.erase(it);
    } else {
      it++;
    }
  }
  get_theta_map_and_x(x_map, theta_map, x);
  show_theta_map(theta_map);
  printf("one_norm_line_search end ====\n");
}

void test_one_norm_line_search() {
  double r = 0.1;
  hash_map_ii theta_map;
  theta_map[0] = 1;
  theta_map[1] = 1;
  theta_map[2] = 1;
  Mat A = random_matrix(3, 6);
  Mat y = random_matrix(3, 1);
  Mat x = Mat::zeros(theta_map.size(), 1, CV_64F);
  Mat x_new = QP_solution(A, y, r, theta_map);
  one_norm_line_search(A, y, r, theta_map, x, x_new);
  show_matrix("x", x);
  for(auto& it : theta_map) {
    printf("theta: %d\n", it.second);
  }
}

// 4a
int check_nonzero_opt_condition(Mat x, Mat A, Mat y, double r, hash_map_ii theta_map) {
  int x_idx = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    double df = partial_differential(x, A, y, theta_map, idx);
    double vx = x.at<double>(x_idx);
    printf("nonzero total df: %f df: %f r*sign(vx): %f\n", fabs(df + r * sign(vx)), df, r * sign(vx));
    x_idx++;
  }
  x_idx = 0;
  for(auto& it : theta_map) {
    int idx = it.first;
    double df = partial_differential(x, A, y, theta_map, idx);
    double vx = x.at<double>(x_idx);
    // printf("nonzero total df: %f df: %f r*sign(vx): %f\n", fabs(df + r * sign(vx)), df, r * sign(vx));
    if(fabs(df + r * sign(vx)) >= EPSILON) {
      return 0;
    }
    x_idx++;
  }
  return 1;
}

// 4b
int check_zero_opt_condition(Mat x, Mat A, Mat y, double r, hash_map_ii theta_map) {
  for(int Ai = 0; Ai < A.size().width; Ai++) {
    if(theta_map.find(Ai) == theta_map.end()) {
      double df = partial_differential(x, A, y, theta_map, Ai);
      if(fabs(df) > r+EPSILON) {
        printf("Ai: %d df: %f\n", Ai, df);
        return 0;
      }
    }
  }
  return 1;
}

int test_read_image(int argc, char** argv ) {
  if ( argc != 2 )
  {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }

  Mat image;
  image = imread( argv[1], 1 );

  if ( !image.data )
  {
    printf("No image data \n");
    return -1;
  }

  Mat gray = rgb_to_gray(image);

  int block_size = 300;
  std::vector<Mat> blocks = get_blocks(block_size, 10, gray);
  namedWindow("Display Image", WINDOW_AUTOSIZE );

  printf("ch: %d\n", image.channels());

  for(int i = 0; i < 10; i++) {
    imshow("Display Image", blocks[i]);
    waitKey(100);
  }
  waitKey(0);
  return 0;
}

hash_map_if feature_sign_search(Mat A, Mat y, double r) {
  Mat x; 
  hash_map_ii theta_map;
  int i = 0;
  do {
    pick_theta_map(x, A, y, r, theta_map);
    printf("pick_theta_map start\n");
    show_theta_map(theta_map);
    printf("pick_theta_map end\n");
    do {
      Mat x_new = QP_solution(A, y, r, theta_map);
      one_norm_line_search(A, y, r, theta_map, x, x_new);
      //show_theta_map(theta_map);
      //show_matrix("x", x);
      //double err = one_norm_error(x, A, y, r, theta_map);
      //double QP_err = QP_error(x, A, y, r, theta_map);
      //printf("err: %f QP_err: %f\n", err, QP_err);
    } while(!check_nonzero_opt_condition(x, A, y, r, theta_map));
    i++;
    printf("iiiiiiiiiiiii: %d\n", i);
  } while(!check_zero_opt_condition(x, A, y, r, theta_map));
  hash_map_if x_map = get_x_map(theta_map, x);
  return x_map; 
}

void test_feature_sign_search() {
  double r = 0.1;
  Mat A = random_matrix(8, 15);
  Mat y = random_matrix(8, 1);
  hash_map_if x_map = feature_sign_search(A, y, r);
}

void test_inverse() {
  Mat A = random_matrix(3, 3);
  for(int i = 0; i < 3; i++) {
    A.at<double>(0, i) = 0;
  }
  show_matrix("A*inv(A)*A", A*A.inv(DECOMP_SVD)*A);
  show_matrix("A", A);
}

int main() {
  //test_QP_solution();
  //test_one_norm_line_search();
  //test_partial_differential();
  //test_inverse();
  test_feature_sign_search();
}

