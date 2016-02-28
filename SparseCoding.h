#include <opencv2/opencv.hpp>
#include <tr1/unordered_map>

using namespace cv;

#define EPSILON 0.01 

typedef std::tr1::unordered_map<int, int> hash_map_ii;
typedef std::tr1::unordered_map<int, double> hash_map_if;

void show_theta_map(const hash_map_ii& theta_map);
void show_x_map(const hash_map_if& x_map);

int sign(double x);

Mat multiply(const Mat& A, const hash_map_if& x_map);

Mat random_matrix(int rows, int cols);

double partial_differential(hash_map_if x_map, Mat A, Mat y, int x_idx);

double partial_differential_ref(hash_map_if x_map, Mat A, Mat y, int x_idx);

hash_map_ii get_theta_map(const hash_map_if& x_map);

void pick_theta_map(hash_map_if& x_map, const Mat& A, const Mat& y, const double r, hash_map_ii& theta_map);

int check_nonzero_opt_condition(const hash_map_if& x_map, const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map);
int check_zero_opt_condition(const hash_map_if& x_map, const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map);

hash_map_if QP_solution(const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map);

double QP_error(const hash_map_if& x_map, const Mat& A, const Mat& y, const double& r, const hash_map_ii& theta_map);

hash_map_if QP_partial_differential(const hash_map_if& x_map, const Mat& A, const Mat& y, const double r, const hash_map_ii& theta_map);

hash_map_if interpolate(const hash_map_if& x_map, const hash_map_if& x_map_new, const double a);

double one_norm_line_search(const Mat& A, const Mat& y, const double r, const hash_map_if& x_map, const hash_map_if& x_map_new, hash_map_if& x_map_best);
