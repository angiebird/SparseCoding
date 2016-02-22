#include <opencv2/opencv.hpp>
#include <tr1/unordered_map>
using namespace cv;
typedef std::tr1::unordered_map<int, int> hash_map_ii;
typedef std::tr1::unordered_map<int, double> hash_map_if;

Mat multiply(const Mat& A, const hash_map_if& x_map);
Mat random_matrix(int rows, int cols);
double partial_differential(hash_map_if x_map, Mat A, Mat y, int x_idx);
double partial_differential_ref(hash_map_if x_map, Mat A, Mat y, int x_idx);
