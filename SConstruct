#env=Environment(CXXFLAGS="-std=c++11 -g -fsanitize=address")
#env['CC'] = 'clang'
#env['CXX'] = 'clang++'

env=Environment(CXXFLAGS="-std=c++11 -g")

libs = ['libopencv_calib3d',
'libopencv_core',
'libopencv_features2d',
'libopencv_flann',
'libopencv_highgui',
'libopencv_imgcodecs',
'libopencv_imgproc',
'libopencv_ml',
'libopencv_objdetect',
'libopencv_photo',
'libopencv_shape',
'libopencv_stitching',
'libopencv_superres',
'libopencv_video',
'libopencv_videostab',
'libgtest',
'libgtest_main']

env.Program('SparseCoding.cc', LIBS=libs, LIBPATH=['/usr/local/lib/', '/usr/lib'])
env.Program(['SparseCoding_test.cc', 'SparseCoding.cc'], LIBS=libs, LIBPATH=['/usr/local/lib/', '/usr/lib'])

