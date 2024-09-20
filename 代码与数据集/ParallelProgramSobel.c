// 环境配置
// 1. 安装OpenMPI
//     sudo apt-get update
//     sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
// 2. 配置图像处理库
//     sudo apt-get install libopencv-dev
// 3. 编译命令 
//     mpic++ -o version2_huge version2_huge.c `pkg-config --cflags --libs opencv4`
// 4. 运行命令 
//     mpiexec -n 4 ./version2_huge //以4个进程运行
// 5. 数据集文件
//     需要在此文件同等目录下建立一个out文件夹用于存储处理后的图像
//     需要一个名为dataset的文件夹存储数据集中的图片
//     需要一个名为image_list.txt的文件存储数据集中每一张图片的路径用于后续对每一张图片进行分析
//     ls /home/sanshi/PROJECT/dataset/train/*.jpg > /home/sanshi/PROJECT/output/test1/image_list.txt
//     此指令将数据集中的图像路径存入txt文件中，此指令中的路径为我的路径，请自行修改
//     /home/sanshi/PROJECT/output/test1/image_list.txt


#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
using namespace cv;
using namespace std;

/*sobel算法对图像卷积操作*/
void sobel_edge_detection(Mat &input, Mat &output) {
    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int sumX = 0, sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = input.at<uchar>(y + i, x + j);
                    sumX += pixel * gx[i + 1][j + 1];
                    sumY += pixel * gy[i + 1][j + 1];
                }
            }
            int magnitude = sqrt(sumX * sumX + sumY * sumY);
            output.at<uchar>(y, x) = (magnitude > 255) ? 255 : magnitude;
        }
    }
}

/*并行多进程的方式处理图像*/
void parallel_process(int rank, int size, const vector<string>& image_files) {
    for (const string& file : image_files) {
        Mat input, gray;
        if (rank == 0) {
            input = imread(file, IMREAD_COLOR);
            if (input.empty()) {
                printf("Could not open or find the image: %s\n", file.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            cvtColor(input, gray, COLOR_BGR2GRAY);
        }

        int rows, cols;
        if (rank == 0) {
            rows = gray.rows;
            cols = gray.cols;
        }
        
        /*广播图像信息*/
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int local_rows = rows / size;
        Mat local_gray(local_rows, cols, CV_8UC1);
        Mat local_output(local_rows, cols, CV_8UC1);

        /*分散图像块*/
        MPI_Scatter(gray.data, local_rows * cols, MPI_UNSIGNED_CHAR, local_gray.data, local_rows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        /*算法处理*/
        sobel_edge_detection(local_gray, local_output);

        Mat output;
        if (rank == 0) {
            output = Mat(rows, cols, CV_8UC1);
        }

        /*广集图像块*/
        MPI_Gather(local_output.data, local_rows * cols, MPI_UNSIGNED_CHAR, output.data, local_rows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            
            // string output_file = "output/test_small/small_parallel/" + file.substr(file.find_last_of("/\\") + 1);
            string output_file = "output/test_huge/huge_parallel/" + file.substr(file.find_last_of("/\\") + 1);
            imwrite(output_file, output);
        }
    }
}

/*串行的方式处理图像*/
void serial_process(const vector<string>& image_files) {
    for (const string& file : image_files) {
        Mat input = imread(file, IMREAD_COLOR);
        if (input.empty()) {
            printf("Could not open or find the image: %s\n", file.c_str());
            continue;
        }

        Mat gray;
        cvtColor(input, gray, COLOR_BGR2GRAY);
        Mat output(gray.rows, gray.cols, CV_8UC1);

        /*图像处理算法*/
        sobel_edge_detection(gray, output);

        // string output_file = "output/test_small/small_serial/" + file.substr(file.find_last_of("/\\") + 1);
        string output_file = "output/test_huge/huge_serial/" + file.substr(file.find_last_of("/\\") + 1);
        imwrite(output_file, output);
    }
}

/*读取图像文件名*/
vector<string> read_image_files(const string& filename) {
    vector<string> files;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        files.push_back(line);
    }
    return files;
}

/*主函数*/
int main(int argc, char** argv) {
    int rank, size;
    double serial_duration;
    double parallel_duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*创建数据结构存储文件名*/
    vector<string> image_files = read_image_files("output/test_huge/image_list.txt");

    /*串行方式并记录运行时间*/
    if (rank == 0) {
        int64 serial_start = getTickCount();

        serial_process(image_files);

        int64 serial_end = getTickCount();
        serial_duration = (serial_end - serial_start) / getTickFrequency();
        printf("串行条件下, 数据集运行时间： %f seconds\n", serial_duration);
    }

    /*同步各个进程*/
    MPI_Barrier(MPI_COMM_WORLD);

    /*并行方式并记录运行时间*/
    int64 parallel_start = getTickCount();
    parallel_process(rank, size, image_files);
    if (rank == 0) {
        int64 parallel_end = getTickCount();
        parallel_duration = (parallel_end - parallel_start) / getTickFrequency();
        printf("并行 %d 核条件下, 数据集运行时间： %f seconds\n", size, parallel_duration);

        double speedup = serial_duration / parallel_duration;
        double efficiency = speedup / size;
        printf("加速比：%f ，效率：%f \n", speedup, efficiency);
    }

    MPI_Finalize();
    return 0;
}
