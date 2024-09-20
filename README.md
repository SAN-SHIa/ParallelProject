# 并行方法优化图像边缘化处理——以Celebrity Face Image Dataset数据集为例
---
## 项目简介

该项目旨在利用并行计算技术（如MPI）优化图像边缘检测算法，以加速大规模图像数据的处理。我们使用了Sobel算法，分别对串行和并行方法进行了实现，并通过比较它们的运行时间、加速比与效率来评估不同核数下的性能表现。
![Uploading 9c0de4509f2d91eb2d771cb0f76fbf4.png…]()

## 使用技术

- **编程语言**: C++
- **并行计算框架**: MPI（OpenMPI）
- **图像处理库**: OpenCV
- **数据集**: Celebrity Face Image Dataset (Kaggle)

## 安装步骤

1. 更新系统并安装OpenMPI：
   ```bash
   sudo apt-get update
   sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
   ```

2. 安装OpenCV库：
   ```bash
   sudo apt-get install libopencv-dev
   ```

3. 克隆本项目代码：
   ```bash
   git clone <your-project-url>
   cd <your-project-folder>
   ```

4. 编译代码：
   ```bash
   mpic++ -o sobel_parallel sobel_parallel.c `pkg-config --cflags --libs opencv4`
   ```

## 运行说明

1. 确保数据集和图像列表文件正确存放：
   - 将图片存放在`dataset`文件夹下。
   - 将图片路径写入`image_list.txt`文件中。

2. 以4个进程运行并行程序：
   ```bash
   mpiexec -n 4 ./sobel_parallel
   ```

3. 处理后的图像会存放在`output`文件夹下。

## 性能分析

- 串行方法在单核上完成图像的Sobel边缘检测。
- 并行方法将图像分块，利用多核进行加速计算。
- 加速比和效率随着核心数量的增加而变化，详见本项目报告中的数据分析章节。

## 数据集

- **名称**: Celebrity Face Image Dataset
- **来源**: Kaggle
- **大小**: 53MB
- **特点**: 多样性高，适用于面部识别和分类任务的研究。

---
