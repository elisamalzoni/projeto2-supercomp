#include <iostream>
#include <stdio.h>
#include <stdlib.h>
// #include "vec3.h"
// #include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"


#define NX 200
#define NY 100
#define SIZE NX*NY*3*sizeof(int)


__global__ void generate(int *A, hitable *world){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    

    float u = float(i) / float(NX);
    float v = float(j) / float(NY);

    //color
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    vec3 p = r.point_at_parameter(2.0);

    vec3 col;
    hit_record rec;
  
    if (world->hit(r, 0.0, MAXFLOAT, rec)) {
      col = 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    } else {
      vec3 unit_direction = unit_vector(r.direction());
      float t = 0.5*(unit_direction.y()+1.0);
      col = (1.0-t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }

    int ir = int(255.99*col[0]);
    int ig = int(255.99*col[1]);
    int ib = int(255.99*col[2]);
    A[(i*NY + j)*3] = ir;
    A[(i*NY + j)*3 + 1] = ig;
    A[(i*NY + j)*3 + 2] = ib;
}

int main() {

    dim3 dimGrid(ceil(NX/(float)16), ceil(NY/(float)16));
    dim3 dimBlock(16, 16);

    hitable *list[2];
    list[0] = new sphere(vec3(0, 0, -1), 0.5);
    list[1] = new sphere(vec3(0, -100.5, -1), 100);
    hitable *world = new hitable_list(list, 2);

    int *cpuA;
    int *gpuA;
    cpuA = (int *)malloc(SIZE);
    cudaMalloc((void **)&gpuA,SIZE);
    generate<<<dimGrid, dimBlock>>>(gpuA, world);
    cudaMemcpy(cpuA, gpuA, SIZE, cudaMemcpyDeviceToHost);
    cudaFree(gpuA);

    std::cout << "P3\n" << NX << " " << NY << "\n255\n";
    for (int j = NY-1; j >= 0; j--){
        for(int i = 0; i < NX; i++){
            std::cout << cpuA[(i*NY + j)*3] << " " << cpuA[(i*NY + j)*3 + 1] << " " << cpuA[(i*NY + j)*3 + 2] << "\n";
        }
    }
    delete[] cpuA;
}