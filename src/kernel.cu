#include <stdio.h>
#include <iostream>
#include <io.h>
#include <curand_kernel.h>

#include "geo_list.cuh"
#include "sphere.cuh"
#include "cam.cuh"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ 
vec3 random_in_unit_sphere(curandState* local_rand_state) 
{
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ 
vec3 colour(const ray& r, geometry** world, curandState* local_rand_state) 
{
    ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f;
            cur_ray = ray(rec.p, target - rec.p);
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ 
void render_init(int max_x, int max_y, curandState* rand_state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ 
void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, 
    geometry** world, curandState* rand_state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += colour(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__
void create_world(geometry** d_list, geometry** d_world, camera** d_camera, float a) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new geo_list(d_list, 2);
        *d_camera = new camera(a);
    }
}

__global__ 
void free_world(geometry** d_list, geometry** d_world, camera** d_camera) 
{
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
    delete* d_camera;
}

int main() 
{
    //480p
    int width = 640;
    int height = 480;
    float aspect_ratio = (float)width / height;
    //Recursion depth of 25 (i.e. 25 ray bounces)
    int depth = 25;

    /*
    My RTX 2060 has:
    - 30 SMs
    - 64 CUDA cores per SM
    - supports 1024 threads per SM (i.e. 1024 threads per block)
    - warp = how threads are packaged: into packs of 32, block are split into warps
    -->best practice, i.e. maximum efficiency, have block sizes be multiples of warp size (32*x)
    - N / blocksize = number of blocks * blocksize ensures I have N threads of execution.
    - (N + blockSize - 1) / blockSize ensures I create the correct number of blocks for when N isn't a mutiple of blockSize.
    - N = 640*480 = 9600 * 32
    --> Make each block the max size of 1024 (32 warps), 300 blocks of 32 warps maximizes GPU usage
    */
    int bx = 16;
    int by = 16;

    int N = width * height;
    size_t fb_size = 3 * N * sizeof(vec3);

    // allocate frameBuffer
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, N * sizeof(curandState)));

    // Camera setup
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    
    //World setup
    geometry** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(geometry*)));
    geometry** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(geometry*)));
    
    create_world<<<1, 1>>>(d_list, d_world, d_camera, aspect_ratio);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Render our buffer
    dim3 blocks(width / bx, height / by);
    dim3 threads(bx, by);
    render_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, width, height, depth, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Output FB as Image
    FILE* fd;
    if (fopen_s(&fd, "out/test_render_4.ppm", "w") != 0 || fd == NULL)
        exit(errno);

    if (_dup2(_fileno(fd), _fileno(stdout)) != 0)
        exit(errno);

    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            float r = fb[pixel_index].r();
            float g = fb[pixel_index].g();
            float b = fb[pixel_index].b();
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    fclose(fd);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));  
    
    cudaDeviceReset();
}