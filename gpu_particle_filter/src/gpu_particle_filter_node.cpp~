
#include <gpu_particle_filter/gpu_particle_filter.h>
#include <gpu_particle_filter/particle_filter_kernel.h>

ParticleFilterGPU::ParticleFilterGPU() {

    std::cout << "GPU TEST"  << "\n";

    char a[N] = "HELLO  \0\0\0\0\0\0";
    int b[N] = {
        15, 10, 6, 0, -11,
        1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    
    test_cuda(a, b);
    printf("%s\n", a);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "gpu_particle_filter");
    ParticleFilterGPU pfg;
    ros::spin();
    return 0;
}

