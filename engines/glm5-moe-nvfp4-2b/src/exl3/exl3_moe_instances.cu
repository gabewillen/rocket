#include "exl3_moe_kernel.cuh"
#include "comp_units/exl3_moe_instances.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_instances[] = {
    exl3_moe_kernel_k0_n128(), exl3_moe_kernel_k0_n256(),
    exl3_moe_kernel_k1_n128(), exl3_moe_kernel_k1_n256(),
    exl3_moe_kernel_k2_n128(), exl3_moe_kernel_k2_n256(),
    exl3_moe_kernel_k3_n128(), exl3_moe_kernel_k3_n256(),
    exl3_moe_kernel_k4_n128(), exl3_moe_kernel_k4_n256(),
    exl3_moe_kernel_k5_n128(), exl3_moe_kernel_k5_n256(),
    exl3_moe_kernel_k6_n128(), exl3_moe_kernel_k6_n256(),
    exl3_moe_kernel_k7_n128(), exl3_moe_kernel_k7_n256(),
    exl3_moe_kernel_k8_n128(), exl3_moe_kernel_k8_n256(),
};
