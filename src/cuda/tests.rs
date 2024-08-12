use super::*;

#[test]
fn test_simple_kernel_wrapper() {
    unsafe {
        println!("TESTING CUDA KERNEL WRAPPER");
        bindings::testKernelWrapper();
    }
}
