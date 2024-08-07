pub mod tensor;
pub mod storage;
mod tensor_arithmetic;
mod storage_arithmetic;
pub mod init;
mod utils;

use num_traits::{cast::FromPrimitive, Num, NumCast};
pub trait Primitive: Copy + NumCast + Num + PartialOrd<Self> + Clone + FromPrimitive {}
impl Primitive for u8 {}
impl Primitive for f32 {}
impl Primitive for f64 {}
impl Primitive for i32 {}
impl Primitive for i64 {}