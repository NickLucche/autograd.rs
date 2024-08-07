pub mod utils;
use std::cell::RefCell;
use std::rc::Rc;

pub type SharedPtr<T> = Rc<RefCell<T>>;

pub fn shared_ptr_new<T>(x: T) -> SharedPtr<T> {
    Rc::new(RefCell::new(x))
}