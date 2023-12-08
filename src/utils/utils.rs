use std::cell::RefCell;
use std::rc::Rc;

pub type SharedPtr<T> = Rc<RefCell<T>>;

pub fn SharedPtrNew<T>(x: T) -> SharedPtr<T> {
    Rc::new(RefCell::new(x))
}
