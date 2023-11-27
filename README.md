# Autograd.rs

- dynamic shapes only
- alternative, candle style, tensor only holds a reference counted ref to the data/storage, you pass around tensor not rc<refcell<tensor>>