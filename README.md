# Autograd.rs 🚀

Autograd.rs is a basic Deep Learning library with (you guessed it) automatic differentiation support, built on top of the `rust-ndarray` crate to execute computations. 
The project is made for educational purposes, in particular to learn about Rust and explore its applications within the machine learning landscape!

The library is heavily inspired by PyTorch and we often refer to its source code for the more intricate details. The API should also look very similar, as I don't think an interface can ever feel much better than what torch built :3

## About Autograd.rs  🧠

Autograd builds a computational graph dynamically/lazily, as you execute through the network's operations, much like PyTorch.
Unlike PyTorch, where every tensor operation inherently triggers the construction of the graph (unless you use `torch.no_grad()`), Autograd.rs reserves this activity for operations via the Layers API only. I think this limits the flexibility of the library, but it makes it easier to maintain and discourages complex architectures (which we won't build anyway). 

There is no `unsafe` code *directly* written/used in this library.

## Quickstart ⚡
### Installation 🛠️

```bash
git clone https://github.com/NickLucche/autograd.rs/
cd autograd.rs
cargo build
```
To use the Autograd.rs library in your Rust files, remember to add it to your `Cargo.toml`:
```toml
[dependencies]
autograd_rs = { version = "0.1", path = "[path to your local Autograd.rs directory]" }
```
Finally, import and use it in your Rust scripts:
```Rust
extern crate autograd_rs;
use autograd_rs::*;
```

### Usage 💡

```Rust
TODO
```
You can find more in [/examples](`/examples/`).


## Components ⚙️

- `src/autograd/` **Automatic Differentiation**: gradients are accumulated onto parameters and Tensors that require it with a single `backward()`. While computations are run in parallel, the graph is traversed on a single core. 
- `src/tensor/` **Tensor**: a Tensor wraps an ndarray dynamically shaped `Array` and maintains a single reference to its data when copied; this should feel more like a "Python object" when passed around (or a shared_ptr), as we're intentionally avoiding any CoW mechanism. It is still subject to Rust borrow checkers rules, but these are mostly checked at runtime (I know, not 100% rustacean here).
- `src/operators/` **Operators**: to add a new operator, one must simply implement its `forward` and `backward` methods through the `operators::operators::Operator` trait.    
- `src/nn/` **NN API**: high-level API for creating models, which are just a way to organize layers, which in turn are a way to bundle operators and its parameters.
- `src/utils`: things like model serialization. You can export the model architecture to PyTorch code, providing an easy way to visualize your model using tools like Netron.

## Currently Supported Operations 💠

Only a small number of operations are currently supported right now:

- Linear (a.k.a Dense/FullyConnected)
- MatMul
- Conv2D (with Im2Col/Fold+Unfold)
- ReLU
- Sigmoid
- MSELoss

But more will follow! If you're eager to add more operations, you can chip in by implementing the forward and backward functions for the Operator trait in `src/operators/operators.rs` and respective Layer class in `src/nn/layers.rs`. 

## Future Plans / TODOs 📝

Some of the vital features I'd like to include ASAP are:

- GPU Support.
- MNIST example, a bit out of fashion but still coming.
- API work: should feel as simple and clean as possible.
- Model serialization.
- API for a `no_grad` mode.


## Deep Dive 🐠

TODO 

## Contributing 🤝

I appreciate any feedback - be it bug reports, feature suggestions, test scenarios, or any kind of constructive inputs, especially on Rust best practices. If you're interested in helping out, here's how you can get started:

1. Fork it (https://github.com/NickLucche/autograd.rs/forks/new)
2. Commit your changes (`git commit -am 'Add some fooBar'`)
3. Push to the branch (`git push origin feature/fooBar`)
4. Create a new Pull Request

## License 📖

This project is licensed under the [Apache License 2.0] - see the [LICENSE](LICENSE) file for details.

Please note that Autograd.rs is currently under active development, so things can break!