# nanochatpp

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Dr4k3z/nanochatpp?utm_source=oss&utm_medium=github&utm_campaign=Dr4k3z%2Fnanochatpp&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

C++23 project using Eigen and GoogleTest.

## Dependencies

- CMake 3.31 or newer, Ninja, and a C++23 compiler (CLion bundles CMake and Ninja).
- [Eigen 5.0.0](https://libeigen.gitlab.io/releases/), pinned as a Git submodule in `external/eigen`.
- [GoogleTest 1.18.0](https://github.com/google/googletest/releases/tag/v1.18.0), fetched by CMake when `BUILD_TESTING=ON` (the default).

After cloning:

```sh
git submodule update --init --recursive
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The first configure needs network access to fetch GoogleTest. Use the `release`
preset for Release builds. Set `-DBUILD_TESTING=OFF` to build only the application.
If Ninja is not on your shell PATH, pass `-DCMAKE_MAKE_PROGRAM=/path/to/ninja`
at configure time; CLion's bundled Ninja is under its `Contents/bin/ninja` directory.

## CLion

Open this directory as a CMake project, reload CMake, and enable the `debug` or
`release` preset in **Settings > Build, Execution, Deployment > CMake**.
Use the bundled toolchain's CMake and Ninja. CLion discovers `nanochatpp_tests`
and its individual GoogleTest cases; run them from the test gutter or Run menu.

In **Settings > Editor > Code Style > C/C++**, enable ClangFormat and use the
project `.clang-format`. In **Settings > Editor > Inspections > C/C++ > General >
Clang-Tidy**, enable the inspection and use the project `.clang-tidy` configuration.
CLion includes clang-tidy, so a separate Homebrew installation is unnecessary.
Local IDE settings and build directories are ignored by Git.

## Style and analysis

`.clang-format` and `.clang-tidy` are copied from the sibling VecDB/cpp-sdk
projects: LLVM-based formatting, four-space indentation, 120 columns, left-aligned
pointers/references; camelCase variables, PascalCase classes, snake_case functions.

```sh
clang-format -i main.cpp tests/eigen_test.cpp
clang-tidy -p build/debug main.cpp tests/eigen_test.cpp
```

For command-line analysis, put clang-tidy on PATH or use CLion's bundled executable
under `Contents/bin/clang`. CMake exports `compile_commands.json` in each build
directory. Format and analyze project sources rather than vendored dependencies.

The bundled clang-tidy currently fails standalone analysis on this machine because
its Clang 21 headers/toolchain do not match Apple's Clang 17 SDK setup (including
ARM NEON intrinsics). Builds, tests, and formatting pass; a complete clang-tidy
run needs matching LLVM resource headers and macOS SDK paths.

## Tensor API and ownership

`NanoChat::Tensor<T>` is a shared-storage handle with its own shape, strides, and
offset. Copies and layout transformations retain storage ownership, so slices
remain valid after their source is destroyed or reassigned. Include
`headers/tensor/tensor.h`.

```cpp
NanoChat::Tensor<float> tensor(2, 3, 4); // zero-initialized
tensor[1, 2, 3].get() = 7.0f;
auto matrix = tensor[1];                // shape {3, 4}, shared storage
auto vector = matrix[2];                // shape {4}, shared storage
auto transposed = tensor.transpose(0, 2);
auto permuted = tensor.permute(std::array<std::size_t, 3>{2, 0, 1});
auto reshaped = tensor.reshape({6, 4});
```

Metadata accessors are `get_shape()`, `get_strides()`, `get_ndim()`,
`get_numel()`, and `is_contiguous()`. `get()` requires a rank-zero result.
`sum`/`+` and `product`/`*` perform elementwise arithmetic on equal shapes,
including strided inputs, and return independent mutable results.

| Operation | Storage behavior |
| --- | --- |
| Copy, indexing, transpose, permute | Share storage; each handle owns its metadata |
| `reshape(shape)` on contiguous input | Share storage and preserve the slice offset |
| `reshape(shape)` on strided input | Copy elements in logical row-major order, then reshape |
| `make_contiguous()` | Share already-contiguous storage; copy strided storage |
| `clone()` | Always copy into independent, contiguous storage |

`transpose(a, b)` swaps two axes. `permute(axes)` requires each axis exactly once.
`reshape` requires an unchanged element count; use `reshape({})` for a scalar
when the count is one. `make_contiguous()` promises a layout, not independent
ownership. Use `clone()` whenever writes must not affect other handles.

Const indexing and layout operations return `Tensor<const T>`, preventing writes
through those results without introducing a separate view API. Read-only handles
still observe writes through existing mutable aliases; they do not freeze shared
storage. `clone()` and arithmetic return new mutable tensors even from read-only
inputs. Copying a `const Tensor<T>` directly retains the ordinary shared-handle
copy semantics; use `Tensor<const T>` for a persistently read-only handle.

Lvalue `get()` returns a scalar reference. Mutable rvalue `get()` returns a
storage-owning `ScalarReference<T>`: `tensor[i].get() = value` remains supported,
and saving the proxy keeps its storage alive. It converts to a scalar value,
never to a raw reference. Read-only/const rvalue `get()` returns a scalar value.
Do not retain a raw reference obtained from a named handle beyond the lifetime
of all handles owning that storage. Temporary transformations remain supported
because their results retain ownership.

## Validation and errors

`headers/tensor/errors.h` defines `TensorError`, `TensorException`, and
`NanoChat::TensorValidation`. Domain failures report a code via `.code()` and a
message via `.what()`. Standard allocation failures retain standard exceptions.

```cpp
TensorValidation::require([&] { return optional_check(); },
                          TensorError::InvalidShape, "Optional validation failed");
TensorValidation::require_invariant(condition, TensorError::InvalidShape,
                                    "Required invariant failed");
```

`require_invariant` always runs, including tensor shape, indexing, overflow, and
layout checks. Optional `require` checks are enabled by default and controlled by
`NANOCHAT_ENABLE_VALIDATION`, independently of `NDEBUG`.

```sh
cmake --preset release -DNANOCHAT_ENABLE_VALIDATION=OFF
cmake --build --preset release
```

The `release-unchecked` preset sets the option OFF and is available in CLion.
It disables optional checks, not invariants. Outside CMake, define
`NANOCHAT_ENABLE_VALIDATION=0` consistently across translation units. The
`nanochatpp_tensor` interface target propagates the configured definition.

Disabled optional predicates are never invoked. Boolean arguments and lambda
captures still evaluate normally, so wrap expensive optional checks in a
predicate. Tests exercise optional checks and always-on invariants in both modes.
