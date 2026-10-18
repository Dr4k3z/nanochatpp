# nanochatpp

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

## Tensor indexing

Tensor rank is dynamic. Indexing consumes leading axes and returns a borrowing,
mutable view (read-only for a const tensor):

```cpp
auto tensor = NanoChat::Tensor<>::create(2, 3, 4);
auto matrix = tensor[1].matrix();      // 3 x 4, row-major Eigen map
auto vector = tensor[1, 2].vector();   // length 4, column vector
vector.setConstant(5.0f);             // modifies the tensor
tensor[1, 2, 3].get() = 7.0f;          // scalar reference
// tensor[1][2][3].get() is equivalent
```

Use `.transpose()` on the vector map for a row vector expression. Views and maps
must not outlive the tensor or survive moving/assigning it. Indexing a temporary
tensor is disabled. Invalid operations throw `TensorException`, carrying a `TensorError` code and
a descriptive message. These checks run in both Debug and Release builds.
Factories and arithmetic return tensors directly.

`NanoChat::TensorView<Element>` is defined in `headers/tensor/tensor_view.h` and can be
used directly in function signatures. Const tensors return `TensorView<const Scalar>`.

## Tensor layouts and ownership

Both `Tensor` and `TensorView` provide `transpose`, `reshape`, `is_contiguous`,
and `contiguous`. Tensor forwards layout operations to TensorView, so indexing
and layout rules have one implementation. Include `headers/tensor/tensor.h` when
materializing a view into a tensor.

```cpp
auto tensor = NanoChat::Tensor<float>::create(2, 3, 4);
auto transposed = tensor.transpose(0, 2); // shape {4, 3, 2}; no element copies
auto reshaped = tensor.reshape(6, 4);    // shape {6, 4}; no element copies
auto owned = transposed.contiguous();   // independent, row-major Tensor<float>
NanoChat::Tensor<float> alsoOwned(transposed); // explicit view-to-owner conversion
auto flat = owned.reshape(24);          // view; keep owned alive
```

`transpose(a, b)` swaps two axes; `transpose()` without axes requires rank two.
Indexing and Eigen `.matrix()`/`.vector()` maps honor the resulting strides.
`reshape` preserves row-major logical order and currently requires a contiguous
layout and the same element count. It accepts a shape or nonnegative variadic
dimensions; `reshape({})` produces a scalar view when there is one element.
Non-contiguous layouts must first be materialized with `contiguous()`; reshape
never silently copies. `view.data()` also requires contiguity because a span
cannot represent strided elements.

Views own their shape/stride metadata, so intermediate views can be temporary.
Their elements still belong to the source tensor: keep that tensor alive and
unmoved. Creating borrowing views directly from temporary tensors is disabled.
View operations cost O(rank) time and metadata storage, with no element copies.
Const tensors produce read-only views. `contiguous()` on a view or an lvalue
tensor copies into an independent owner; on an rvalue tensor it moves storage.
Materialization costs O(numel) for contiguous sources and O(numel * rank) for
strided sources. Invalid layouts/shapes throw `TensorException`; allocations may throw
standard allocation exceptions such as `std::bad_alloc`.

## Validation and errors

Operations start with visible preconditions: use `require(condition, code, message)`
for one check, or `validate_input(input).require(...).require(...)` for a pipeline.
Each predicate receives a const reference to the input and runs immediately, in
order. The first failing requirement throws its own `TensorException`; subsequent
predicates and the operation body do not run. Successful validation does not
transform or copy the input. Lvalue inputs are borrowed; temporary inputs are
owned by the validator, so a stored validator must not outlive its borrowed input.

```cpp
using namespace NanoChat;
using TensorValidation::validate_input;

void process(const std::vector<int>& dimensions) {
    validate_input(dimensions)
        .require([](const auto& values) {
            return std::all_of(values.begin(), values.end(), [](int x) { return x > 0; });
        }, TensorError::InvalidDimension, "All dimensions must be positive")
        .require([](const auto& values) {
            return std::accumulate(values.begin(), values.end(), 0) == 10;
        }, TensorError::InvalidShape, "Dimensions must sum to 10");

    // Actual operation starts here. Both requirements passed.
}
```

Include `<algorithm>` and `<numeric>` for this example. Predicates can also be
named functions or reusable lambdas. The validator stores no registry or list of
rules and uses no type erasure. Checks are explicit, not implicit in the name
`validate_input`. Tensor operations use the same mechanism internally. Only the
nontrivial shape-overflow and borrowed-storage checks retain dedicated helpers.
Stride calculation remains in `tensor_layout.h`.

Both types use one exception-based API. `Tensor::create`, arithmetic, and
`contiguous` return values directly (there is no `std::expected` wrapper or
`.value()` step). Domain errors throw `NanoChat::TensorException` with a stable
`TensorError` code and a descriptive `what()` message. Allocation failures retain
standard C++ exceptions; they are not converted to domain errors.

```cpp
try {
    auto tensor = NanoChat::Tensor<float>::create(2, 3);
    auto view = tensor.transpose();
    auto sum = view + view; // owning tensor; owners and views can be mixed too
    auto reshaped = sum.reshape(6);
} catch (const NanoChat::TensorException& error) {
    // error.code(): InvalidDimension, InvalidShape, IncompatibleShapes,
    // IndexOutOfBounds, SizeOverflow, or NonContiguous
    std::cerr << error.what() << '\n';
}
```

Both owners and views expose `get`, `matrix`, `vector`, indexing, layout
operations, and elementwise arithmetic. Templates select owner/view and const
access statically; strided inputs are evaluated in logical order. Arithmetic
allocates the result once, borrows the right operand, and uses a contiguous fast
path when available. Views still require their source tensor to remain alive.

## Production validation switch

Validation is enabled by default, independently of `NDEBUG` or build type. Use
CMake's `NANOCHAT_ENABLE_VALIDATION` option to disable it for an application:

```sh
cmake --preset release -DNANOCHAT_ENABLE_VALIDATION=OFF
cmake --build --preset release
```

Alternatively, select the `release-unchecked` preset in CLion or run:

```sh
cmake --preset release-unchecked
cmake --build --preset release-unchecked
ctest --preset release-unchecked
```

The `nanochatpp_tensor` interface target propagates the setting to consumers.
Outside CMake, compile with `-DNANOCHAT_ENABLE_VALIDATION=0` (or `=1` to enable).
Use the same setting in every translation unit linked into a program to avoid
inconsistent inline/template definitions. Existing Debug and Release presets keep
validation on unless explicitly overridden in their CMake cache.

Pipeline predicates are not invoked when disabled. Use the lazy single-check
overload to skip evaluating expensive conditions too:

```cpp
TensorValidation::require([&] { return expensive_check(input); },
                          TensorError::InvalidShape, "Invalid input");
```

`require(expensive_check(input), ...)` still evaluates its boolean argument before
the function is called, even with validation disabled. Ordinary argument and
lambda-capture construction also still occurs. Internal checks use lazy predicates
or compile-time guards; index-validation and storage-bound loops are omitted.
Necessary shape/stride calculations and layout-dependent execution paths remain.
This option controls NanoChat validation, not Eigen or standard-library assertions.

Unchecked builds require valid inputs. Wrong shapes, indices, axes, or storage
layouts can then cause undefined behavior rather than exceptions. The error-contract
suite always runs with checks enabled; separate test executables exercise both
modes and verify that disabled predicates do not run.
