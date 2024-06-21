# Fhelipe
Fhelipe is a tensor compiler with automatic data packing for Fully Homomorphic Encryption.

Fully Homomorphic Encryption (FHE) enables computing on encrypted data, letting clients securely offload computation to untrusted servers.
While enticing, FHE is extremely hard to program, because FHE schemes expose a low-level interface that prevents abstraction and composition.

The Fhelipe FHE compiler addresses this:
Fhelipe exposes a easy-to-use tensor programming interface, reducing code size by 10×–48×,
and is the first compiler to match or exceed the performance of large hand-optimized FHE applications, like deep neural networks.

For more details, check out the [Fhelipe paper].

## Get the repository
1. Clone the repository: `git clone git@github.com:fhelipe-compiler/fhelipe.git`
2. Initialize submodules: `git submodule update --init --recursive`

## Run in Docker
The easiest way to run Fhelipe is in [Docker]:

1. Build the Fhelipe Docker image: `docker build . --tag fhelipe`
2. Run it in a container: `docker run -ti fhelipe:latest`

That's all! You don't need to install anything manually.

## Manual install
If you want to run Fhelipe locally (without Docker), the process is a bit more involved.

### Install dependencies
In our setup, we use:
- Ubuntu 22.04
- clang-15 (we use C++20).
- Go 1.19
- CMake 3.21
- ninja 1.7.2
- Python 3.8

On Ubuntu 22.04, you can install most of this using
```bash
sudo apt-get install build-essential clang-15 golang cmake ninja-build python3-venv scons
```
On older versions, you need to use PPAs to obtain recent-enough versions of these packages.

### Setup the Python environment
1. Set up a virtual environment:
    1. Create it: `python3 -m venv fhenv`.
    2. Activate it: `source fhenv/bin/activate`.
        Again, you can add this to your `.bashrc` (`.envrc`).
2. Install the frontend: `pip install -e frontend` (`frontend` is a relative
   path).

### Build the compiler
In `./backend`:
1. `scons lib`: build external libraries.
2. `scons deps`: build the submodules.
3. `scons -j16 --release`: build the compiler.

For development builds, run `scons -j16` without `--release`.

## Compiling applications
The scripts put the compiler output in `~/fhelipe_experiments` and `~/fhelipe_tables`.

Here's an example for running ResNet-20:
1. Compile:
    ```bash
    python scripts/compile.py --program condor_fhelipe_resnet
    ```
2. Set up shared data (the weights):
    ```bash
    python scripts/in_shared.py --program condor_fhelipe_resnet
    ```
3. Run. The output will be in the `out_unenc` folder.
    * Unencrypted:
        ```bash
        python scripts/run.py --program condor_fhelipe_resnet
        ```
    * Encrypted (using the Lattigo FHE library):
        ```bash
        python scripts/run.py --program condor_fhelipe_resnet --lattigo
        ```
        This will encrypt the inputs, run the compiled binary, and decrypt the outputs.


## Running the Tests
- Backend tests: `./backend/build/tests`
- Python frontend tests: `python -m unittest discover -s frontend`
- Integration tests: `python -m unittest`.
  Adding `-k` lets you run a single test (e.g., `python -m unittest -k "TestLogReg.test_5_it"`),
  leaving the frontend environment in `.testing-tmp`; then, you can debug the
  backend by using the printed commands.

You can also run all tests (as done by continuous integration) using
`./scripts/run_ci_tests.sh`.

[Fhelipe paper]: https://dl.acm.org/doi/10.1145/3656382

[gtest]: https://github.com/google/googletest
[gflags]: https://github.com/gflags/gflags
[glog]: https://github.com/google/glog

[docker]: https://docs.docker.com/manuals/
[rust]: https://www.rust-lang.org/tools/install

