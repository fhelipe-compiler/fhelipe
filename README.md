# Fhelipe
Fhelipe is a tensor compiler with automatic data packing for Fully Homomorphic Encryption.

Fully Homomorphic Encryption (FHE) enables computing on encrypted data, letting clients securely offload computation to untrusted servers.
While enticing, FHE is extremely hard to program, because FHE schemes expose a low-level interface that prevents abstraction and composition.

The Fhelipe FHE compiler addresses this:
Fhelipe exposes a easy-to-use tensor programming interface, reducing code size by 10×–48×,
and is the first compiler to match or exceed the performance of large hand-optimized FHE applications, like deep neural networks.

For more details, check out the [Fhelipe paper][paper].

## Get the repository
1. Clone the repository: `git clone git@github.com:fhelipe-compiler/fhelipe.git`
2. Initialize submodules: `git submodule update --init --recursive`

## Run in Docker
The easiest way to run Fhelipe is in [Docker].
On an x86 machine:

1. Build the Fhelipe Docker image: `docker build . --tag fhelipe`
2. Run it in a container: `docker run -ti fhelipe:latest`

That's all! You don't need to install anything manually.

Unfortunately, this fails on non-x86 machines because a few dependencies are x86-only.
On Apple Silicon, you might be able to get this work using my [pre-built x86 image] and [Rosetta],
but we don't officially support this and we won't be able to help with troubleshooting.

[pre-built x86 image]: https://people.csail.mit.edu/alexalex/fhelipe_docker.tar.gz
[Rosetta]: https://support.apple.com/en-us/102527

## Manual install
If you want to run Fhelipe locally (without Docker), the process is a bit more involved.

### Install dependencies
In our setup, we use:
- Ubuntu 22.04
- clang-15 (we use C++20).
- Go 1.19
- CMake 3.21
- ninja 1.7.2
- Python 3.10

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

## Results
Up-to-date performance of Fhelipe on CraterLake against CHET and Manual baselines:

|             | Fhelipe [ms] | Manual [ms] | CHET [ms] | vs Manual | vs CHET |
|-------------|--------------|-------------|-----------|-----------|---------|
| resnet      | 243.7        | 247.0       | 591.8     | 1.0×      | 2.4×    |
| rnn         | 463.1        | 489.7       | 2974.1    | 1.1×      | 6.4×    |
| logreg      | 143.6        | 1741.2      | 5163.6    | 12.1×     | 36.0×   |
| LoLa-MNIST  | 0.3          | 0.9         | 88.9      | 3.4×      | 318.8×  |
| **gmean**   |              |             |           | 2.6×      | 20.6×   |
| FFT         | 241.0        |             | 257.5     |           | 1.1×    |
| TTM         | 30.3         |             | 8105.0    |           | 267.9×  |
| MTTKRP      | 64.8         |             | 8105.6    |           | 125.1×  |
| **gmean**   |              |             |           |           | 33.0×   |

Differences from Table.4 in the [paper] are due to:
- [Fix][layouts-fix] of a layouts bug affecting TTM and MTTKRP
- Various small performance improvements


[paper]: https://dl.acm.org/doi/10.1145/3656382

[gtest]: https://github.com/google/googletest
[gflags]: https://github.com/gflags/gflags
[glog]: https://github.com/google/glog

[docker]: https://docs.docker.com/manuals/
[layouts-fix]: https://github.com/fhelipe-compiler/fhelipe/commit/8329cda980ad5a3307727adf63e0db6f15c215b8
