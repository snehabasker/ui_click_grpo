#!/bin/bash
# A C/C++ compiler + make WITHOUT sudo, for Triton / vLLM / SGLang JIT. micromamba lives in ~/.local/bin, the compilers in
# ~/toolchain (conda-forge). If a system compiler is installed later (sudo apt install build-essential) nothing here conflicts:
# serve scripts only prepend ~/toolchain/bin when `gcc` is not already on PATH.
set -euo pipefail
mkdir -p "$HOME/.local/bin"
if [ ! -x "$HOME/.local/bin/micromamba" ]; then
  # the minimal Ubuntu image has no bzip2 binary, so unpack with Python's tarfile instead of `tar -xj`
  curl -L --fail -s -o /tmp/micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest
  "$HOME/venvs/bench/bin/python" -c "import tarfile; tarfile.open('/tmp/micromamba.tar.bz2').extract('bin/micromamba', '$HOME/.local')"
  chmod +x "$HOME/.local/bin/micromamba"
fi
"$HOME/.local/bin/micromamba" --version
if [ ! -x "$HOME/toolchain/bin/gcc" ]; then
  "$HOME/.local/bin/micromamba" create -y -q -p "$HOME/toolchain" -c conda-forge gcc gxx make binutils
fi
export PATH="$HOME/toolchain/bin:$PATH"
gcc --version | head -1
g++ --version | head -1
make --version | head -1
# smoke test: compile and run a C program, then link against the WSL CUDA driver library the way Triton will
printf '#include <stdio.h>\nint main(){puts("cc ok");return 0;}\n' > /tmp/cc_test.c
gcc /tmp/cc_test.c -o /tmp/cc_test && /tmp/cc_test
ls -l /usr/lib/wsl/lib/libcuda.so* 2>/dev/null | head -2
ldconfig -p | grep -m1 libcuda.so || echo "libcuda not in ldconfig cache (Triton also checks /usr/lib/wsl/lib)"
echo "TOOLCHAIN_OK"
