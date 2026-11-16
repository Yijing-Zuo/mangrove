#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
what="${1:-cuda}"
build_emu() {
    g++ -O2 -std=c++17 -shared -fPIC -pthread -DMG_EMU -x c++ mg_kernels.cu -o libmangrove_emu.so
    echo "built libmangrove_emu.so"
}
build_cuda() {
    NVCC="${CUDA_HOME:+$CUDA_HOME/bin/}nvcc"
    ARCHS="${MG_ARCHS:-80 90}"
    GEN=""
    for a in $ARCHS; do GEN="$GEN -gencode arch=compute_${a},code=sm_${a}"; done
    LIBDIR=""
    if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib" ]; then LIBDIR="-L$CUDA_HOME/lib"; fi
    "$NVCC" -O3 -shared -Xcompiler -fPIC -fmad=false $GEN $LIBDIR -o libmangrove.so mg_kernels.cu mg_nodemap.cu
    echo "built libmangrove.so with $("$NVCC" --version | tail -1) for archs: $ARCHS"
}
case "$what" in
    emu) build_emu ;;
    cuda) build_cuda ;;
    all) build_emu; build_cuda ;;
    *) echo "usage: $0 [cuda|emu|all]"; exit 2 ;;
esac
