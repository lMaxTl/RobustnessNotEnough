# Make PyTorch use the CUDA stack it was built against. Sourced after the
# virtualenv is activated, by every script here.
#
# HoreKa's default environment puts a system cuDNN on LD_LIBRARY_PATH, and the
# dynamic loader searches LD_LIBRARY_PATH *before* the RUNPATH that points at
# the copy the torch wheel ships. So torch loads the system library and dies:
#
#   RuntimeError: cuDNN version incompatibility: PyTorch was compiled against
#   (9, 10, 2) but found runtime version (9, 5, 1).
#
# sbatch exports the submitting environment into the job (--export=ALL is the
# default), so this follows the login shell onto every compute node - it is not
# only a login-node problem.
#
# Prepending the wheels' own NVIDIA libraries fixes the search order. All of
# them, not just cuDNN: the same argument applies to cuBLAS and NCCL, and a
# half-bundled stack is a worse place to be than either consistent one.

_re_nvidia_libs="$(python -c 'import pathlib, sysconfig; print(":".join(sorted(str(p) for p in (pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia").glob("*/lib") if p.is_dir())))' 2>/dev/null || true)"
if [[ -n "${_re_nvidia_libs:-}" ]]; then
    export LD_LIBRARY_PATH="${_re_nvidia_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset _re_nvidia_libs
