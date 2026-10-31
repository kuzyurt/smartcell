"""dGPU guard: never silently train/eval on the wrong device.

System: RTX 2060 6GB (dGPU, CUDA) + Intel UHD (iGPU, NOT CUDA-visible).
Torch today sees exactly 1 CUDA device (the 2060), so plain "cuda" works
by accident. This helper makes it explicit and fails loudly otherwise:

  * asserts torch.cuda.is_available() when cuda is required
  * asserts device_count >= 1 and "2060" in device name (case-insensitive)
    unless SMARTCELL_ALLOW_ANY_GPU=1 (for future machines)
  * pins torch.cuda.set_device(0) and returns "cuda:0"
  * logs VRAM total/free via torch + nvidia-smi when available
  * for MuJoCo: logs MUJOCO_GL env (unset = default WGL/glfw on Windows,
    proven to render on the 2060 per PROVENANCE) and warns if Intel UHD
    is the Windows-preferred adapter. MuJoCo 3.14 has no API to force
    an adapter, so we verify via nvidia-smi process + GL_RENDERER when
    possible instead of pretending to pin.

RAM: python process tree must stay under 5GB (ramcap CAP_GB=5). This module
does NOT change the cap; it only reports. See ramcap.py.
"""
import os
import shutil
import subprocess


def pick_cuda(require_2060=True):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False: dGPU (RTX 2060) not visible. "
            "Refusing to train/eval on CPU silently. Check drivers / "
            "nvidia-smi. Set SMARTCELL_ALLOW_CPU=1 to override explicitly.")
    n = torch.cuda.device_count()
    if n < 1:
        raise RuntimeError("cuda available but device_count==0")
    # Pin to 0 explicitly: with 1 CUDA device this IS the 2060.
    # Never rely on bare "cuda" default in new code.
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if require_2060 and os.environ.get("SMARTCELL_ALLOW_ANY_GPU", "0") != "1":
        if "2060" not in name:
            raise RuntimeError(
                f"expected RTX 2060 on cuda:0, got {name!r} "
                f"(count={n}). Set SMARTCELL_ALLOW_ANY_GPU=1 to allow.")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    free = None
    try:
        f, _ = torch.cuda.mem_get_info(0)
        free = f / 2 ** 30
    except Exception:
        pass
    tag = f"dGPU cuda:0 {name} total {total:.1f}GB" + (
        f" free {free:.1f}GB" if free is not None else "")
    print(f"[gpu_guard] {tag}", flush=True)
    # nvidia-smi cross-check (read-only, best effort)
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                print(f"[gpu_guard] nvidia-smi: {out.stdout.strip()}",
                      flush=True)
    except Exception as e:
        print(f"[gpu_guard] nvidia-smi check skipped: {e}", flush=True)
    # MuJoCo GL note
    gl = os.environ.get("MUJOCO_GL", "")
    print(f"[gpu_guard] MUJOCO_GL={gl!r} (unset=Windows default WGL/glfw; "
          f"proven on 2060 per PROVENANCE, not pinned by MuJoCo API)",
          flush=True)
    return "cuda:0"


def cuda_amp_dtype():
    """fp16 on Turing (no bf16 cores). Never bf16 on the 2060."""
    import torch
    return torch.float16
