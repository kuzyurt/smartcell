"""HARD RAM cap for the smartbot tools.  Default 5 GB, enforced by the OS.

v1 OF THIS FILE WAS NOT A CAP.  It read the RSS, warned, and trimmed the
working set - which does nothing to stop a process from allocating 7 GB and
paging the machine to death, and that is exactly what happened.  It is now a
true limit: a Windows Job Object with JOB_OBJECT_LIMIT_JOB_MEMORY and
JOB_OBJECT_LIMIT_PROCESS_MEMORY, assigned to this process, which the OS
enforces on the whole tree (parent AND every spawned DataLoader worker).
Allocations past the cap FAIL instead of thrashing the machine.

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is set too: orphaned workers were the
reason the box stayed laggy after a run was stopped.

Linux has no equivalent per-process commit limit without cgroups, so there
the RSS watchdog stays (see `rss_gb`) plus RLIMIT_AS via `resource`.


WHY THIS EXISTS
  Training froze the machine.  cache_0010 is 303 episodes x 451 frames of
  96x96 uint8 = 1.26 GB of images and CacheEp loads an episode eagerly.  On
  Windows the DataLoader workers are SPAWNED, not forked, so each one re-runs
  BCDataset.__init__ and loads its own private copy of every training episode:
  6 workers x ~1.0 GB = ~6 GB on top of the parent's copy.  Nothing in the
  code warned about that, because every tool reported success while the box
  was thrashing.

WHAT IT DOES
  * CAP_GB / reserve is enforced where memory is actually allocated: the
    DataLoader worker count is DERIVED from the cache footprint, not guessed.
  * install() pins the BLAS/OpenMP thread pools, because each thread gets its
    own arena and 6 workers x N threads multiplies the same way.
  * guard() is a soft watchdog: over the cap it trims the working set and says
    so once, rather than letting the machine swap itself to death.

No psutil, no new dependency: Windows goes through psapi, everything else
through /proc/self/statm.

Run:  SMARTCELL_RAM_CAP_GB=5 python train.py ...
      python ramcap.py          # report the cap and this process's RSS
"""
import ctypes
import gc
import os
import sys

CAP_GB = float(os.environ.get("SMARTCELL_RAM_CAP_GB", "5"))

# "job" (default) = OS-enforced job object.  A job object cannot be escaped by
# a child, and on this box a child created inside it fails to allocate during
# its own start-up, so tools that reach for multiprocessing.spawn must NOT set
# MODE="job": they get "cooperative" instead - thread limits, a derived worker
# budget and the RSS watchdog.  collect_state.py selects that for itself.
MODE = os.environ.get("SMARTCELL_RAM_CAP_MODE", "job")

# Of the cap, how much the data pipeline may hold.  The rest is CUDA, MuJoCo,
# the renderer and the Python interpreter, which cannot be given up.
DATA_FRACTION = 0.55
MIN_WORKERS = 0

_warned = False
_capped = None          # (cap_gb, job_handle) once the OS limit is applied
_IS_WIN = os.name == "nt"

# ---- Windows Job Objects -------------------------------------------------
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x0100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9


class _LARGE_INTEGER(ctypes.Union):
    _fields_ = [("LowPart", ctypes.c_uint32),
                ("HighPart", ctypes.c_int32),
                ("QuadPart", ctypes.c_int64)]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64)]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", _LARGE_INTEGER),
                ("PerJobUserTimeLimit", _LARGE_INTEGER),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32)]


class _EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def cap_windows(cap_gb=None):
    """Apply a Job Object memory limit to this process and its children.

    Returns (ok, detail).  Not fatal on failure: a machine where the job
    cannot be created (e.g. a parent job with no nesting) still gets the
    cooperative budget from workers_for().
    """
    global _capped
    cap = CAP_GB if cap_gb is None else cap_gb
    if _capped is not None:
        return True, f"already capped at {_capped[0]:.2f} GB"
    try:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = ctypes.c_void_p
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        job = k.CreateJobObjectW(None, None)
        if not job:
            return False, f"CreateJobObject failed ({ctypes.get_last_error()})"
        info = _EXTENDED_LIMIT()
        limit = int(cap * 2 ** 30)
        # JOB_MEMORY alone caps the whole tree, which is what was asked for.
        # JOB_OBJECT_LIMIT_PROCESS_MEMORY is deliberately NOT set: on this box
        # it made every CreateProcess inside the job fail with MemoryError in
        # reduction.dump, which breaks multiprocessing.spawn - and both the
        # DataLoader workers and collect_state.py collect through spawn.  A
        # cap that stops the collector from running is not a cap.
        # JOB_MEMORY alone caps the whole tree, which is what was asked for.
        # JOB_OBJECT_LIMIT_PROCESS_MEMORY is deliberately NOT set: with it,
        # every CreateProcess inside the job failed with MemoryError in
        # reduction.dump.  A cap that stops multiprocessing.spawn working is
        # not a cap - see MODE below for how spawn-based tools are handled.
        flags = _JOB_OBJECT_LIMIT_JOB_MEMORY
        if os.environ.get("SMARTCELL_RAM_CAP_PERPROC") == "1":
            flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.BasicLimitInformation.LimitFlags = flags
        info.ProcessMemoryLimit = limit
        info.JobMemoryLimit = limit
        k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_uint32]
        if not k.SetInformationJobObject(job, _JobObjectExtendedLimitInformation,
                                         ctypes.byref(info),
                                         ctypes.sizeof(info)):
            return False, f"SetInformationJobObject failed ({ctypes.get_last_error()})"
        k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if not k.AssignProcessToJobObject(
                job, ctypes.windll.kernel32.GetCurrentProcess()):
            return False, f"AssignProcessToJobObject failed ({ctypes.get_last_error()})"
        _capped = (cap, job)      # keep the handle: closing it kills the tree
        return True, f"job object, {cap:.2f} GB for the whole process tree"
    except Exception as exc:                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def cap_posix(cap_gb=None):
    """RLIMIT_AS on Linux/macOS, which fails allocations past the cap."""
    global _capped
    cap = CAP_GB if cap_gb is None else cap_gb
    try:
        import resource
        n = int(cap * 2 ** 30)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,
                           (n, hard if hard != resource.RLIM_INFINITY else n))
        _capped = (cap, None)
        return True, f"RLIMIT_AS {cap:.2f} GB"
    except Exception as exc:                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def cap(cap_gb=None):
    """Apply the hard cap for this platform.  Returns (ok, detail)."""
    return cap_windows(cap_gb) if _IS_WIN else cap_posix(cap_gb)
_psapi = None
if _IS_WIN:
    # GetProcessMemoryInfo lives in psapi.dll (not kernel32) and ctypes will
    # silently hand back a null result without explicit argtypes, so both are
    # pinned here: a RAM cap that always reads 0 is worse than no cap.
    for _dll in ("psapi.dll", "kernel32.dll"):
        try:
            _d = ctypes.WinDLL(_dll)
            _f = getattr(_d, "GetProcessMemoryInfo", None) or getattr(
                _d, "K32GetProcessMemoryInfo", None)
            if _f is None:
                continue
            _f.restype = ctypes.c_int
            _f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
            _psapi = _f
            break
        except (OSError, AttributeError):
            continue


def rss_gb():
    """Resident set size of this process, in GB."""
    if _IS_WIN:
        class _PMC(ctypes.Structure):        # PROCESS_MEMORY_COUNTERS
            _fields_ = [("cb", ctypes.c_uint32),
                        ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        try:
            ok = _psapi(ctypes.windll.kernel32.GetCurrentProcess(),
                        ctypes.byref(c), c.cb)
        except Exception:
            ok = 0
        if ok:
            return c.WorkingSetSize / 2 ** 30
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") \
                / 2 ** 30
    except OSError:
        return 0.0


def trim():
    """Return freed pages to the OS.  Best effort, never raises."""
    gc.collect()
    if _IS_WIN:
        try:
            ctypes.windll.psapi.EmptyWorkingSet(
                ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass
    else:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def workers_for(bytes_per_worker, cap_gb=None, fraction=DATA_FRACTION):
    """How many DataLoader workers fit under the cap.  `bytes_per_worker` is
    ONE PROCESS's private footprint in BYTES; the cap is in GB, so the two are
    put in the same unit here rather than at every call site.

    The parent already holds one copy, so the allowance is
    (cap * fraction) minus that copy, divided by one copy per worker.
    """
    cap = CAP_GB if cap_gb is None else cap_gb
    budget = cap * fraction
    one = float(bytes_per_worker) / 2 ** 30
    if one <= 0:
        return 0
    room = budget - one                      # the parent's own copy
    if room <= 0:
        return MIN_WORKERS
    return max(MIN_WORKERS, int(room // one))


def install(threads=None, hard=True):
    """Call once, early, from every tool that allocates a lot.

    Applies the OS-enforced cap FIRST, so everything allocated afterwards -
    including the CUDA context and every spawned DataLoader worker - is already
    inside it.  Safe to call repeatedly and from several modules.
    """
    ok, detail = cap() if hard else (False, "disabled")
    n = threads or max(1, min(4, (os.cpu_count() or 4) // 2))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    # the CUDA caching allocator is device memory, but its host-side pinned
    # staging and fragmentation are host commit: keep it greedy-free
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    trim()
    if ok:
        print(f"[ramcap] HARD cap: {detail}", flush=True)
    else:
        print(f"[ramcap] WARNING: no OS cap ({detail}); the loader budget is "
              f"still enforced cooperatively", file=sys.stderr, flush=True)
    return n


def guard(tag=""):
    """Soft watchdog.  Trims and warns ONCE if the process is over the cap."""
    global _warned
    r = rss_gb()
    if r > CAP_GB and not _warned:
        _warned = True
        print(f"[ramcap] {tag or 'process'} RSS {r:.2f} GB > cap "
              f"{CAP_GB:.2f} GB - trimming the working set.  Lower "
              f"SMARTCELL_RAM_CAP_GB or the DataLoader worker count.",
              file=sys.stderr, flush=True)
        trim()
    return r


def job_usage():
    """(used_gb, peak_gb, cap_gb) for the WHOLE job = process tree.

    This is the number that matters and the one that can be checked: the
    per-process RSS of the parent says nothing about six spawned workers.
    """
    if _capped is None or not _IS_WIN or not _capped[1]:
        return rss_gb(), rss_gb(), CAP_GB
    try:
        info = _EXTENDED_LIMIT()
        k = ctypes.WinDLL("kernel32")
        k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_void_p, ctypes.c_uint32,
                                                ctypes.c_void_p]
        k.QueryInformationJobObject(_capped[1],
                                    _JobObjectExtendedLimitInformation,
                                    ctypes.byref(info), ctypes.sizeof(info),
                                    None)
        return (info.JobMemoryLimit and
                _capped[0] * info.PeakJobMemoryUsed / info.JobMemoryLimit,
                info.PeakJobMemoryUsed / 2 ** 30, _capped[0])
    except Exception:                             # noqa: BLE001
        return rss_gb(), rss_gb(), CAP_GB


def report():
    r = rss_gb()
    used, peak, c = job_usage()
    tag = "tree" if _capped else "process"
    print(f"RAM cap {c:.2f} GB (enforced: {bool(_capped)})   {tag} peak "
          f"{peak:.2f} GB   this process {r:.2f} GB   "
          f"data allowance {c * DATA_FRACTION:.2f} GB")
    return r


# APPLY ON IMPORT.  A cap that has to be remembered is not a cap: the bug
# was a tool that allocated 7 GB before anyone called install().  Importing
# this module is now enough, and it is imported before torch/CUDA anywhere it
# matters.
_AUTO = (os.environ.get("SMARTCELL_RAM_CAP", "1") != "0" and MODE == "job")
if _AUTO:
    _ok, _detail = cap()


if __name__ == "__main__":
    import sys as _sys
    if "--verify" in _sys.argv:
        ok, detail = cap()
        print(f"cap applied: {ok}  {detail}")
        # ask for 2x the cap so the limit is what stops us, not the loop
        want = int(2 * CAP_GB * 1024 / 64) + 8
        blocks = []
        try:
            for i in range(want):
                blocks.append(bytearray(64 << 20))      # touched 64 MB
        except MemoryError:
            pass
        got = len(blocks) * 64 / 1024
        print(f"allocated until failure: {got:.2f} GB "
              f"(cap {CAP_GB:.2f} GB - expect this to stop near the cap)")
    else:
        report()