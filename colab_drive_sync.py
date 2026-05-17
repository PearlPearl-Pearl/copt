"""
colab_drive_sync.py
───────────────────
Two-cell Colab helper that mounts Google Drive and keeps important
outputs synced automatically while a long training run is in progress.

USAGE
──────
# ── Cell 1: mount Drive & configure ────────────────────────────────────────
exec(open("colab_drive_sync.py").read())
mount_drive()          # opens the Drive auth prompt once

# ── Cell 2: start background sync ──────────────────────────────────────────
start_sync(interval=120)   # sync every 2 minutes

# ── then run your training ──────────────────────────────────────────────────
# !bash run_mis_comparison.sh   (or any other script)

# ── Cell (optional): force an immediate sync at any time ───────────────────
do_sync()

# ── Cell (optional): stop the background thread and do a final flush ───────
stop_sync()
"""

import os, glob, shutil, threading, time

# ── configure these paths ────────────────────────────────────────────────────

# Root of the cloned repo inside Colab
LOCAL_ROOT = "/content/copt"

# Where to save inside your Drive — will be created automatically
DRIVE_ROOT = "/content/drive/MyDrive/AIMS/copt_runs"

# File patterns to sync (relative to LOCAL_ROOT, ** = recursive)
SYNC_PATTERNS = [
    "*.log",
    "*.png",
    "results/**/metrics.csv",
    "results/**/probs_test.pt",
    "results/**/*.ckpt",
]

# ── internal state ────────────────────────────────────────────────────────────
_stop_event  = threading.Event()
_sync_thread = None


# ── public API ────────────────────────────────────────────────────────────────

def mount_drive(mount_point="/content/drive"):
    """Mount Google Drive (opens the auth prompt the first time)."""
    from google.colab import drive as _drive
    _drive.mount(mount_point)
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    print(f"Drive mounted.  Saving outputs to:\n  {DRIVE_ROOT}")


def do_sync(verbose=True):
    """Copy all matching files from LOCAL_ROOT → DRIVE_ROOT right now."""
    copied = 0
    for pattern in SYNC_PATTERNS:
        for src in glob.glob(os.path.join(LOCAL_ROOT, pattern), recursive=True):
            rel = os.path.relpath(src, LOCAL_ROOT)
            dst = os.path.join(DRIVE_ROOT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # only copy if src is newer than dst (or dst doesn't exist)
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                shutil.copy2(src, dst)
                copied += 1
    if verbose:
        ts = time.strftime("%H:%M:%S")
        print(f"[sync {ts}]  {copied} file(s) updated in Drive.")
    return copied


def start_sync(interval=120):
    """
    Start a background thread that calls do_sync() every `interval` seconds.
    Safe to call multiple times — only one thread runs at a time.
    """
    global _sync_thread, _stop_event

    if _sync_thread is not None and _sync_thread.is_alive():
        print(f"Sync already running (every {interval}s). Call stop_sync() first to restart.")
        return

    _stop_event = threading.Event()

    def _loop():
        while not _stop_event.wait(interval):
            try:
                do_sync()
            except Exception as e:
                print(f"[sync] warning: {e}")
        # final flush when stopped
        try:
            do_sync()
            print("[sync] Final flush complete.")
        except Exception as e:
            print(f"[sync] Final flush warning: {e}")

    _sync_thread = threading.Thread(target=_loop, daemon=True)
    _sync_thread.start()
    print(f"Background sync started (every {interval}s).")
    print(f"  LOCAL  →  {LOCAL_ROOT}")
    print(f"  DRIVE  →  {DRIVE_ROOT}")
    print("Call do_sync() for an immediate save, stop_sync() to stop.")


def stop_sync():
    """Stop the background thread and do one final sync."""
    global _sync_thread
    if _sync_thread is None or not _sync_thread.is_alive():
        print("No sync thread running.")
        return
    _stop_event.set()
    _sync_thread.join(timeout=30)
    _sync_thread = None
    print("Sync stopped.")
