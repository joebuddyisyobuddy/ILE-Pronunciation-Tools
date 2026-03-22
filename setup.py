#!/usr/bin/env python3
"""
ILE Pronunciation Coach — One-Click Setup

Usage:
    python setup.py                     # auto-detect extension ID
    python setup.py EXTENSION_ID        # provide ID manually
    python setup.py --full              # download full-precision model (1.26 GB)
    python setup.py --uninstall         # remove native host registration

What happens:
    1. Check Python version (3.10+)
    2. Install dependencies (librosa, onnxruntime, numpy)
    3. Download the phoneme model from HuggingFace (~318 MB)
    4. Find your browser extension automatically
    5. Register the native messaging host
    6. Smoke test

Takes about 2 minutes on a typical connection.
"""

import sys
import os
import platform
import subprocess
import json
import shutil
import stat
import re
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

HOST_NAME = "com.ile.phoneme_host"
EXTENSION_NAME = "ILE Phonetic Coach"

DEPENDENCIES = ["librosa", "onnxruntime", "numpy"]

# HuggingFace direct download URLs (onnx-community pre-built models)
HF_BASE = "https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/resolve/main"
MODELS = {
    "int8":  {"url": f"{HF_BASE}/onnx/model_int8.onnx",  "size": "318 MB",  "file": "model_int8.onnx"},
    "full":  {"url": f"{HF_BASE}/onnx/model.onnx",        "size": "1.26 GB", "file": "model.onnx"},
    "q4":    {"url": f"{HF_BASE}/onnx/model_q4.onnx",     "size": "242 MB",  "file": "model_q4.onnx"},
    "fp16":  {"url": f"{HF_BASE}/onnx/model_fp16.onnx",   "size": "632 MB",  "file": "model_fp16.onnx"},
}
DEFAULT_MODEL = "int8"

MODEL_MIN_SIZE_MB = 50  # anything smaller is not a real model
MIN_PYTHON = (3, 10)

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
HOST_SCRIPT = SCRIPT_DIR / "phoneme_host.py"

SYSTEM = platform.system()

# ─── Terminal output helpers ──────────────────────────────────────────────────

def supports_color():
    if SYSTEM == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = supports_color()

def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def green(t):  return c(t, "32")
def red(t):    return c(t, "31")
def yellow(t): return c(t, "33")
def cyan(t):   return c(t, "36")
def bold(t):   return c(t, "1")
def dim(t):    return c(t, "2")

def step(n, total, msg):
    print(f"\n  {cyan(f'[{n}/{total}]')} {bold(msg)}")

def ok(msg):
    print(f"         {green('✓')} {msg}")

def warn(msg):
    print(f"         {yellow('!')} {msg}")

def fail(msg):
    print(f"         {red('✗')} {msg}")

def info(msg):
    print(f"         {dim('→')} {msg}")

def fatal(msg):
    fail(msg)
    print(f"\n  {red('Setup failed.')} Fix the issue above and run setup.py again.\n")
    sys.exit(1)

# ─── pip helper ───────────────────────────────────────────────────────────────

def pip_install(packages):
    pip_cmd = [sys.executable, "-m", "pip", "install", "--quiet",
               "--disable-pip-version-check"] + packages
    try:
        subprocess.run(pip_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if "externally-managed" in (e.stderr or ""):
            warn("System Python detected, retrying with --break-system-packages")
            pip_cmd.insert(4, "--break-system-packages")
            try:
                subprocess.run(pip_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e2:
                fail("pip install failed:")
                print(e2.stderr)
                hint = "py -m pip" if SYSTEM == "Windows" else "python3 -m pip"
                fatal(f"Try: {hint} install {' '.join(packages)}")
        else:
            fail("pip install failed:")
            print(e.stderr)
            hint = "py -m pip" if SYSTEM == "Windows" else "python3 -m pip"
            fatal(f"Try: {hint} install {' '.join(packages)}")

# ─── Step 1: Python version ──────────────────────────────────────────────────

def check_python(n, total):
    step(n, total, "Checking Python version")
    v = sys.version_info
    info(f"Found Python {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < MIN_PYTHON:
        fatal(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
              f"Download from https://python.org/downloads/")
    ok(f"Python {v.major}.{v.minor} — good")

# ─── Step 2: Dependencies ────────────────────────────────────────────────────

def install_deps(n, total):
    step(n, total, "Installing dependencies")
    pip_install(DEPENDENCIES)
    for dep in DEPENDENCIES:
        ok(dep)

# ─── Step 3: Download model ──────────────────────────────────────────────────

def any_model_exists():
    """Check if any valid ONNX model exists in models/."""
    if not MODELS_DIR.exists():
        return None
    for variant in MODELS.values():
        path = MODELS_DIR / variant["file"]
        if path.exists() and path.stat().st_size > MODEL_MIN_SIZE_MB * 1024 * 1024:
            return path
    return None


def download_model(n, total, variant_key=DEFAULT_MODEL):
    step(n, total, "Downloading phoneme model")

    existing = any_model_exists()
    if existing:
        size_mb = existing.stat().st_size / (1024 * 1024)
        ok(f"Model already exists: {existing.name} ({size_mb:.0f} MB)")
        _check_vocab()
        return existing

    variant = MODELS[variant_key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / variant["file"]

    info(f"Downloading {variant['file']} ({variant['size']}) from HuggingFace...")
    info(f"Source: onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX")

    try:
        import urllib.request

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                bar_len = 25
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r         {dim('→')} {bar} {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(variant["url"], dest, reporthook=_progress)
        print()  # newline after progress bar

        size_mb = dest.stat().st_size / (1024 * 1024)
        if size_mb < MODEL_MIN_SIZE_MB:
            dest.unlink()
            fatal(f"Downloaded file is too small ({size_mb:.1f} MB). Download may have failed.")

        ok(f"Model downloaded: {variant['file']} ({size_mb:.0f} MB)")

    except Exception as e:
        if dest.exists():
            dest.unlink()
        fatal(f"Download failed: {e}\n"
              f"         Try downloading manually from:\n"
              f"         {variant['url']}\n"
              f"         and place it in the models/ folder.")

    _check_vocab()
    return dest


def _check_vocab():
    """Verify vocab.json exists and is valid."""
    vocab_path = MODELS_DIR / "vocab.json"
    if not vocab_path.exists():
        # Check if it's in the script directory instead
        alt_path = SCRIPT_DIR / "vocab.json"
        if alt_path.exists():
            shutil.copy2(alt_path, vocab_path)
            ok(f"Copied vocab.json to models/")
        else:
            warn("vocab.json not found — phoneme labels may not work")
            info("This file should be included in the repository")
            return

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        ok(f"Vocab loaded — {len(vocab)} phoneme tokens")
    except (json.JSONDecodeError, IOError) as e:
        warn(f"vocab.json may be corrupted: {e}")

# ─── Step 4: Extension ID ────────────────────────────────────────────────────

def get_browser_extension_dirs():
    home = Path.home()
    candidates = []
    if SYSTEM == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        candidates = [
            ("Chrome", local / "Google" / "Chrome" / "User Data"),
            ("Edge",   local / "Microsoft" / "Edge" / "User Data"),
            ("Brave",  local / "BraveSoftware" / "Brave-Browser" / "User Data"),
        ]
    elif SYSTEM == "Darwin":
        app_support = home / "Library" / "Application Support"
        candidates = [
            ("Chrome", app_support / "Google" / "Chrome"),
            ("Edge",   app_support / "Microsoft Edge"),
            ("Brave",  app_support / "BraveSoftware" / "Brave-Browser"),
        ]
    else:
        config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
        candidates = [
            ("Chrome", config / "google-chrome"),
            ("Edge",   config / "microsoft-edge"),
            ("Brave",  config / "BraveSoftware" / "Brave-Browser"),
        ]

    results = []
    for browser_name, user_data_dir in candidates:
        for profile in ["Default", "Profile 1", "Profile 2", "Profile 3"]:
            ext_dir = user_data_dir / profile / "Extensions"
            if ext_dir.is_dir():
                results.append((browser_name, profile, ext_dir))
    return results


def find_extension_id():
    browser_dirs = get_browser_extension_dirs()
    if not browser_dirs:
        return None, None

    found = []
    for browser_name, profile, ext_dir in browser_dirs:
        try:
            for ext_id_dir in ext_dir.iterdir():
                if not ext_id_dir.is_dir() or len(ext_id_dir.name) != 32:
                    continue
                for version_dir in ext_id_dir.iterdir():
                    manifest = version_dir / "manifest.json"
                    if manifest.is_file():
                        try:
                            with open(manifest, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            name = data.get("name", "").lower()
                            if "ile" in name and any(k in name for k in
                                    ["phonetic", "phoneme", "pronunciation"]):
                                found.append((ext_id_dir.name, browser_name, profile))
                        except (json.JSONDecodeError, IOError):
                            pass
                    break
        except PermissionError:
            pass

    if len(found) == 1:
        return found[0][0], found[0][1]
    elif len(found) > 1:
        warn("Found extension in multiple browsers/profiles:")
        for eid, browser, profile in found:
            info(f"  {browser} ({profile}): {eid}")
        info(f"Using: {found[0][0]} ({found[0][1]})")
        return found[0][0], found[0][1]
    return None, None


def resolve_extension_id(n, total, cli_arg=None):
    step(n, total, "Finding extension ID")

    if cli_arg:
        if re.match(r'^[a-z]{32}$', cli_arg):
            ok(f"Using provided ID: {cli_arg}")
            return cli_arg
        else:
            fatal(f"'{cli_arg}' doesn't look like a Chrome extension ID.\n"
                  "         Extension IDs are 32 lowercase letters, e.g.:\n"
                  "         mnnlgacdajodekhlfbiilihbicimfaga\n"
                  "         Find yours at chrome://extensions with Developer Mode on.")

    info("Scanning browser directories...")
    ext_id, browser = find_extension_id()

    if ext_id:
        ok(f"Found in {browser}: {ext_id}")
        return ext_id

    fail("Could not auto-detect extension ID")
    print()
    print(f"         {bold('To find your extension ID:')}")
    print(f"         1. Open {cyan('chrome://extensions')} in your browser")
    print(f"         2. Enable {bold('Developer mode')} (toggle in top right)")
    print(f"         3. Find {bold(EXTENSION_NAME)}")
    print(f"         4. Copy the ID (32 lowercase letters under the name)")
    print(f"         5. Re-run: {cyan(f'python setup.py YOUR_EXTENSION_ID')}")
    print()
    sys.exit(1)

# ─── Step 5: Register native host ────────────────────────────────────────────

def get_native_hosts_dir(browser_name=None):
    home = Path.home()
    if SYSTEM == "Windows":
        return SCRIPT_DIR
    elif SYSTEM == "Darwin":
        base = home / "Library" / "Application Support"
        dirs = {
            "Edge":  base / "Microsoft Edge" / "NativeMessagingHosts",
            "Brave": base / "BraveSoftware" / "Brave-Browser" / "NativeMessagingHosts",
        }
        return dirs.get(browser_name, base / "Google" / "Chrome" / "NativeMessagingHosts")
    else:
        config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
        dirs = {
            "Edge":  config / "microsoft-edge" / "NativeMessagingHosts",
            "Brave": config / "BraveSoftware" / "Brave-Browser" / "NativeMessagingHosts",
        }
        return dirs.get(browser_name, config / "google-chrome" / "NativeMessagingHosts")


def register_native_host(n, total, ext_id, browser_name=None):
    step(n, total, "Registering native messaging host")

    if SYSTEM == "Windows":
        launcher = SCRIPT_DIR / "phoneme_host.bat"
        python_path = Path(sys.executable).resolve()
        launcher.write_text(
            f'@echo off\r\n"{python_path}" "{HOST_SCRIPT}"\r\n',
            encoding="utf-8"
        )
        host_path = str(launcher)
        ok(f"Created {launcher.name}")
    else:
        launcher = SCRIPT_DIR / "phoneme_host.sh"
        python_path = shutil.which("python3") or shutil.which("python") or sys.executable
        launcher.write_text(
            f'#!/bin/bash\nexec "{python_path}" "{HOST_SCRIPT}"\n',
            encoding="utf-8"
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        host_path = str(launcher)
        ok(f"Created {launcher.name}")

    manifest = {
        "name": HOST_NAME,
        "description": "ILE Pronunciation Coach — local phoneme recognition via wav2vec2",
        "path": host_path,
        "type": "stdio",
        "allowed_origins": [f"chrome-extension://{ext_id}/"]
    }

    if SYSTEM == "Windows":
        manifest_path = SCRIPT_DIR / f"{HOST_NAME}.json"
    else:
        hosts_dir = get_native_hosts_dir(browser_name)
        hosts_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = hosts_dir / f"{HOST_NAME}.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    ok(f"Wrote manifest to {manifest_path}")

    # Register for other installed browsers
    if SYSTEM != "Windows":
        other_browsers = {"Chrome", "Edge", "Brave"} - {browser_name or "Chrome"}
        for other in other_browsers:
            other_dir = get_native_hosts_dir(other)
            if other_dir.parent.exists():
                other_dir.mkdir(parents=True, exist_ok=True)
                other_manifest = other_dir / f"{HOST_NAME}.json"
                with open(other_manifest, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                info(f"Also registered for {other}")

    if SYSTEM == "Windows":
        try:
            import winreg
            manifest_str = str(manifest_path)
            for reg_path in [
                rf"SOFTWARE\Google\Chrome\NativeMessagingHosts\{HOST_NAME}",
                rf"SOFTWARE\Microsoft\Edge\NativeMessagingHosts\{HOST_NAME}",
                rf"SOFTWARE\BraveSoftware\Brave-Browser\NativeMessagingHosts\{HOST_NAME}",
            ]:
                try:
                    key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, reg_path,
                                             0, winreg.KEY_SET_VALUE)
                    winreg.SetValueEx(key, "", 0, winreg.REG_SZ, manifest_str)
                    winreg.CloseKey(key)
                except OSError:
                    pass
            ok("Registry keys set for Chrome, Edge, Brave")
        except Exception as e:
            fatal(f"Failed to write registry key: {e}")

    ok("Native host registered")

# ─── Step 6: Smoke test ──────────────────────────────────────────────────────

def smoke_test(n, total):
    step(n, total, "Running smoke test")

    try:
        import numpy
        ok(f"numpy {numpy.__version__}")
    except ImportError:
        fail("numpy import failed")
        return False

    try:
        import onnxruntime
        ok(f"onnxruntime {onnxruntime.__version__}")
    except ImportError:
        fail("onnxruntime import failed")
        return False

    try:
        import librosa
        ok(f"librosa {librosa.__version__}")
    except ImportError:
        fail("librosa import failed")
        return False

    model_path = any_model_exists()
    if model_path:
        try:
            info("Loading model for inference test...")
            import onnxruntime as ort
            import numpy as np

            sess = ort.InferenceSession(str(model_path),
                                        providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            dummy = np.zeros((1, 16000), dtype=np.float32)
            result = sess.run(None, {input_name: dummy})
            ok(f"Inference test passed — output shape: {result[0].shape}")
        except Exception as e:
            warn(f"Inference test failed: {e}")
            info("The model may still work — try recording in the extension")
    else:
        warn("Skipping inference test (no model found)")

    return True

# ─── Uninstall ────────────────────────────────────────────────────────────────

def uninstall():
    print(f"\n  {bold('Uninstalling ILE Pronunciation Coach native host...')}\n")
    removed = False

    if SYSTEM != "Windows":
        for browser in ["Chrome", "Edge", "Brave"]:
            manifest = get_native_hosts_dir(browser) / f"{HOST_NAME}.json"
            if manifest.exists():
                manifest.unlink()
                ok(f"Removed {manifest}")
                removed = True
    else:
        manifest = SCRIPT_DIR / f"{HOST_NAME}.json"
        if manifest.exists():
            manifest.unlink()
            ok(f"Removed {manifest}")
            removed = True
        try:
            import winreg
            for reg_path in [
                rf"SOFTWARE\Google\Chrome\NativeMessagingHosts\{HOST_NAME}",
                rf"SOFTWARE\Microsoft\Edge\NativeMessagingHosts\{HOST_NAME}",
                rf"SOFTWARE\BraveSoftware\Brave-Browser\NativeMessagingHosts\{HOST_NAME}",
            ]:
                try:
                    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, reg_path)
                    ok(f"Removed registry: {reg_path}")
                    removed = True
                except FileNotFoundError:
                    pass
        except ImportError:
            pass

    for fname in ["phoneme_host.bat", "phoneme_host.sh"]:
        p = SCRIPT_DIR / fname
        if p.exists():
            p.unlink()
            ok(f"Removed {fname}")
            removed = True

    if removed:
        print(f"\n  {green('✓ Native host unregistered.')}")
        print(f"    Model files in models/ were kept. Delete manually if needed.")
        print(f"    To remove the extension, go to chrome://extensions\n")
    else:
        print(f"\n  {dim('Nothing to remove — native host was not installed.')}\n")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--uninstall" in args or "uninstall" in args:
        uninstall()
        sys.exit(0)

    if "--help" in args or "-h" in args:
        print(__doc__)
        print(f"  Model variants (pass as flag):")
        for key, v in MODELS.items():
            default = " (default)" if key == DEFAULT_MODEL else ""
            print(f"    --{key:6s}  {v['size']:>8s}  {v['file']}{default}")
        print()
        sys.exit(0)

    # Parse model variant
    variant_key = DEFAULT_MODEL
    for key in MODELS:
        if f"--{key}" in args:
            variant_key = key
            args.remove(f"--{key}")
            break

    # Parse extension ID
    ext_id_arg = None
    for arg in args:
        if not arg.startswith("-"):
            ext_id_arg = arg.lower()
            break

    # Banner
    print()
    print(f"  {bold('ILE Pronunciation Coach — Setup')}")
    print(f"  {dim('─' * 40)}")

    total = 6

    check_python(1, total)
    install_deps(2, total)
    model_path = download_model(3, total, variant_key)
    ext_id = resolve_extension_id(4, total, ext_id_arg)
    register_native_host(5, total, ext_id)
    smoke_test(6, total)

    print()
    print(f"  {dim('─' * 40)}")
    print(f"  {green('✓ Setup complete!')}")
    print()
    print(f"  {bold('Next steps:')}")
    print(f"    1. Go to {cyan('chrome://extensions')}")
    print(f"    2. Click the {bold('reload')} button on {EXTENSION_NAME}")
    print(f"    3. Open any lesson and start recording!")
    print()
    print(f"  {dim(f'Extension ID: {ext_id}')}")
    print(f"  {dim(f'Host name:    {HOST_NAME}')}")
    print(f"  {dim(f'Model:        {model_path}')}")
    print()


if __name__ == "__main__":
    main()
