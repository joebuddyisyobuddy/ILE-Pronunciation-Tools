#!/usr/bin/env python3
"""
ILE Pronunciation Coach — Setup

Usage:
    python setup.py                     # auto-detect extension ID
    python setup.py EXTENSION_ID        # provide ID manually
    python setup.py --uninstall         # remove native host registration

What happens:
    1. Check Python version (3.10+)
    2. Install ALL dependencies (numpy, scipy, librosa, onnxruntime, etc.)
    3. Download the phoneme model from HuggingFace (~318 MB)
    4. Find your browser extension automatically
    5. Register the native messaging host
    6. Smoke test

Takes about 2-5 minutes depending on connection speed.

Don't have Python? Install it from the python-installer folder first.
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
MIN_PYTHON = (3, 10)

# Install in order — foundational packages first, then packages that depend on them.
# pip resolves transitive deps, but installing in order prevents version conflicts.
INSTALL_GROUPS = [
    # Group 1: Core numerics (no complex deps)
    ["numpy"],
    # Group 2: Scientific stack (depends on numpy)
    ["scipy", "scikit-learn"],
    # Group 3: ONNX Runtime + its deps
    ["protobuf", "flatbuffers", "sympy", "onnxruntime"],
    # Group 4: Audio processing (depends on numpy, scipy)
    ["soundfile", "audioread", "soxr", "librosa"],
]

# HuggingFace direct download — pre-built ONNX, no local export needed
MODEL_URL = "https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/resolve/main/onnx/model_int8.onnx"
MODEL_FILENAME = "model_int8.onnx"
MODEL_MIN_SIZE_MB = 50

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
HOST_SCRIPT = SCRIPT_DIR / "phoneme_host.py"
VOCAB_PATH = MODELS_DIR / "vocab.json"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

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
    print(f"         {green('OK')} {msg}")

def warn(msg):
    print(f"         {yellow('!!')} {msg}")

def fail(msg):
    print(f"         {red('FAIL')} {msg}")

def info(msg):
    print(f"         {dim('->')} {msg}")

def fatal(msg):
    fail(msg)
    print(f"\n  {red('Setup failed.')} Fix the issue above and run setup.py again.\n")
    sys.exit(1)

# ─── pip helper ───────────────────────────────────────────────────────────────

def pip_install(packages, attempt=1):
    """Install packages via pip with fallbacks for common issues."""
    pip_cmd = [sys.executable, "-m", "pip", "install",
               "--quiet", "--disable-pip-version-check"] + packages

    try:
        subprocess.run(pip_cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""

        # Externally managed (Debian/Ubuntu system Python)
        if "externally-managed" in stderr and attempt == 1:
            warn("System-managed Python detected, retrying with --break-system-packages")
            pip_cmd.insert(4, "--break-system-packages")
            try:
                subprocess.run(pip_cmd, check=True, capture_output=True, text=True)
                return True
            except subprocess.CalledProcessError as e2:
                fail(f"pip install failed for: {' '.join(packages)}")
                print(f"         {e2.stderr[:300]}" if e2.stderr else "")
                return False

        # No pip module
        elif "No module named pip" in stderr:
            fail("pip is not installed")
            if SYSTEM == "Windows":
                info("Try: python -m ensurepip --upgrade")
            else:
                info("Try: python3 -m ensurepip --upgrade")
                info("  or: sudo apt install python3-pip  (Ubuntu/Debian)")
                info("  or: brew install python  (macOS)")
            return False

        # General failure
        else:
            fail(f"pip install failed for: {' '.join(packages)}")
            if stderr:
                for line in stderr.strip().split("\n")[:5]:
                    print(f"         {line}")
            return False

# ─── Step 1: Python version ──────────────────────────────────────────────────

def check_python(n, total):
    step(n, total, "Checking Python version")
    v = sys.version_info
    info(f"Found Python {v.major}.{v.minor}.{v.micro} at {sys.executable}")

    if (v.major, v.minor) < MIN_PYTHON:
        fatal(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.\n"
              "         If you just installed Python, make sure you're running\n"
              "         the new version, not an older one on your PATH.\n"
              "         Windows: try 'py setup.py' instead of 'python setup.py'")

    # Check pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                       check=True, capture_output=True, text=True)
        ok(f"Python {v.major}.{v.minor} with pip")
    except (subprocess.CalledProcessError, FileNotFoundError):
        warn("pip not found, attempting to install it...")
        try:
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"],
                           check=True, capture_output=True, text=True)
            ok("pip installed via ensurepip")
        except subprocess.CalledProcessError:
            fatal("Could not install pip.\n"
                  "         Windows: re-run the Python installer and check 'pip'\n"
                  "         Mac: brew install python\n"
                  "         Linux: sudo apt install python3-pip")

# ─── Step 2: Install ALL dependencies ─────────────────────────────────────────

def install_deps(n, total):
    step(n, total, "Installing dependencies")
    info("This installs numpy, scipy, librosa, onnxruntime, and all sub-dependencies")
    info("First run may take 2-3 minutes...")
    print()

    # Upgrade pip first to avoid old resolver issues
    info("Ensuring pip is up to date...")
    pip_install(["--upgrade", "pip"])

    all_ok = True
    for i, group in enumerate(INSTALL_GROUPS):
        info(f"Installing group {i+1}/{len(INSTALL_GROUPS)}: {', '.join(group)}")
        if pip_install(group):
            for pkg in group:
                ok(pkg)
        else:
            all_ok = False
            # Try packages individually to isolate the failure
            warn(f"Group install failed, trying packages individually...")
            for pkg in group:
                if pip_install([pkg]):
                    ok(pkg)
                else:
                    fail(f"{pkg} — could not install")
                    all_ok = False

    if not all_ok:
        print()
        warn("Some packages failed to install.")
        info("You can try installing them manually:")
        all_pkgs = [pkg for group in INSTALL_GROUPS for pkg in group]
        hint = "py -m pip" if SYSTEM == "Windows" else "python3 -m pip"
        info(f"  {hint} install {' '.join(all_pkgs)}")
        print()

# ─── Step 3: Download model ──────────────────────────────────────────────────

def download_model(n, total):
    step(n, total, "Downloading phoneme model")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if model already exists
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        if size_mb > MODEL_MIN_SIZE_MB:
            ok(f"Model already exists ({size_mb:.0f} MB)")
            _check_vocab()
            return

    info(f"Downloading {MODEL_FILENAME} (~318 MB) from HuggingFace...")
    info("Source: onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX")

    tmp_path = MODEL_PATH.with_suffix(".tmp")

    try:
        import urllib.request

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"\r         -> [{bar}] {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, tmp_path, reporthook=_progress)
        print()

        # Verify
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        if size_mb < MODEL_MIN_SIZE_MB:
            tmp_path.unlink()
            fatal(f"Download too small ({size_mb:.1f} MB). Try again.")

        # Atomic rename
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        tmp_path.rename(MODEL_PATH)
        ok(f"Model downloaded ({size_mb:.0f} MB)")

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        fatal(f"Download failed: {e}\n"
              f"         You can download manually from:\n"
              f"         {MODEL_URL}\n"
              f"         Save it as: {MODEL_PATH}")

    _check_vocab()


def _check_vocab():
    """Verify vocab.json exists."""
    if VOCAB_PATH.exists():
        try:
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            ok(f"Vocab loaded — {len(vocab)} phoneme tokens")
        except (json.JSONDecodeError, IOError) as e:
            warn(f"vocab.json may be corrupted: {e}")
    else:
        warn("vocab.json not found in models/ folder")
        info("This file should be included in the repository.")
        info("Phoneme labels may not work without it.")

# ─── Step 4: Find extension ID ───────────────────────────────────────────────

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
        info(f"Using first: {found[0][0]} ({found[0][1]})")
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

    # Verify phoneme_host.py exists
    if not HOST_SCRIPT.exists():
        fatal(f"phoneme_host.py not found at {HOST_SCRIPT}\n"
              "         Make sure you have the complete repository.")

    # Create launcher script
    python_exe = Path(sys.executable).resolve()

    if SYSTEM == "Windows":
        launcher = SCRIPT_DIR / "phoneme_host.bat"
        launcher.write_text(
            f'@echo off\r\n"{python_exe}" "{HOST_SCRIPT}"\r\n',
            encoding="utf-8"
        )
        host_path = str(launcher)
        ok(f"Created launcher: {launcher.name}")
    else:
        launcher = SCRIPT_DIR / "phoneme_host.sh"
        python_path = shutil.which("python3") or shutil.which("python") or str(python_exe)
        launcher.write_text(
            f'#!/bin/bash\nexec "{python_path}" "{HOST_SCRIPT}"\n',
            encoding="utf-8"
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        host_path = str(launcher)
        ok(f"Created launcher: {launcher.name}")

    # Write manifest JSON
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
    ok(f"Wrote manifest: {manifest_path}")

    # Register for other installed browsers too
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

    # Windows: registry keys
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

    # Verify the chain
    info("Verifying installation paths...")
    errors = []

    if not Path(host_path).exists():
        errors.append(f"Launcher not found: {host_path}")
    if not HOST_SCRIPT.exists():
        errors.append(f"Host script not found: {HOST_SCRIPT}")
    if not manifest_path.exists():
        errors.append(f"Manifest not found: {manifest_path}")
    if not python_exe.exists():
        errors.append(f"Python not found: {python_exe}")

    if errors:
        for e in errors:
            fail(e)
        fatal("Path verification failed")
    else:
        ok("All paths verified")

    ok("Native host registered")

# ─── Step 6: Smoke test ──────────────────────────────────────────────────────

def smoke_test(n, total):
    step(n, total, "Running smoke test")

    failures = []

    for pkg_name, import_name in [("numpy", "numpy"), ("scipy", "scipy"),
                                   ("librosa", "librosa"), ("onnxruntime", "onnxruntime"),
                                   ("soundfile", "soundfile")]:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            ok(f"{pkg_name} {version}")
        except ImportError as e:
            fail(f"{pkg_name} — {e}")
            failures.append(pkg_name)

    if failures:
        warn(f"Missing packages: {', '.join(failures)}")
        hint = "py -m pip" if SYSTEM == "Windows" else "python3 -m pip"
        info(f"Try: {hint} install {' '.join(failures)}")
        return False

    # Test model loading if model exists
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > MODEL_MIN_SIZE_MB * 1024 * 1024:
        try:
            info("Loading ONNX model for inference test...")
            import onnxruntime as ort
            import numpy as np

            sess = ort.InferenceSession(str(MODEL_PATH),
                                        providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            dummy = np.zeros((1, 16000), dtype=np.float32)
            result = sess.run(None, {input_name: dummy})
            ok(f"Inference test passed — output shape: {result[0].shape}")
        except Exception as e:
            warn(f"Inference test failed: {e}")
            info("The model may still work — try recording in the extension")
    else:
        warn("Skipping inference test (model not fully downloaded)")

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
        print(f"\n  {green('OK')} Native host unregistered.")
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
        sys.exit(0)

    ext_id_arg = None
    for arg in args:
        if not arg.startswith("-"):
            ext_id_arg = arg.lower()
            break

    # Banner
    print()
    print(f"  {bold('ILE Pronunciation Coach — Setup')}")
    print(f"  {dim('=' * 40)}")

    total = 6

    check_python(1, total)
    install_deps(2, total)
    download_model(3, total)
    ext_id = resolve_extension_id(4, total, ext_id_arg)
    register_native_host(5, total, ext_id)
    smoke_test(6, total)

    print()
    print(f"  {dim('=' * 40)}")
    print(f"  {green('OK')} {bold('Setup complete!')}")
    print()
    print(f"  {bold('Next steps:')}")
    print(f"    1. Go to {cyan('chrome://extensions')}")
    print(f"    2. Click the {bold('reload')} button on {EXTENSION_NAME}")
    print(f"    3. Open any lesson and start recording!")
    print()
    print(f"  {dim(f'Extension ID: {ext_id}')}")
    print(f"  {dim(f'Host name:    {HOST_NAME}')}")
    print(f"  {dim(f'Model:        {MODEL_PATH}')}")
    print()


if __name__ == "__main__":
    main()