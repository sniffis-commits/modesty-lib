@echo off
REM ---------------------------------------------------------------------------
REM Builds and tests nsfw-lib-rust with the MSVC toolchain only.
REM
REM The default Rust on this machine is the GNU build, which needs MinGW's
REM dlltool.exe. This script forces the rustup MSVC toolchain plus the Visual
REM Studio environment so no MinGW tool is ever needed.
REM
REM The ONNX Runtime binaries are not fetched by cargo (the pyke CDN is not
REM reachable from here), so they are taken from the local ONNX Runtime
REM release instead - see ONNXRUNTIME_DIR below.
REM
REM The ort crate links against onnxruntime.dll with `raw-dylib`, so the
REM Windows loader resolves it at process start using the standard search
REM order (application directory, then System32, then PATH). A stale
REM onnxruntime.dll in C:\WINDOWS\system32 is therefore picked up before
REM PATH, which fails with "requested API version [31] is not available".
REM To avoid that, the correct DLL is copied next to every test binary.
REM
REM Usage:  build_msvc.bat [cargo test arguments]
REM   e.g.  build_msvc.bat --features onnx gender_classification_tests
REM ---------------------------------------------------------------------------
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
set "CARGO_BUILD_TARGET=x86_64-pc-windows-msvc"

set "ONNXRUNTIME_DIR=%LOCALAPPDATA%\onnxruntime\onnxruntime-win-x64-1.31.0"
set "ORT_LIB_LOCATION=%ONNXRUNTIME_DIR%\lib"
set "ORT_PREFER_DYNAMIC_LINK=1"
REM Keep the local ONNX Runtime ahead of anything else on PATH as well.
set "PATH=%ONNXRUNTIME_DIR%\lib;%PATH%"

cd /d "%~dp0"

REM Split the arguments at "--" so the build-only phase does not receive the
REM test-runner arguments.
setlocal EnableDelayedExpansion
set "PRE_ARGS="
set "POST_ARGS="
set "SEEN_SEP="
for %%a in (%*) do (
    if not defined SEEN_SEP (
        if "%%a"=="--" (
            set "SEEN_SEP=1"
        ) else (
            set "PRE_ARGS=!PRE_ARGS! %%a"
        )
    ) else (
        set "POST_ARGS=!POST_ARGS! %%a"
    )
)
endlocal & set "PRE_ARGS=%PRE_ARGS%" & set "POST_ARGS=%POST_ARGS%"

REM Build the test binaries first so the DLLs can be placed next to them.
cargo test %PRE_ARGS% --no-run
if errorlevel 1 exit /b 1

REM Copy the ONNX Runtime DLLs next to every built executable.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src = '%ORT_LIB_LOCATION%'; Get-ChildItem -Path '%~dp0target\%CARGO_BUILD_TARGET%\debug' -Recurse -Filter *.exe -ErrorAction SilentlyContinue | ForEach-Object { $dest = $_.DirectoryName; Get-ChildItem -Path $src -Filter *.dll | ForEach-Object { if ($_.DirectoryName -ne $dest) { Copy-Item -Path $_.FullName -Destination $dest -Force -ErrorAction SilentlyContinue } } }"

if "%POST_ARGS%"=="" (
    cargo test %PRE_ARGS% -- --nocapture
) else (
    cargo test %PRE_ARGS% -- %POST_ARGS%
)

REM Also build release artifacts for dist/daemon (deploy.py --daemon expects them).
echo Building release artifacts for dist/daemon...
cargo build --release --features onnx,download --target %CARGO_BUILD_TARGET%
if errorlevel 1 (
    echo Release build failed.
    exit /b 1
)

REM Build the daemon crate (depends on nsfw-lib-rust with onnx+download features via its Cargo.toml)
cd /d "%~dp0..\nsfw-scanner-daemon"
cargo build --release --target %CARGO_BUILD_TARGET%
if errorlevel 1 (
    echo Daemon release build failed.
    exit /b 1
)
cd /d "%~dp0"

REM Build the scan-rust CLI (depends on nsfw-lib-rust with onnx feature)
cd /d "%~dp0..\nsfw-scan-rust"
cargo build --release --features onnx --target %CARGO_BUILD_TARGET%
if errorlevel 1 (
    echo Scan-rust release build failed.
    exit /b 1
)
cd /d "%~dp0"

REM Copy ONNX Runtime DLLs next to release binaries too.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src = '%ORT_LIB_LOCATION%'; Get-ChildItem -Path '%~dp0target\%CARGO_BUILD_TARGET%\release' -Recurse -Filter *.exe -ErrorAction SilentlyContinue | ForEach-Object { $dest = $_.DirectoryName; Get-ChildItem -Path $src -Filter *.dll | ForEach-Object { if ($_.DirectoryName -ne $dest) { Copy-Item -Path $_.FullName -Destination $dest -Force -ErrorAction SilentlyContinue } } }"

REM Also copy DLLs next to daemon binary
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src = '%ORT_LIB_LOCATION%'; Get-ChildItem -Path '%~dp0..\nsfw-scanner-daemon\target\%CARGO_BUILD_TARGET%\release' -Recurse -Filter *.exe -ErrorAction SilentlyContinue | ForEach-Object { $dest = $_.DirectoryName; Get-ChildItem -Path $src -Filter *.dll | ForEach-Object { if ($_.DirectoryName -ne $dest) { Copy-Item -Path $_.FullName -Destination $dest -Force -ErrorAction SilentlyContinue } } }"
