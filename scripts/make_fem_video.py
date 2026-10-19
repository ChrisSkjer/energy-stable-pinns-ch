"""Turn a FEM run's frames/frame_*.png into an animation.

Writes an animated GIF with Pillow (no extra install), or an MP4 if the
output ends in .mp4 and ffmpeg is on PATH (WSL: conda install -n fenicsx-env
-c conda-forge ffmpeg; Windows: winget install ffmpeg).

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/make_fem_video.py results/eps0.05_nx100_dte-4
    .venv\\Scripts\\python.exe scripts/make_fem_video.py results/eps0.05_nx100_dte-4 --fps 20 --every 2
    .venv\\Scripts\\python.exe scripts/make_fem_video.py results/eps0.05_nx100_dte-4 -o cross.mp4

The frames are numbered by time step (frame_000000, frame_000005, ...), so
ffmpeg's `-i frame_%06d.png` stops at the first gap; here the files are
sorted and piped in instead.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="FEM output dir containing frames/")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output file, .gif or .mp4 (default: <run_dir>/<run name>.gif)")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second")
    parser.add_argument("--every", type=int, default=1, help="Use every N-th frame")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Resize factor, e.g. 0.5 for a smaller GIF")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="Seconds to hold the last frame before looping")
    return parser.parse_args()


def load_frames(frames_dir: Path, every: int, scale: float) -> list[Image.Image]:
    paths = sorted(frames_dir.glob("frame_*.png"))[::every]
    if not paths:
        raise SystemExit(f"no frame_*.png in {frames_dir}")
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if scale != 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        frames.append(img)
    return frames


def write_gif(frames: list[Image.Image], out: Path, fps: float, hold: float) -> None:
    ms = round(1000 / fps)
    durations = [ms] * len(frames)
    durations[-1] = max(ms, round(1000 * hold))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def write_mp4(frames: list[Image.Image], out: Path, fps: float, hold: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH; use a .gif output, or install ffmpeg")
    frames = frames + [frames[-1]] * round(hold * fps)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
        # yuv420p needs even width/height, and is what most players expect
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img in frames:
        img.save(proc.stdin, format="PNG")
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


def main():
    args = parse_args()
    out = args.output or args.run_dir / f"{args.run_dir.resolve().name}.gif"
    frames = load_frames(args.run_dir / "frames", args.every, args.scale)
    if out.suffix.lower() == ".mp4":
        write_mp4(frames, out, args.fps, args.hold)
    elif out.suffix.lower() == ".gif":
        write_gif(frames, out, args.fps, args.hold)
    else:
        raise SystemExit(f"unsupported output type {out.suffix!r}; use .gif or .mp4")
    print(f"saved {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
