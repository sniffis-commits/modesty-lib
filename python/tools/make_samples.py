"""Extract the sample frames the fast tests use.

The real-media test needs the four test videos (big, gitignored).  The
*sample* frames are the same frames, cut out and kept in the repo - they
pin the model's behaviour on the cases that matter, and they let the
test suite run in seconds instead of minutes.

The file name carries the expected rating::

    SAMPLES/testJewish_frame54_6.png   ->  the expected rating is 6

Run it from the repo root::

    python packages/nsfw-lib/tools/make_samples.py

It needs the test videos (they are gitignored) and ffmpeg.  The frames
are picked because they are the *interesting* ones: the good cases and the
known bad ones (the frames that used to be false positives).
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SAMPLES = os.path.join(ROOT, "SAMPLES")

# (file, timecode, expected rating, what the frame shows)
SAMPLES_LIST = (
    ("testJewish.mp4", 40.48, 1,
     "a man singing with his head tilted back - the exposed neck and jaw "
     "read as skin (the known false positive)"),
    ("testJewish.mp4", 54.48, 6,
     "a wedding: a bride in a white dress and a groom celebrating - a lot "
     "of light skin, but not an act"),
    ("testJewish.mp4", 71.48, 6,
     "a couple hugging - romance (the model calls it 'sexy')"),
    ("testJewish.mp4", 108.48, 1,
     "a rabbi with a long white beard - light tones, not skin, not an act"),
    ("test2Hindi.mp4", 124.48, 5,
     "a smiling man - the frame the old tiles called 'porn 0.93'"),
    ("test2Hindi.mp4", 179.48, 10,
     "a close-up with a hand on a face - both models agree it is explicit"),
    ("test2Hindi.mp4", 260.48, 1,
     "a man on a beach with a shoe - hands and skin, not revealing"),
    ("testShakiraTheEvil.mp4", 44.46, 7,
     "a bikini - revealing, but not an act"),
    ("testShakiraTheEvil.mp4", 80.47, 6,
     "a dancer in a revealing outfit - romance"),
    ("testHindi.mp4", 127.46, 5,
     "a pencil drawing - the picture decision is 5 (the gender model sees a "
     "drawn face); the *file* is 2 (synthetic content, not rated)"),
)


def extract(video, timecode, out_path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
           "-ss", "%.2f" % timecode, "-frames:v", "1", "-vf", "scale='min(640,iw)':-2",
           out_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return "ffmpeg failed: %s" % exc
    if proc.returncode != 0 or not os.path.exists(out_path):
        return "ffmpeg failed: %s" % proc.stderr.decode("utf-8", "replace")[:200]
    return None


def main():
    os.makedirs(SAMPLES, exist_ok=True)
    missing, written = [], []
    for name, timecode, expected, what in SAMPLES_LIST:
        video = os.path.join(ROOT, name)
        if not os.path.exists(video):
            missing.append(name)
            continue
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(SAMPLES, "%s_frame%s_%d.png" % (
            stem, ("%g" % timecode).replace(".", "_"), expected))
        error = extract(video, timecode, out_path)
        if error:
            print("  %-40s FAILED (%s)" % (os.path.basename(out_path), error))
            continue
        written.append((out_path, what))
        print("  %-40s  %s" % (os.path.basename(out_path), what))

    # the manifest: what each frame is, in plain words
    manifest = os.path.join(SAMPLES, "README.md")
    with open(manifest, "w", encoding="utf-8") as handle:
        handle.write("# Samples\n\n"
                     "The frames the fast tests scan.  The name carries the\n"
                     "**expected rating**: `<file>_frame<timecode>_<rating>.png`.\n"
                     "They are cut out of the real test videos (which are\n"
                     "gitignored) so the test suite does not need them.\n\n"
                     "| file | what it shows |\n|---|---|\n")
        for out_path, what in written:
            handle.write("| `%s` | %s |\n" % (
                os.path.basename(out_path), what))
    print("\n%d sample(s) written to %s" % (len(written), SAMPLES))
    if missing:
        print("skipped (the video is not here): %s" % ", ".join(sorted(set(missing))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
