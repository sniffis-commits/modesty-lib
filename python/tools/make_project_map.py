"""Write PROJECT-MAP.md (the tables aligned, the tree with spaces).

Kept as a script so the map can be regenerated when the project moves:
    python packages/nsfw-lib/tools/make_project_map.py
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "PROJECT-MAP.md")


def table(rows):
    """A markdown table with every column padded to its content."""
    widths = [0] * len(rows[0])
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(rows[0])) + " |"]
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(c.ljust(widths[i])
                                      for i, c in enumerate(row)) + " |")
    return "\n".join(lines)


TREE = """```
.
|-- README.md                  the front door: what it is, how to install it
|-- PROJECT-MAP.md             this file: every part, its state, what is open
|-- ROADMAP.md                 every phase, every open item
|-- CHANGELOG.md               what changed, per version
|-- PREPRODUCTION.md           the checklist before the first release
|-- NAMING.txt                 the brand candidates and what they promise
|-- LICENSES.html              the licence review (models, lists, deps)
|-- LICENSE                    MIT (the weights have their own terms)
|-- test.py                    the test suite (mocked + the real media)
|-- tryme.py                   a live demo over real sites
|-- run.bat / run.sh           the service launcher (worker + API + site)
|-- tools/
|   |-- tasks.py               the installer: register/start/stop/remove the
|   |                          services (schtasks + autostart key on Windows,
|   |                          cron on Linux)
|   `-- cleanup.py             the way back: stop everything, remove every
|                              registration, clear the leftovers
|-- SAMPLES/                   the frames the fast tests scan
|-- docs/
|   `-- SCREEN-GUARD-SPEC.md   what the Android app is and how to build it
|-- examples/
|   `-- skiplist/              the skip-list format + two proof-of-concept players
|       |-- README.md          the format, in plain words
|       |-- player.html        a one-file player (the browser is the player)
|       `-- vlc-skiplist.lua   a VLC playlist parser (VLC ships Lua)
|-- packages/
|   |-- nsfw-lib/              the scanner itself - the whole engine
|   |   |-- src/nsfw_scanner/
|   |   |   |-- scanner.py     the rating rule, the image path, the CLI
|   |   |   |-- media.py       the video/audio path, the progress, the problems
|   |   |   |-- decoder.py     where the frames come from (ffmpeg or PyAV)
|   |   |   |-- faces.py      the face detector (optional, deprecated)
|   |   |   |-- synthetic.py  "is this a render / a drawing?"
|   |   |   |-- plugins.py    bring your own classifier
|   |   |   |-- skiplist.py   read/write/make a skip list
|   |   |   |-- known.py      match before scanning (DB, hash list, API)
|   |   |   |-- badwords.py   the hostname check (no fetch, no model)
|   |   |   |-- lexicons.py   the configurable sensuality word lists
|   |   |   |-- subtitles.py  SRT / VTT / ASS parsing
|   |   |   |-- data/lexicons/ the word lists (editable)
|   |   |   `-- models/       the ONNX weights (not in git)
|   |   |-- tools/
|   |   |   |-- make_samples.py    cut the sample frames out of the videos
|   |   |   `-- distill/       the distillation (standalone, see its README)
|   |   |       |-- distill.bat / distill.sh   the whole pipeline
|   |   |       |-- pack.bat / pack.sh         pack it for delivery
|   |   |       |-- run.py                    the single entry point
|   |   |       `-- nsfw_distill/             the package (teacher, student, gate)
|   |   `-- README.md          the library's own documentation
|   |-- nsfw-cli/              a thin wrapper: `nsfw-scan` -> the library
|   |-- nsfw-webservice/       the HTTP API + the SQLite cache
|   |   |-- src/server.py     the routes (ours and the open /v1)
|   |   |-- src/cache_db.py   the cache, the media hashes, the problems
|   |   `-- tests/soak_test.py the load test
|   |-- nsfw-db/               the free blocklist database
|   |   |-- src/import_blocklists.py  the importers
|   |   |-- src/check_domains.py     the DNS/whois health checker
|   |   `-- data/nsfw.sqlite         the shipped database (4.7M domains)
|   |-- nsfw-website/          the PHP service site
|   |   |-- inc/              the shared functions (the API, the verdicts)
|   |   |-- modules/          the pages (scanner, admin, health, community)
|   |   |-- sys/config.php    the configuration (the brand, the thresholds)
|   |   |-- data/             the site's own SQLite
|   |   `-- www/ templates/   the assets
|   |-- nsfw-android/          the screen-guard app (one APK per market)
|   |   |-- core/             the engine in pure Kotlin: Resize, Preprocess,
|   |   |                     Rating, Variant, PollCache, Pin, NsfwModels
|   |   |                     (+ the JVM tests, the golden one included)
|   |   |-- app/              MainActivity, CaptureService, CoverActivity,
|   |   |                     Guard, Settings, ModelAssets; 4 flavours
|   |   |                     (kids, family, strict, adult)
|   |   |-- tools/make_golden.py  the vectors the Python side shares
|   |   `-- README.md         the build, the PIN, the emulator caveat
|   `-- nsfw-frontend-example/ the static demo page (the reference client)
|-- old-code/                  the archived earlier attempts (Delphi, threaded)
`-- .github/workflows/         the weekly database build
```"""

SCALE = table([
    ["rating", "meaning"],
    ["`1`", "no unsafe signal detected (not a safety guarantee)"],
    ["`2`", "synthetic content (a render, a drawing, a UI screenshot) - **not rated**"],
    ["`5`", "women/girls present, no stronger unsafe signal detected"],
    ["`6`", "romance - clothed, affectionate"],
    ["`7`", "revealing, not fully explicit"],
    ["`8`", "strongly suggestive (reserved; the band is off by default)"],
    ["`10`", "explicit adult content"],
])

ENGINE = table([
    ["part", "what it does", "where", "state", "what's next"],
    ["the rating rule", "10 needs the second model to agree; 7 needs both models; 6 is romance", "`nsfw_scanner/scanner.py`", "working", "-"],
    ["the image path", "one image -> a rating, a reason, the evidence", "`nsfw_scanner/scanner.py`", "working", "-"],
    ["the media path", "a video -> frames at 1 fps, the worst frame decides", "`nsfw_scanner/media.py`", "working", "-"],
    ["the frame sources", "ffmpeg (raw frames), ffmpeg (PNG), PyAV in-process", "`nsfw_scanner/decoder.py`", "working", "the buffered transport (~12 %)"],
    ["the subtitles", "SRT / VTT / ASS, embedded or sidecar", "`nsfw_scanner/subtitles.py`", "working", "an online subtitle provider (parked)"],
    ["the synthetic gate", "\"a render or a drawing - not rated\" (rating 2)", "`nsfw_scanner/synthetic.py`", "working", "-"],
    ["the face detector", "crops the face for the gender model", "`nsfw_scanner/faces.py`", "partial", "deprecated: off by default for video"],
    ["the plugins", "your own classifier: rate / censor / skip / report", "`nsfw_scanner/plugins.py`", "partial", "`--list-plugins`"],
    ["the skip list", "read, write and make one", "`nsfw_scanner/skiplist.py`", "working", "a real player plugin"],
    ["the lexicons", "the sensuality word lists (editable files)", "`nsfw_scanner/lexicons.py`", "working", "-"],
    ["the badwords", "the hostname check (no fetch, no model)", "`nsfw_scanner/badwords.py`", "working", "-"],
    ["match before scanning", "the local DB, a hash list, the hosted API", "`nsfw_scanner/known.py`", "working", "-"],
])

MODELS = table([
    ["part", "what it does", "where", "state", "what's next"],
    ["the joint model", "5 classes: drawings / hentai / neutral / porn / sexy", "`models/nsfw_joint_model.onnx`", "working", "-"],
    ["the binary model", "2 classes: the calibrated \"is this an act?\"", "`models/nsfw_model.onnx`", "working", "-"],
    ["the gender model", "women present (the rating 5)", "`models/gender_model.onnx`", "working", "-"],
    ["the face model", "finds the faces", "`models/face_model.onnx`", "partial", "deprecated"],
    ["the distilled student", "one small model instead of the two NSFW models", "`tools-internal-staging/distill/`", "partial", "the run (needs a corpus); the export works"],
    ["the teachers", "where the weights come from", "open_nsfw, Adience, UltraFace", "external", "attribution is in LICENSES.html"],
])

PRODUCTS = table([
    ["part", "what it does", "where", "state", "what's next"],
    ["the CLI", "`nsfw-scan <target>` -> a rating, a reason, the problems", "`packages/nsfw-cli`", "working", "-"],
    ["the benchmark", "CPU vs GPU, model vs model, with a table", "`nsfw-scan --benchmark`", "working", "-"],
    ["the webservice", "the JSON API + the SQLite cache", "`packages/nsfw-webservice`", "working", "-"],
    ["the open API `/v1`", "hash -> result, three fields", "`webservice/server.py`", "working", "the site's page for it"],
    ["the database", "4.7M adult domains, rebuilt weekly", "`packages/nsfw-db`", "working", "-"],
    ["the scan queue", "the background worker (URLs and media)", "`packages/nsfw-webservice/src/scan_queue.py`", "working", "-"],
    ["the website", "the PHP service: the checker, the scanner, the admin", "`packages/nsfw-website`", "working", "-"],
    ["the admin", "the cache, the DB, the blocked IPs, the API keys", "`nsfw-website/modules/admin`", "working", "-"],
    ["the demo page", "the static reference client", "`packages/nsfw-frontend-example`", "working", "-"],
    ["the screen guard", "the Android app: 4 APKs, one per market", "`packages/nsfw-android`", "partial", "the real-device check; the student"],
])

TOOLING = table([
    ["part", "what it does", "where", "state", "what's next"],
    ["the tasks", "register / start / stop / remove the four services", "`tools/tasks.py`", "working", "-"],
    ["the cleanup", "stop everything, remove every registration, clear the rest", "`tools/cleanup.py`", "working", "-"],
    ["the emulator", "boot the guard AVD for the app tests", "`run.bat emulator`", "working", "-"],
])

NOTBUILT = table([
    ["part", "what it is", "where", "state", "what's next"],
    ["the audio tiers", "speech-to-text, audio events, sexual vocalizations", "`packages/nsfw-lib/tools/evaluate_audio.py`", "evaluation started", "rights-cleared audio corpus, then the PANNs route"],
    ["the frame bus", "one decode, many readers", "-", "planned", "after the buffered transport"],
    ["a real player plugin", "mpv / Kodi / Jellyfin reading a skip list", "`examples/skiplist`", "planned", "the PoC proves the format"],
])

GLOSSARY = table([
    ["word", "what it means"],
    ["the **verdict**", "the unsafe-content rating we give (1-10)"],
    ["the **status**", "whether the domain is alive, parked or dead (a liveness fact)"],
    ["**synthetic**", "a render, a drawing, a UI screenshot - the model cannot judge it, so it is *not rated* (2)"],
    ["the **skin hypothesis**", "the joint model saying \"there is a lot of bare skin here\" (not: \"this is explicit\")"],
    ["the **second opinion**", "the binary model, which decides whether the skin hypothesis means an actual act"],
    ["the **skip list**", "a text file of sha256 hashes to skip, with a reason for each"],
    ["a **plugin**", "a classifier you add, with a rule for what it may do to the rating"],
    ["the **student**", "the small distilled model that replaces the two NSFW teachers"],
    ["the **corpus**", "the folder of images the student is trained on (labelled by the teachers)"],
])

FLOW = """```
  a URL / a file / a video
            |
            v
   +------------------+     hostname?      +--------------------+
   |  nsfw-scanner    | ---> badwords ---> | rating 10, no fetch |
   |  (the library)   |                    +--------------------+
   +------------------+
            |
            |  images  ---> the joint model  ---> the classes
            |            -> the binary model  ---> "is this an act?"
            |            -> the gender model ---> women present
            |            -> the face model   ---> the face crop
            |            -> the plugins     ---> your own classifiers
            |
            |  video  ---> ffmpeg (or PyAV) ---> 1 frame per second
            |                                 -> the same models
            |                                 -> the subtitles
            |            the worst frame decides the file
            v
   rating 1..10 + the reason + the problem areas + the sha256
            |
            +---> the known targets (the DB, a hash list, the API)
            |
            v
   +------------------+        +--------------------+
   |  the webservice  | -----> | the site's SQLite  |
   |  (the cache)     |        | (the media hashes, |
   +------------------+        |  the problems)     |
            |                   +--------------------+
            |  /v1/hash, /v1/sample (the open API)
            v
   +------------------+
   |  the PHP site   |  the checker, the scanner queue, the admin
   +------------------+
```"""

OPEN = """**Needs other people**
- the **distillation run** - the corpus is on the friend's side; the moment
  the student arrives there is a six-step receiving checklist (put it in
  `models/`, validate it against the teachers, run the sample test, then
  decide whether it becomes the default - the wheel would drop from
  ~124 MB to ~32 MB)
- the **audio tiers** - their *own* corpus (audio, not pictures: the audio
  detector is a different model on a different pipeline; it only shares the
  *technique* with the picture distillation)
- the **PyPI account + token** - the wheel and the packaging CI are ready;
  the credential is what is missing

**Ready now, small**
- `--list-plugins` (the plugin system's last piece)
- the site's **open-API page** (the API works; the site does not document it)
- the **weekly DB release** - the workflow works; teach it to run on push
  and let the first release happen by hand

**Ready now, medium**
- the **buffered transport** (a shared-memory ring: a measured ~12 % on
  every media scan)
- the **frame bus's first reader** (the site's "show me the problem
  frames" preview)
- a **real player plugin** (mpv / Kodi / Jellyfin)
- the **screen guard's real-device check** (the logic is proven on the
  emulator; the cover's *visibility* is not - while the projection runs,
  the emulator stops painting the app's own views)

**Decisions only the owner can make**
- the **brand** (NAMING.txt - nothing is renamed yet)
- the **corpus** for the friend
- the screen guard's **uninstall protection** (the device-owner /
  app-pinning route needs a provisioning decision)"""

DOC = """# The project map

What this project is, what every part of it does, what state it is in,
and what is still open - in one file, in plain words.

**Modesty - nsfw protection toolkit** rates websites, images and video/audio on a 1-10 scale
for adult content. It is MIT-licensed, runs entirely on your machine,
and is built as a small library with thin wrappers around it: a CLI, an
HTTP API, a blocklist database, and a PHP website that sells speed and
hosting around the free parts.

The rating scale (one scale, everywhere - the library, the API, the
site, the plugins):

%s

---

## 1. The directory tree

%s

---

## 2. The components and their state

The state words: **working** (in use, tested) - **partial** (works, with
a known limit) - **planned** (designed, not written) - **parked**
(deliberately not now) - **external** (somebody else's - the teachers,
the corpus).

### The engine

%s

### The models

%s

### The products around the engine

%s

### The tooling around the products

%s

### The things that are not built yet

%s

---

## 3. How the data flows

%s

---

## 4. The words we use

%s

---

## 5. What is open, per part

%s
""" % (SCALE, TREE, ENGINE, MODELS, PRODUCTS, TOOLING, NOTBUILT, FLOW, GLOSSARY, OPEN)


def main():
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(DOC)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
