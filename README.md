# Offline screen translator (Chinese → English)

Point it at a part of the screen (e.g. the VM window) and it paints the English translation
directly over the Chinese text, live, like a phone camera translator.

**Nothing leaves this PC.** OCR (RapidOCR / PP-OCR) and translation (OPUS-MT via CTranslate2)
both run locally from model files on disk; the script makes no network connections.

## Setup

You only need to do this once. The internet is needed for these steps, but not for running the translator.

**1. Python environment** (Windows Python 3.13):

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**2. Translation model.** The model is not stored in git (it is ~80 MB). Download the original
OPUS-MT model and convert it to CTranslate2 format with the converter that comes with the venv.
From WSL, in the project folder:

```bash
mkdir -p models
curl -L -o models/opus-zho-eng-src.zip https://object.pouta.csc.fi/Tatoeba-MT-models/zho-eng/opus+bt-2021-04-30.zip
unzip models/opus-zho-eng-src.zip -d models/opus-zho-eng-src
.venv/Scripts/ct2-opus-mt-converter.exe --model_dir models/opus-zho-eng-src --output_dir models/opus-mt-zho-eng --quantization int8
cp models/opus-zho-eng-src/{source.spm,target.spm,README.md,LICENSE} models/opus-mt-zho-eng/
rm -r models/opus-zho-eng-src models/opus-zho-eng-src.zip   # remove the ~600 MB of download files
```

- Model download: [opus+bt-2021-04-30.zip](https://object.pouta.csc.fi/Tatoeba-MT-models/zho-eng/opus+bt-2021-04-30.zip) (~290 MB)
- Model page: [Helsinki-NLP Tatoeba-Challenge, zho-eng](https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/zho-eng)

The result is `models/opus-mt-zho-eng/`, with `model.bin`, `config.json`,
`shared_vocabulary.json`, `source.spm` and `target.spm`. The converter refuses to overwrite an
existing output folder; add `--force` to rebuild it.

## Run

Double-click **`run.bat`**, then drag a rectangle around the VM window. Or from WSL:

```bash
.venv/Scripts/python.exe translator.py --select       # pick the area with the mouse (recommended)
.venv/Scripts/python.exe translator.py                # whole primary monitor
.venv/Scripts/python.exe translator.py --monitor 2    # second monitor
.venv/Scripts/python.exe translator.py --no-overlay   # only print translations in the console
```

### Toolbar

A small floating toolbar appears just above the translated area (drag it anywhere by its
`⠿` grip). Clicking it does not take keyboard focus away from the VM.

| Button | Action |
|---|---|
| `●` | status: blue = loading models, green = translating, amber = paused |
| `❚❚ Pause` / `▶ Resume` | stop / restart translating (shows the original Chinese while paused) |
| `⤢ Resize` | pick a new area: the screen dims, drag a new rectangle (`Esc` keeps the old one) |
| `A−` `100%` `A+` | translation font size, 50%–300%. Above 100% the English may extend past the original text or wrap |
| `↕−` `1.0×` `↕+` | line spacing: distance between translated rows (and wrapped lines), 1.0×–3.0× the text height |
| `⎙ Screenshot` | picture of the area as you see it, translations included: copied to the clipboard (paste with `Ctrl+V`) and saved as a PNG in `screenshots/` |
| `✕` | quit |

Translations never cover each other: when a bigger font, a wrapped phrase or more line spacing
needs more height than the original line, the rows below are pushed down (a whole row at a time,
so table columns stay aligned). Rows pushed past the bottom of the area keep going below it.

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+Shift+T` | pause / resume |
| `Ctrl+Alt+Shift+Q` | quit (`Ctrl+C` in the console also works) |

If the VM window has keyboard focus it may capture the hotkeys; use the toolbar instead.

Other options: `--font-scale` (starting font size, e.g. `1.5`), `--line-spacing` (starting line
spacing, e.g. `1.5`), `--screenshot-dir` (where screenshots are saved), `--no-toolbar`, `--interval`
(seconds between screen checks, default 0.25), `--min-score` (OCR confidence, default 0.6),
`--opacity` (0.2–1). See `--help`.

Windows' own screenshot tools (`PrtScn`, `Win+Shift+S`) and screen sharing only see the original
Chinese: the overlay and toolbar are hidden from capture (that is how the
translator avoids reading its own English).

## How it works

1. Grabs the selected area of the screen; if nothing changed since last time, does nothing.
2. Detects text boxes; only boxes with new pixels go through text recognition (cached).
   Pieces of one line that the detector split apart are joined back together.
3. Translates new Chinese lines (cached), using `glossary.txt` first for exact matches.
   Only the Chinese is translated: anything else in a line (product names like `gitlab`, code,
   paths, URLs, e-mails, version numbers) is kept exactly as written.
4. Draws each translation over its original line, in a window that is click-through and
   hidden from screen capture (so it never reads its own English).

Speed: ~0.5–1 s per update on a full 1920×1200 screen, faster on a smaller area.

## Fixing bad translations

The offline model is good with sentences but can miss single words out of context (it
translated 否 "No" as "Yes"). Add corrections to **`glossary.txt`** (`中文 = English`, one per
line) and restart. Entries replace the translation when a whole on-screen line matches exactly.

## Files

- `translator.py`: the program
- `glossary.txt`: your translation overrides
- `models/opus-mt-zho-eng/`: translation model (Helsinki-NLP Tatoeba-MT `zho-eng/opus+bt-2021-04-30`,
  CC-BY 4.0, converted with `ct2-opus-mt-converter --quantization int8`). Not in git; see
  [Setup](#setup).
- `requirements.txt`: pinned packages; `.venv/`: Windows Python 3.13 virtual environment (not in git)

## Troubleshooting

- **VM window looks black / nothing is detected**: the VM viewer may block screen capture.
  Run with `--no-overlay` and check whether lines are printed. If not, the translator must
  run inside the VM instead. Download wheels here with
  `pip download -r requirements.txt --only-binary=:all: --platform win_amd64 --python-version 3.13 -d wheels`
  (match the VM's Python version), copy the project folder (without `.venv`, but with
  `models/`) plus `wheels/`
  into the VM, then run `python -m venv .venv` and
  `.venv\Scripts\pip install --no-index --find-links wheels -r requirements.txt` there.
- **Overlay hidden behind the VM**: put the VM viewer in windowed or borderless mode, not
  exclusive full-screen.
