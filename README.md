# epub2tts-chatterbox

epub2tts-chatterbox is a free and open source Python app to easily create a full-featured audiobook from an epub or text file using realistic voice-cloning text-to-speech by [Chatterbox](https://github.com/resemble-ai/chatterbox). CUDA compatible GPU is required, or Apple silicone.

## 🚀 Features

- [x] Creates standard format M4B audiobook file
- [x] Automatic chapter break detection
- [x] Embeds cover art if specified
- [x] Resumes where it left off if interrupted
- [x] NOTE: epub file must be DRM-free

## 📖 Usage

<details>
<summary>Usage instructions</summary>

*NOTE:* If you want to specify where NLTK tokenizer will be stored (about 50mb), use an environment variable: `export NLTK_DATA="your/path/to/nltk_data"`

## OPTIONAL - activate the virtual environment if using
1. `source .venv/bin/activate`

## FIRST - extract epub contents to text and cover image to png:
1. `epub2tts-chatterbox mybook.epub`
2. **edit mybook.txt**, replacing `# Part 1` etc with desired chapter names, and removing front matter like table of contents and anything else you do not want read. **Note:** First two lines can be Title: and Author: to use that in audiobook metadata.

## Read text to audiobook:

* `epub2tts-chatterbox mybook.txt --cover mybook.png --sample <speaker sample>`
* Specify a speaking sample with `--sample <speaker>`. Ideally your speaking sample should be 30-60 seconds long and can be WAV or MP3 (or a few other formats I don't recall). Sample should be clean audio, no background music or sounds.


## All options
* `-h, --help` - show this help message and exit
* `--sample SampleAudioFile` - Speaker sample to use (example: george.wav)
* `--cover image.[jpg|png]` - Image to use for cover
* `--notitles` - Do not read chapter titles when creating audiobook

## Deactivate virtual environment
`deactivate`
</details>

## 🐞 Reporting bugs
<details>
<summary>How to report bugs/issues</summary>

Thank you in advance for reporting any bugs/issues you encounter! If you are having issues, first please [search existing issues](https://github.com/aedocw/epub2tts-chatterbox/issues) to see if anyone else has run into something similar previously.

If you've found something new, please open an issue and be sure to include:
1. The full command you executed
2. The platform (Linux, Windows, OSX, Docker)
3. Your Python version if not using Docker

</details>

## 🗒️ Release notes
<details>
<summary>Release notes </summary>

* 20250224: Changed to read individual setences rather than entire paragraph, for reading speed consistency
* 20250221: Added `--notitles` option
* 20250216: Initial release

</details>

## 📦 Install

**Preferred method:** Use Docker (avoids Python dependency issues).

*NOTE:* If you want to specify where NLTK tokenizer will be stored (about 50mb), use an environment variable: `export NLTK_DATA="your/path/to/nltk_data"`

### Docker Installation (Recommended)

Docker provides a consistent environment with Python 3.11 and all dependencies pre-installed.

**Build the Docker image:**
```bash
docker build -t epub2tts-chatterbox .
```

**Convert an EPUB file:**
```bash
docker run --rm -v $(pwd):/data epub2tts-chatterbox /data/mybook.epub
```

**Convert with voice cloning:**
```bash
docker run --rm -v $(pwd):/data epub2tts-chatterbox /data/mybook.txt --sample /data/voice.wav --cover /data/cover.png
```

**Convert without chapter titles:**
```bash
docker run --rm -v $(pwd):/data epub2tts-chatterbox /data/mybook.txt --notitles
```

**All Docker options:**
- `--rm` - Remove container after exit
- `-v $(pwd):/data` - Mount current directory to /data in container
- First run will download Chatterbox TTS model (~10GB)

### MAC INSTALLATION

This installation requires Python 3.11 (specifically) and [Homebrew](https://brew.sh/) (I use homebrew to install espeak, [pyenv](https://stackoverflow.com/questions/36968425/how-can-i-install-multiple-versions-of-python-on-latest-os-x-and-use-them-in-par) and ffmpeg).

```
#install dependencies
brew install mecab espeak pyenv ffmpeg
#install epub2tts-chatterbox
git clone https://github.com/aedocw/epub2tts-chatterbox
cd epub2tts-chatterbox
pyenv install 3.11
pyenv local 3.11
#OPTIONAL - install this in a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install .
```

### LINUX INSTALLATION

These instructions are for Ubuntu 24.04.1 LTS and 22.04  (20.04 showed some depedency issues), but should work (with appropriate package installer mods) for just about any distro. Ensure you have `ffmpeg` installed before use. Requires Python 3.11.

```
#install dependencies
sudo apt install espeak-ng ffmpeg python3-venv python3.11
#clone the repo
git clone https://github.com/aedocw/epub2tts-chatterbox
cd epub2tts-chatterbox
#OPTIONAL - install this in a virtual environment using Python 3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install .
```

### WINDOWS INSTALLATION

Running epub2tts in WSL2 with Ubuntu 22 is the easiest approach, but these steps should work for running directly in windows.

(TBD)

</details>


## Updating

<details>
<summary>UPDATING YOUR INSTALLATION</summary>

1. cd to repo directory
2. `git pull`
3. Activate virtual environment you installed epub2tts in if you installed in a virtual environment using "source .venv/bin/activate"
4. `pip install . --upgrade`
</details>


## Author

👤 **Christopher Aedo**

- Website: [aedo.dev](https://aedo.dev)
- GitHub: [@aedocw](https://github.com/aedocw)
- LinkedIn: [@aedo](https://linkedin.com/in/aedo)

👥 **Contributors**

[![Contributors](https://contrib.rocks/image?repo=aedocw/epub2tts-chatterbox)](https://github.com/aedocw/epub2tts-chatterbox/graphs/contributors)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!\
Feel free to check the [issues page](https://github.com/aedocw/epub2tts-chatterbox/issues) or [discussions page](https://github.com/aedocw/epub2tts-chatterbox/discussions).

## Show your support

Give a ⭐️ if this project helped you!
