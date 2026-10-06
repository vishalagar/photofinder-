# photofinder

A local face finder for a big folder of group/trip photos. Upload a selfie (or pick a person
from the auto-detected group) and see every photo they appear in. Nothing leaves your machine.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Download the photos from Google Drive into a folder (e.g. `photos/`), then:

```bash
python photofinder.py index photos/   # one-time; downloads the face model (~300MB) on first run
python photofinder.py serve photos/   # open http://127.0.0.1:5000
```

## Tuning
In `photofinder.py`: raise `MATCH_THRESHOLD` (default 0.40) if you see wrong people, lower it if
you miss photos. `CLUSTER_EPS` controls how aggressively faces are grouped into "people".
