# Audio fixtures

Four recordings from NASA's historical sounds page
(https://www.nasa.gov/historical-sounds/), works of the United States
government and so in the public domain in the United States. NASA's media
usage guidelines ask that NASA material not be used to imply endorsement;
nothing here does.

Each was fetched, checked against the SHA-256 below, decoded by llama.cpp's
own decoder to 16 kHz mono, cut to its first 20 seconds and written as 16-bit
PCM by `tools/make_reference_audio.sh`.

| File | Source | SHA-256 of the source |
|---|---|---|
| `kennedy.wav` | "JFK: We Choose the Moon", `586447main_JFKwechoosemoonspeech.mp3`, first 20 s | `b2e6f87c7254870ce6183ffaff839d255cef00a8c53aefd26d0d3898f33b7949` |
| `christmas.wav` | "Apollo 8: Merry Christmas", `581549main_Apollo-8_Merry-Christmas.mp3` | `a551004c3a02af8f1afb517d218485383306729cfc735f54296a5bb9dbb051cd` |
| `countdown.wav` | "STS-135: Countdown to Launch", `590318main_ringtone_135_launch.mp3`, first 20 s | `9132e401834205b9bd0dbf93e2ab895516932975cc8de0b5cc50aee54eeb3d98` |
| `sputnik.wav` | "Sputnik: Beep", `578626main_sputnik-beep.mp3` | `84ab44b58396913c75ba0d8f5c29b18bca382269a10527bcf90a39c5c1460215` |

The files are at `https://www.nasa.gov/wp-content/uploads/2015/01/<name>`.

`sounds.h` computes the synthetic sounds from their sample index; `wav.h`
reads the recordings.
