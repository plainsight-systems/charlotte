# Video fixtures

Four clips of a few seconds, cut by `tools/make_video_fixtures.sh` from videos
NASA publishes in its Image and Video Library, for the search page's video
check (`web/search.js`, `src/core/runtime/video.h`). Each is 640 × 360 at 30
frames a second, H.264, with no sound, cut with FFmpeg 9.0.2 and libx264.

NASA material is not protected by copyright in the United States unless
noted otherwise (NASA's media usage guidelines,
<https://www.nasa.gov/nasa-brand-center/images-and-media/>). Its use here
implies no endorsement by NASA.

| Clip | What it shows | Source | From | SHA-256 of the clip |
| --- | --- | --- | --- | --- |
| `launch.mp4` | Artemis I lifting off its pad at night, 8 s | [Artemis I Launch (Press Site)](https://images.nasa.gov/details/Artemis%20I%20Launch%202022%20CU%20tracking%20from%20Press%20Site_compressed), MSFC, 2022-11-16 | 0:46 | `a868b97aa467a0babc639e9d14c9bdb01030eac1932e0f349a158c10f176e97b` |
| `spacewalk.mp4` | Astronauts working outside the space station, 7 s | [First All-Woman Spacewalk](https://images.nasa.gov/details/NHQ_2019_1018_all%20woman%20spacewalk), HQ, 2019-10-24 | 0:13 | `0dc5ec2b387580197501721a40d5d58e5ceb67d525464663bf75fca8cfc6a06e` |
| `landing.mp4` | Perseverance lowered to Mars on cables, dust rising, 8 s | [NASA Provides First Front-Row Seat to Perseverance Rover's Landing on Mars](https://images.nasa.gov/details/NHQ_2021_0222_VF_NASA%20Provides%20First%20Front-Row%20Seat%20to%20Perseverance%20Rover%E2%80%99s%20Landing%20on%20Mars), HQ, 2021-02-22 | 3:49 | `bfc75e9fc788f7ac6927ebadf5d29cf11e85a1cec7ec1255053a8738de830a81` |
| `earth.mp4` | Earth's curve and clouds seen from orbit, 8 s | [Earth Views from the International Space Station](https://images.nasa.gov/details/Earth%20Views%20from%20the%20International%20Space%20Station), HQ, 2019-04-30 | 0:13 | `1d5c46b901d6e88576bde2bc32e4b4f1be5303e61b39839b8eed32399f8bbf0a` |

Each source is the library's `~medium.mp4`, 1,280 × 720, pinned in the script
by its SHA-256.
