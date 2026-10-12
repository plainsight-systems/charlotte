// Axis H. The videos chosen for the search panel's index (search.js), for a
// model whose catalog entry names a vision companion whose encoder reads
// video (runtime/video.h): an "Add videos" button and the panel as a drop
// target for video files, each listed with its name and length and a button
// to remove it. The recordings' twin (recordings.js), for its own kind.
//
// createVideos({ onChange, open }) returns { element, videos(),
// dropOnto(target), release() }; open is video_input.js's unless given.
//   - A video is checked as it is added: opened on the page and its size and
//     duration kept, the element released, so one the browser can't play,
//     with no picture, or longer than 10 minutes is named and not added,
//     rather than ending an indexing later. It is opened again when it is
//     indexed.
//   - Each entry keeps its file and an object URL for it, which the search
//     panel's results play from at the clip's start (a <video> element, the
//     URL's media fragment #t=start,end), kept until the video is removed or
//     release() is called.
//   - onChange() follows every addition and removal: the index no longer
//     holds the videos shown. A file refused adds nothing, so changes
//     nothing.
//   - The panel's drop target takes images to photos.js, audio to
//     recordings.js and video to this file, each by its MIME type.
