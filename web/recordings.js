// Axis H. The recordings chosen for the search panel's index (search.js),
// for a model whose catalog entry names an audio companion: an "Add
// recordings" button and the panel as a drop target for audio files, each
// listed with its name and length and a button to remove it. The photo
// picker's twin (photos.js), for its own kind.
//
// createRecordings({ onChange, decode }) returns { element, recordings(),
// dropOnto(target), release() }; decode is audio_input.js's unless given.
//   - A recording is checked as it is added: decoded on the page and its
//     length kept, its samples dropped, so one the browser can't decode, or
//     longer than 10 minutes, is named and not added, rather than ending an
//     indexing later. It is decoded again when it is indexed: holding every
//     added recording's samples would hold up to 38 MB each.
//   - Each entry keeps its file and an object URL for it, which the search
//     panel's results play from at the clip's start (an <audio> element,
//     the URL's media fragment #t=start,end), kept until the recording is
//     removed or release() is called.
//   - onChange() follows every addition and removal: the index no longer
//     holds the recordings shown.
//   - The panel's drop target takes images to photos.js and audio to this
//     file, each by its MIME type.
