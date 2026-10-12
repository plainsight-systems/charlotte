// Axis H. On a phone, the page fits what can be seen: with the on-screen
// keyboard up, the visual viewport is about half the screen, but iOS keeps
// laying the page out for the whole of it and scrolls it away under the
// keyboard, header and all. So the page's height follows the visual
// viewport (--app-height, read by style.css on a narrow screen), and while
// the keyboard takes the screen's lower part, `keyboard-open` lets the page
// set aside what the chat does not need then: the notice and the footer.
// Since the page no longer scrolls to the field focused, a field within a
// panel that scrolls on its own — the search panel's query, below its
// lists — is scrolled into view within that panel once the keyboard is up,
// with the form around it.

// The visual viewport shorter than the tallest it has been at this width by
// more than this is the keyboard, not the browser's own bars coming and
// going. Not innerHeight: iOS shrinks it with the keyboard for a moment, and
// a comparison against it then reads no keyboard (measured on iOS 26).
const kKeyboardPx = 150;

export function fitToVisualViewport(root) {
  const viewport = globalThis.visualViewport;
  if (viewport === undefined || viewport === null) return;
  let width = viewport.width;
  let tallest = viewport.height;
  const fit = () => {
    if (viewport.width !== width) {   // turned: a new screen to measure
      width = viewport.width;
      tallest = viewport.height;
    }
    tallest = Math.max(tallest, viewport.height);
    root.style.setProperty('--app-height', `${Math.round(viewport.height)}px`);
    const keyboard = tallest - viewport.height > kKeyboardPx;
    root.classList.toggle('keyboard-open', keyboard);
    const field = document.activeElement;
    if (keyboard && field?.matches?.('input, textarea')) (field.closest('form') ?? field).scrollIntoView({ block: 'nearest' });
    // iOS scrolls the page to keep the field in view; sized to fit, it need not.
    if (globalThis.scrollY !== 0) globalThis.scrollTo(0, 0);
  };
  viewport.addEventListener('resize', fit);
  viewport.addEventListener('scroll', fit);
  fit();
}
