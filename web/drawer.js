// Axis H. On a narrow screen the models panel is a drawer from the left,
// so the chat has the screen: the menu button opens it, and the scrim, its
// close button or Escape closes it. Where the screen is wide the panel sits
// beside the chat, and none of this applies (style.css).
//
// While the drawer is closed on a narrow screen it is inert, so focus and
// assistive technology skip what cannot be seen. Opening moves focus to the
// drawer itself, so the next Tab reaches its first control and no ring is
// drawn on a tap; closing returns focus to the menu button where focus was
// inside.

const kNarrow = '(max-width: 720px)';

export function createDrawer({ drawer, toggle, scrim, close, root }) {
  const narrow = globalThis.matchMedia(kNarrow);
  let open = false;

  const apply = () => {
    root.classList.toggle('drawer-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    scrim.hidden = !open;
    drawer.inert = narrow.matches && !open;
  };
  const show = () => {
    open = true;
    apply();
    drawer.focus({ preventScroll: true });
  };
  const hide = () => {
    const hadFocus = drawer.contains(document.activeElement);
    open = false;
    apply();
    if (hadFocus && narrow.matches) toggle.focus();
  };

  toggle.addEventListener('click', () => (open ? hide() : show()));
  scrim.addEventListener('click', hide);
  close.addEventListener('click', hide);
  document.addEventListener('keydown', (event) => {
    if (open && event.key === 'Escape') hide();
  });
  narrow.addEventListener('change', () => {
    if (!narrow.matches) open = false;
    apply();
  });
  apply();

  return { open: show, close: hide };
}
