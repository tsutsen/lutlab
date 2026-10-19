// lutlab docs: wire the Godot-style sidebar ourselves.
// Godot's custom.css hides every section (`.wy-menu-vertical p.caption + ul { display: none }`)
// and only shows `.active` ones; its own custom.js (jQuery, tied to Godot's templates) normally
// drives that. We re-wire the two collapse levels here, with no dependencies.
(function () {
  function init() {
    var menu = document.querySelector('.wy-menu-vertical');
    if (!menu) return;

    // A "section" is a <p class="caption"> immediately followed by a <ul> of items.
    var sections = [];
    menu.querySelectorAll('p.caption').forEach(function (cap) {
      var ul = cap.nextElementSibling;
      if (ul && ul.tagName === 'UL') sections.push({ cap: cap, ul: ul });
    });

    // On load, open the section that contains the current page.
    var current = menu.querySelector('li.current');
    if (current) {
      sections.forEach(function (s) {
        if (s.ul.contains(current)) {
          s.cap.classList.add('active');
          s.ul.classList.add('active');
        }
      });
    }

    // Click a section caption -> toggle that whole section.
    sections.forEach(function (s) {
      s.cap.addEventListener('click', function () {
        var open = s.ul.classList.toggle('active');
        s.cap.classList.toggle('active', open);
      });
    });

    // Click a per-page spoiler button -> toggle that item's submenu, without navigating.
    menu.querySelectorAll('button.toctree-expand').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var li = btn.closest('li');
        var sub = li ? li.querySelector(':scope > ul') : null;
        if (!sub) return;
        var open = getComputedStyle(sub).display === 'none';
        sub.style.display = open ? 'block' : 'none';
        btn.classList.toggle('open', open);
      });
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
