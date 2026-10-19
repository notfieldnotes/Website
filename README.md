# NotFieldNotes — Website

Custom Shopify theme code for **notfieldnotes.com** (`xdymd9-w3.myshopify.com`).

These are the hand-written sections and snippets added to the theme, kept here
as the source of truth. They are not a complete theme: the rest of the theme is
vendor code, and the per-page settings live in Shopify's own template JSON.

## Layout

    sections/   custom sections, one file each
    snippets/   customised theme snippets

## Sections

| File | What it does |
| --- | --- |
| `image-trail.liquid` | Skiper UI image cursor trail: images pop up along the pointer's path and fade away; uses image blocks or a collection's product photos |
| `awards-services.liquid` | The sxlvx.de sequence: pinned globe with a spinning 3D ring of award names that the camera zooms into, a cream circle opening to fill the screen, then a stickered heading and sticky stacking service rows with animated cards |
| `scroll-story.liquid` | The gsap.com/scroll opening: a paper plane loops through a pinned first scene and lands on the next line, a ball rides a glossy tube, then heading and buttons beside turning rings |
| `scramble-text.liquid` | GSAP ScrambleText demo: a sentence built from parts that scramble into place one after another, each with its own characters, duration and flicker speed, with a blinking cursor at the end |
| `product-view.liquid` | Pinned GSAP reel, after henrietta.wearetattoodigital.com: each product wipes its colour up, slides its name across as a huge wordmark, and swings the pack in and out |
| `product-scroller.liquid` | Row of product cards from a collection; pins and slides sideways with the page scroll on desktop, swipeable row on phones |
| `product-showcase.liquid` | Product hero card: big photo in front with 3D drift, product name as a backdrop wordmark, price, swatches, quantity stepper and buy buttons |
| `scrolling-logos.liquid` | Theme section, extended with a background image (opacity / size / position) and a square-corners option |
| `slide-menu.liquid` | MENU slides the page away on a tilt; each item cross-fades its own background |
| `scroll-images.liquid` | Scroll-driven image mosaic with captions and centre words |
| `string-trail.liquid` | A line running the length of the page with an object sliding along it |
| `scatter-scroll.liquid` | Parallax field of scattered shapes, each on a damped follow |
| `word-reveal.liquid` | Clip-and-rise reveal, word by word, on every heading |
| `smooth-scroll.liquid` | Eased (inertia) scrolling on the real document scroll |
| `preloader.liquid` | Wordmark preloader cycling through languages, with flicker |
| `page-backdrop.liquid` | One shared background image behind every section of a page |
| `back-link.liquid` | "« पीछे" back link in the top-left of every page |

## Snippets

| File | What it does |
| --- | --- |
| `cart-footer.liquid` | Cart and drawer footer (currently the stock theme version) |

## Working on these

The live code is on the Shopify theme, not here. Edit a file, then paste it into
**Shopify admin → Online Store → Themes → Edit code**, or push it with the
Shopify CLI. Never edit the published theme directly — work on a draft theme
and publish when it looks right.

Settings are defined in each file's `{% schema %}` block. Note that once a
section has been saved in the theme editor, Shopify stores those values in the
template JSON, and they win over any default changed here.
