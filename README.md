# Portfolio drop-in: CatchSim

Everything the CatchSim project page needs, laid out to copy straight into
the portfolio site's publish folder:

```
projects/catchsim/
  index.html                       project page (relative asset paths)
  assets/
    images/   mission_overview.png, landing_detail.png, monte_carlo.png
    videos/   catch3d.mp4
    interactive/ catch3d_viewer.html  (interactive 3-D viewer)
    data/     monte_carlo.csv
_headers                           merge into the site's root _headers
homepage-card.html                 card to paste into the homepage
```

## Deploying

1. Copy `projects/` into the site's publish folder (next to its `index.html`).
2. Merge `_headers` into the site root's `_headers` (create it if missing).
3. Paste `homepage-card.html` into the homepage's project list and restyle
   the class names to match.
4. Deploy. Prefer `netlify deploy --prod --dir <publish folder>` or a Git-linked
   site. If you drag-and-drop in the Netlify UI, drop the whole folder, not
   individual files.

## Keeping media intact

- The repo's `.gitattributes` marks images and video as `binary`, so git never
  converts line endings inside them (a common cause of broken PNG/MP4 files).
- Don't put the media in Git LFS on a Git-linked Netlify site unless Netlify's
  LFS support is turned on; otherwise the site serves the small LFS pointer
  text files instead of the images.
- Check a deployed file with `curl -I <url>`: the `Content-Length` should match
  the local file size (`catch3d.mp4` is 30,706,900 bytes).
