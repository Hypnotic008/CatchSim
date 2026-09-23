# Portfolio site

The complete static site for dalenwagga-eng-portfolio.netlify.app. This folder
is the Netlify publish directory: deploy it as-is.

```
index.html                  homepage: projects, experience, education, tools
404.html
_headers                    Netlify headers (video streaming, PDF, CSV types)
assets/css/site.css         shared styles
assets/resume/              Dalen_Wagga_Resume.pdf
projects/catchsim/          page + assets/{images,videos,interactive,data}
projects/water-deluge/      page + assets/images
projects/forward-flap/      page + assets/images
```

Each project keeps its media in its own `assets/` folder, and every link is
relative, so the site also works opened straight from disk.

## Deploying

- **Netlify CLI:** `netlify deploy --prod --dir portfolio` (from the repo root,
  linked to the existing site with `netlify link`).
- **Drag and drop:** in the Netlify dashboard, open the site, go to Deploys,
  and drop the whole `portfolio` folder. Don't drop individual files.
- **Git-linked:** set the base/publish directory to `portfolio`.

## Keeping media intact

- The repo's `.gitattributes` marks images, video and PDF as `binary`, so git
  never converts line endings inside them.
- Don't move the media into Git LFS on a Git-linked site unless Netlify's LFS
  support is on; otherwise Netlify serves LFS pointer text instead of files.
- After deploying, `curl -I <url>` a file: `Content-Length` should match the
  local size (`catch3d.mp4` is 30,706,900 bytes).
