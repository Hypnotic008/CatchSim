# Your own 3-D models

Drop a glTF binary here and the 3-D viewer (and the rendered video) uses it
instead of the built-in procedural model:

| file          | replaces                          | kept procedural                       |
|---------------|-----------------------------------|---------------------------------------|
| `booster.glb` | the whole booster                 | engine plumes and glow                |
| `tower.glb`   | tower lattice, top and drawworks  | the animated chopsticks and carriage  |

Both are fitted automatically. The longest axis is found, the booster is
scaled to 72.3 m and seated with its aft end at the engines, and the tower
is scaled to 146 m and stood on the ground at Tower 2's position and heading.
This works for most models exported from Blender (Y-up, `File > Export >
glTF 2.0`, format *glTF Binary*). If a model comes in the wrong way round, add
a `models.json` next to it:

```json
{
  "booster": { "flip": false, "roll": 0, "hide_bells": true, "scale": 1.0 },
  "tower":   { "yaw": 0, "has_arms": false, "height": 146 }
}
```

- `booster.flip`: swap nose and tail if the booster arrives upside down.
- `booster.roll`: degrees about the long axis. The middle grid fin must point
  away from the tower at the catch.
- `booster.hide_bells`: hide the procedural engine bells (default true), for
  models that have their own.
- `tower.yaw`: degrees about vertical, so the model's arms open the same way
  as the procedural ones.
- `tower.has_arms`: true if the file has its own arms. The procedural ones
  are then hidden too, and the arms no longer animate.
- `rotation: [x, y, z]` (degrees, either model) overrides the automatic
  alignment entirely. `"booster": false` or `"tower": false` ignores a file.

The single-file viewer (`figs/catch3d_viewer.html`) embeds models up to
60 MB. Check the model's licence before using or redistributing it.
Downloaded models are often free for personal use but not for
redistribution, so keep those out of a public repository.
