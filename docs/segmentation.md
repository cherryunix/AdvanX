# Pose-driven segmentation

`tools/segment_pose_cache.py` converts AdvanX's aligned 24 fps arrays into
non-overlapping source intervals. It works entirely from the numerical cache and
does not decode, cut, or re-encode the training videos.

## Signals

Every frame is checked for:

- face presence and face confidence;
- rigid head-fit error;
- smoothed pitch, yaw, and roll relative to the source calibration;
- smoothed horizontal and vertical iris displacement;
- eye openness relative to that source's 80th percentile;
- face scale and position in the frame.

Abrupt changes in face center, face scale, or head rotation create hard boundaries.
Short failures such as ordinary blinks can be bridged, but a bridge never crosses a
hard boundary. Continuous runs are divided into near-equal intervals whose lengths
stay within the selected profile.

## Profiles

| Profile | Length | Intended use |
| --- | --- | --- |
| `loose` | 6–24 s | Recover nearly all usable footage and establish cut boundaries. |
| `standard` | 8–20 s | Main training pool with stable face, gaze, framing, and head pose. |
| `strict` | 8–16 s | Small seed or evaluation pool with conservative thresholds. |

The score ranks intervals inside a profile. It combines raw threshold coverage,
face and body coverage, centered head pose, centered gaze, and distance from hard
boundaries. A higher score does not prove that the background, speech, or content is
suitable.

## Duplicate handling

Container hashes miss videos that contain the same decoded frames but were copied
or re-muxed. Before segmentation, the tool hashes the exact target timeline and a
small set of pose signals. Sources with identical pose trajectories are recorded in
`duplicates.json` and excluded from candidate totals so they do not receive extra
training exposure.

## Review report

`tools/build_segment_review.py` keeps the best interval from every source for the
overview. Each thumbnail is a strip sampled at 15%, 50%, and 85% of that interval.
The HTML page can play the exact source range, filter by path and shot size, store
keep/drop decisions locally, and export them as JSON. Full interval manifests stay
available as JSON and UTF-8 CSV.

## Human calibration

`tools/reclassify_segments.py` treats exported keep/drop decisions as supervised
labels. It fits a one-dimensional decision boundary to the P95 absolute smoothed
yaw, the dominant interpretable feature in the review set, and reports stratified
cross-validation before applying the rule.

The output has three classes:

- `keep` and `drop` for labelled intervals and unlabelled intervals outside the
  boundary margin;
- `review` for unlabelled intervals close to the learned boundary.

Every row records the fitted threshold, distance from the threshold, suggested
binary class, final class, and whether that result came from a human decision, the
automatic rule, or the review band. `tools/build_review_queue.py` creates a focused
offline page for the remaining boundary intervals.

## Known limit

The current cache follows the primary subject. It cannot reliably reject a second
person shown on a television, poster, or distant background when that person was
not recalled by the detector. Review those conditions from the source video or add
a dedicated multi-person/background detector before treating them as automatic
exclusions.
