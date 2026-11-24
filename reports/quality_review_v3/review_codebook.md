# RetinaSaarthi — Blinded Image Quality Review

## Instructions

Review the original-resolution appearance of each displayed image.
Do not use DR grade, model predictions, or quality metric values
to make the human assessment.

## Image quality
- good: visually suitable for screening
- borderline: some degradation, usability uncertain
- poor: substantial image-quality defect
- uncertain: cannot assess confidently

## Field completeness
- complete: retinal field appears adequately captured
- partially_clipped: part of the retinal field appears cut off
- obstructed: a region is obscured
- uncertain: cannot determine

## Field size / framing
- small: relatively small retinal field in the frame
- typical: framing appears reasonable
- large: retinal field occupies most of the frame
- uncertain: cannot determine

Field size is not the same as field completeness.

## Blur at native resolution
- yes: visible blur
- no: no obvious blur
- uncertain: cannot determine

Inspect vessels and fine structures at native resolution.

## Usability decision
- usable
- recapture
- manual_review

A mask-detection failure alone must not automatically mean recapture.

## Defect tags

Use semicolon-separated tags where applicable:
dark; overexposed; glare; colour_cast; haze; blur;
uneven_illumination; clipped; obstructed; lesion_heavy; other

Lesion-heavy is a visual observation, not a diagnosis.

## Notes

Describe visible reasons for borderline, recapture, or manual review.
Do not infer DR grade from this review.
