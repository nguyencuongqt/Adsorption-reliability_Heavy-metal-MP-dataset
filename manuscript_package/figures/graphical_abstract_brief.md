# Graphical Abstract Brief

## Core message

Random row-level validation and single-method interpretation can overstate reliability in literature-derived adsorption ML.

## Suggested layout

- left panel: grouped dataset structure (`23 studies -> 149 experiments -> 1009 rows`)
- middle panel: validation hierarchy with a downward performance arrow from `random_cv` to `group_exp` to `group_aut`
- right panel: interpretation audit showing one trustworthy branch and one spurious branch exposed by synthetic null features

## Suggested callouts

- `Random CV is optimistic`
- `Study-level transfer is hardest`
- `Stable importance is not always reliable`

## Visual direction

- use three clean color families: blue for validation, orange for transfer difficulty, red for null-feature warnings
- keep text minimal and large enough to survive submission-system thumbnails
