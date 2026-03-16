# Reliability Study Manuscript Package

This package is specific to the fresh `adsorption_reliability_study` layer. It is not derived from the legacy benchmark manuscript package.

## Purpose

- Main manuscript for the new study on grouped validation reliability and null-anchored interpretation reliability
- Supplementary information with additional tables and figures
- Editable tables and figures copied from rerun outputs in `adsorption_reliability_study/results`
- DOCX versions built from the package text sources

## Source of truth

- Data and outputs come from [adsorption_reliability_study/results](G:\My Drive\Multi_metal MP prediction\adsorption_reliability_study\results)
- Manuscript text sources live directly in this package
- Legacy project outputs outside `adsorption_reliability_study/` are not treated as current evidence for this package

## Build workflow

- `python adsorption_reliability_study/scripts/10_build_manuscript_package.py`
- `python adsorption_reliability_study/scripts/11_build_manuscript_docx.py`
