The experiment is divided into stages, each stage is described by its own .ipynb or .md file.

Stages exchange data through input/output .json files.

## Stages

1. `1.collect-dataset.ipynb` Collect recognition models hypothesis for audios.
  - Audios are taken from LibriSpeech dev-other, test-other sets.
  - For recognition, the original audio and its noisy variants are taken.
  - Outputs:
    - `recognitions.json` – result dataset
      - `record` audio file name
      - `noise_level` audio noise level
      - `model` recognition model
      - `reference` original text
      - `hypothesis` recognition model hypothesis
    - `wers.json` – recognition models WERs
      - `model` recognition model
      - `noise_level` audio noise level
      - `wer` WER value
2. `2.collect-assessors-evaluations.md` Use crowdsourcing platform to collect SbS and meaning loss evaluations of texts.
  - Inputs:
    - `recognitions.json`
  - Outputs:
    - `votes_sbs_raw.json` – raw markup for SbS recognition hypotheses comparison between two noise levels
      - `model` recognition model
      - `noise_left` audio noise level (for audio located left in task interface)
      - `noise_right` audio noise level (for audio located right in task interface)
      - `reference` original text
      - `hypothesis_left` recognition model hypothesis (for audio located left in task interface)
      - `hypothesis_right` recognition model hypothesis (for audio located right in task interface)
      - `choice` LEFT or RIGHT – which hypothesis is better in the user's opinion
    - `votes_sbs.json` – aggregated (i.e. by majority) votes for SbS recognition hypotheses comparison between two noise levels
      - `model` recognition model
      - `less_noise_idx` audio noise level (less one)
      - `more_noise_idx` audio noise level (more one)
      - `less_noise_votes` votes for less noisу hypotheses
      - `more_noise_votes` votes for more noisу hypotheses
    - `votes_check_raw.json` – raw markup for meaning preservance or loss between recognition hypothesis and reference text
      - `model` recognition model
      - `noise` audio noise level
      - `reference` original text
      - `hypothesis` recognition model hypothesis
      - `ok` TRUE if hypothesis preserves meaning of reference in the user's opinion, or FALSE otherwise
    - `votes_check.json` - aggregated (i.e. by majority) votes for meaning preservance or loss between recognition hypothesis and reference text
      - `model` recognition model
      - `level_idx` audio noise level
      - `votes_total` total votes
      - `votes_ok` votes for "meaning is preserved"
3. `3.learn-mera.ipynb`
  - Inputs:
    - `votes_sbs_raw.json`
    - `votes_check_raw.json`
  - Outputs:
    - `mera_sbs.json`
      - `votes_sbs_raw.json` with MERa values measured as `mera(reference, hypothesis_left)` and `mera(reference, hypothesis_right)`
    - `mera.json`
      - `votes_sbs_raw.json` with MERa values measured as `mera(reference, hypothesis)`
4. `4.plot-graphs.ipynb` Metrics alignment with assessors evaluations plots.
  - Inputs:
    - `wers.json`
    - `votes_sbs.json`
    - `votes_check.json`
    - `mera.json`
    - `mera_sbs.json`

## Noise level

Noise level is a real number from 0 to 1, indicating the proportion of the noise tracks relative to the original audio (larger number means more noisy audio).
In input/output files noise levels are presented as their indices in noise levels array for computational difficulties avoidance.
`null` noise level corresponds to original audio without changes.