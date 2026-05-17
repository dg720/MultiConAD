# ASR Robustness Code

This folder holds MultiConAD ASR robustness code and reference implementations copied from the ADReSS-focused `mlmi-thesis-private` repository.

## Structure

```text
processing/asr_robustness/
  mlmi_reference/
    transcribe_asr_alternative.py
    build_asr_transcript_variants.py
    build_participant_selection_ablation.py
    build_whisperx_prompt_pause_ablation.py
    build_whisperx_postprocess_ablation.py
    generate_mfa_pause_encoded_transcripts.py
    generate_pause_encoded_transcripts.py
    compute_asr_wer.py
```

The files under `mlmi_reference/` are copied for methodology alignment. They may still import `mlmi_thesis.paths` or assume ADReSS-specific train/test directories, so treat them as reference code until they are adapted into MultiConAD-specific runners.

Canonical methods to preserve when adapting:

- plain Whisper is both-speaker only;
- WhisperX single-speaker uses the longest diarized speaker, not `SPEAKER_00`;
- prompt/interviewer utterance cleanup happens before pause encoding;
- ASR pause encoding uses `.` for gaps `>= 0.5s` and `< 2.0s`, and `...` for gaps `>= 2.0s`;
- WER/CER scoring must normalize reference and ASR text consistently without mutating the original files.

