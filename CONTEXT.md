# Mun

Mun turns local source media into transcripts using locally installed speech models. Its language distinguishes media selection, model compatibility, and optional transcript enrichment.

## Language

**Source Media**:
An audio or video file selected for transcription.
_Avoid_: Input, recording

**Batch**:
The deduplicated, ordered collection of source media handled in one run.
_Avoid_: Job, queue

**Plain Transcript**:
The spoken text returned by a speech model without timestamps, speaker labels, translation, correction, or summarization.
_Avoid_: Raw transcript, basic transcript

**Transcript Result**:
The versioned record produced for one source media file, containing its status, transcript variants, provenance, and diagnostics.
_Avoid_: Output, response

**Transcript Variant**:
One rendering of the speech within a transcript result, such as the original-language transcript or English translation.
_Avoid_: Version, track

**Segment**:
A contiguous portion of a transcript variant associated with an interval on the source-media timeline.
_Avoid_: Chunk, cue

**Speaker**:
An anonymous, file-local participant label assigned by speaker diarization.
_Avoid_: Person, identity

**Speaker Diarization**:
The assignment of anonymous speaker labels and speaking times to transcript segments.
_Avoid_: Speaker recognition, speaker identification

**English Translation**:
An English rendering produced from source media whose spoken language is not English, preserved separately from the original-language transcript.
_Avoid_: Transcription, translated transcript

**Speech Model**:
A locally installed Hugging Face model used to turn speech into text.
_Avoid_: AI, engine

**Compatible Model**:
A speech model that the installed runtime can load and whose requested capabilities Mun can verify.
_Avoid_: Supported model

**Tested Model**:
A compatible model whose pinned revision and capabilities have been verified by the Mun project.
_Avoid_: Recommended model

**Capability**:
A verified behavior of a speech model, such as timestamps, language selection, language detection, or English translation.
_Avoid_: Feature
