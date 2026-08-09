# Canonical transcript and output contract

Accepted for Mun 1.0 on 2026-08-09.

## Canonical record

Mun produces one `TranscriptResult` for each source media file. JSON is the
canonical representation; every other format is a deterministic projection of
the same record. Missing capabilities use absent or `null` fields rather than
invented values.

```json
{
  "schema_version": 1,
  "status": "completed",
  "source": {
    "name": "interview.wav",
    "relative_path": "sessions/interview.wav",
    "duration_ms": 92000,
    "sha256": null
  },
  "transcripts": [
    {
      "kind": "original",
      "language": {
        "tag": "es",
        "source": "detected",
        "confidence": 0.98
      },
      "text": "Buenos días.",
      "segments": [
        {
          "id": "segment_1",
          "start_ms": 0,
          "end_ms": 1400,
          "text": "Buenos días.",
          "speaker_id": "speaker_1",
          "words": [
            {"text": "Buenos", "start_ms": 0, "end_ms": 900},
            {"text": "días.", "start_ms": 900, "end_ms": 1400}
          ]
        }
      ]
    },
    {
      "kind": "english_translation",
      "language": {"tag": "en", "source": "model", "confidence": null},
      "text": "Good morning.",
      "segments": []
    }
  ],
  "speakers": [
    {
      "id": "speaker_1",
      "assigned_speech_ms": 1400,
      "speaking_ms": 1400
    }
  ],
  "overlap_ms": 0,
  "diagnostics": [],
  "provenance": {
    "mun_version": "1.0.0",
    "created_at": "2026-08-09T20:00:00Z",
    "model": {
      "repository": "owner/model",
      "revision": "immutable-commit-sha",
      "artifact_sha256": null
    },
    "runtime": {"name": "transformers", "version": "5.x"},
    "requested_device": "auto",
    "effective_device": "mps",
    "precision": "float16"
  }
}
```

Required top-level fields are `schema_version`, `status`, `source`,
`transcripts`, `speakers`, `diagnostics`, and `provenance`. Optional scalar
values are `null`; optional collections are empty arrays. Absolute source paths
are excluded by default. An explicit diagnostic mode may add one, and source
hashing is opt-in because it reads the complete media file.

## Invariants

- Status is `completed`, `partial`, `failed`, or `cancelled`. `partial` means at
  least one useful transcript or requested rendering exists but the requested
  work is incomplete. `failed` contains no useful transcript.
- Times are non-negative integer milliseconds on the source-media timeline.
  Intervals are half-open: `start_ms` is inclusive and `end_ms` is exclusive.
  Words fall within their segment. Arrays retain model order when times tie.
- Language tags are nullable BCP 47 tags. Language source is `detected`,
  `forced`, `model`, or `unknown`; confidence is either `null` or a value from
  0 through 1.
- Original speech and English translation are separate transcript variants.
  They share the source timeline, but segments or words are present only when
  the producing runtime supplies them; Mun never invents alignment.
- Speaker IDs are anonymous and stable only within one transcript result.
  `assigned_speech_ms` sums assigned segment durations, while `speaking_ms`
  measures the union of that speaker's intervals. `overlap_ms` is the duration
  where multiple speakers are active.
- A diagnostic has `severity` (`warning` or `error`), stable `code`, human
  `message`, processing `stage`, and `recoverable`. It contains no traceback,
  secret, username, home path, or absolute source path by default.
- Provenance identifies the exact model revision, derived artifact when used,
  runtime, requested and effective device, precision, Mun version, and creation
  time. Unavailable values remain `null`.

Schema version 1 permits additive fields. Consumers must ignore unknown fields.
Removing a field, changing its meaning or type, or tightening previously valid
values requires a new integer schema version.

## Batch result and exit status

A batch machine result contains `schema_version`, an ordered `files` array of
`TranscriptResult` records, and summary counts by file status. Processing
continues after a file failure. Completed outputs remain preserved on failure
or cancellation.

The CLI exits 0 when every requested file completes, 1 when any file is partial
or failed, and 130 when cancelled. It writes diagnostics to stderr and reserves
stdout for the requested machine result.

## Deterministic renderers

All text files use UTF-8 and LF endings.

- **TXT** contains only a transcript variant's trimmed text and one final
  newline. It is the default. With English translation requested, Mun writes
  `<base>.original.txt` and `<base>.en.txt`.
- **Markdown** is one report with source metadata, original transcription,
  optional English translation, optional timed segments/speakers, diagnostics,
  and compact provenance. Absent optional sections are omitted.
- **JSON** contains the complete canonical `TranscriptResult` and is the machine
  interchange format.
- **JSONC** contains the same data and field meanings as JSON with stable
  explanatory comments. It is for people and configuration-aware tools, not
  the CLI machine protocol.
- **SRT** and **VTT** project one cue per timed segment without merging,
  splitting, line wrapping, or text correction. Cues retain source order and
  may overlap. A speaker label, when present, prefixes cue text as
  `[speaker_1] `. With English translation requested, Mun writes `.original`
  and `.en` files only for variants that contain timed segments.

JSON, JSONC, and Markdown each contain all transcript variants in one file.
Requesting a renderer that cannot represent available data records a diagnostic
and makes that file result `partial`; other requested renderings are still
written atomically. Existing files are never overwritten without explicit
permission.
