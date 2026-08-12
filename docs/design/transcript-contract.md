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
  "result_digest": "sha256-of-the-canonical-identity-bytes",
  "status": "completed",
  "source": {
    "name": "interview.wav",
    "relative_path": "sessions/interview.wav",
    "duration_ms": 92000,
    "sha256": "sha256-of-the-exact-source-bytes"
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
    "runtime": {
      "name": "transformers",
      "version": "5.x",
      "environment": {
        "python_version": "3.11.14",
        "python_implementation": "CPython",
        "operating_system": "darwin",
        "machine": "arm64"
      }
    },
    "requested_device": "auto",
    "effective_device": "mps",
    "precision": "float16"
  },
  "operation": {
    "parameters": {
      "language": "es",
      "timestamps": true,
      "translate": true,
      "chunk_length": 30,
      "stride_length": 5,
      "requested_device": "auto",
      "effective_device": "mps",
      "precision": "float16"
    },
    "prepared_media": {
      "used": true,
      "sha256": "sha256-of-the-exact-wav-passed-to-the-runtime",
      "media_format": "wav",
      "sample_rate_hz": 16000,
      "channels": 1,
      "converter": {"name": "ffmpeg", "version": "ffmpeg version 8.0"}
    },
    "source_hash_policy": "sha256_source_bytes"
  }
}
```

Required top-level fields are `schema_version`, `status`, `source`,
`transcripts`, `speakers`, `diagnostics`, `provenance`, `trust`, and
`agent_eligibility`. Optional scalar
values are `null`; optional collections are empty arrays. Absolute source paths
are excluded by default. An explicit diagnostic mode may add one. Mun hashes the
complete source bytes before transcription. The `sha256_source_bytes` policy
means the digest identifies byte content independently of the source path.

`operation.parameters` records every `TranscriptionOptions` value that can affect
inference, together with the runtime's effective device and precision. The
prepared-media record describes the exact audio input passed to the speech
runtime. For a directly usable 16 kHz mono PCM WAV, `used` is `false`, its digest
matches the runtime input, and `converter` is `null`. When Mun converts media,
`used` is `true`, the digest covers the exact temporary WAV after FFmpeg exits,
and the record includes WAV format, sample rate, channels, and the FFmpeg version.
Temporary paths and names are never recorded.

Runtime environment fields are limited to stable, non-secret replay facts:
Python version and implementation, operating-system identifier, and machine
architecture. They do not contain usernames, home directories, environment
variables, hostnames, absolute paths, or secrets.

## Typed trust and agent eligibility

Trust is additive provenance, not a safety claim. Every canonical result records:

- media as `untrusted_bytes`;
- model as `verified_artifact` or `unsafe_remote_code`;
- machine content as `untrusted_model_output`; and
- agent eligibility as `ineligible` with a human-judgment reason.

Artifact verification means local bytes match the pinned manifest. It does not make model behavior or transcript text trusted. A corrected transcript remains `untrusted_content`, remains agent-ineligible, and adds review state, correction-set ID, and correction-set digest. Loaders accept legacy schema-version-1 results by assigning the conservative defaults. No transition can remove media or content taint, including results produced by unsafe remote code.

## Machine result identity

`result_digest` is the SHA-256 digest of canonical UTF-8 JSON bytes for the
machine result. Canonicalization sorts object keys, uses compact JSON syntax,
and normalizes CRLF and CR string newlines to LF. Array order remains
significant. The canonical identity excludes the `result_digest` claim itself
and observational receipt fields named `created_at`.

The identity therefore covers source identity, the exact prepared input,
model repository, revision and verified artifact-manifest digest, runtime and
stable environment facts, inference parameters, every transcript variant,
speakers, status, overlap, and diagnostics. Changing any of those facts changes
the digest. Equivalent results recorded at different creation times retain the
same digest.

Readers may continue to accept legacy schema-version-1 records without a
`result_digest`. When a digest is present, loaders must recompute it and reject
a mismatch as a typed validation failure. Additive unknown fields participate
in identity and remain compatible with the schema rule below.

Result identity proves only that a claimed record is consistent with its
recorded derivation inputs and outputs. It does not prove recognition
correctness, semantic accuracy, authenticity, or truth.

## Immutable correction overlays

Human corrections are stored separately from the canonical machine result. Mun
never edits the machine-result object or JSON. A schema-version-1 correction set
contains:

```json
{
  "schema_version": 1,
  "correction_set_id": "review-2026-08-12-a",
  "created_at": "2026-08-12T19:00:00Z",
  "parent_result_digest": "exact-machine-result-digest",
  "review_state": "reviewed",
  "corrections": [
    {
      "transcript_kind": "original",
      "segment_id": "segment_1",
      "original_text_digest": "sha256-of-the-exact-original-segment-text",
      "replacement": "Corrected text.",
      "note": "Optional note, at most 500 characters."
    }
  ]
}
```

`review_state` is explicitly `reviewed` or `unreviewed`. Targets are unique
within a correction set. Application requires an exact parent digest, an
existing transcript-kind and segment-ID pair, and a matching SHA-256 digest of
the original segment's UTF-8 text. Any mismatch rejects the complete overlay;
Mun does not partially apply it. Replacement text and notes remain untrusted
data and are never interpreted as markup, commands, or proof.

A machine-view JSON export explicitly records `view: "machine"`,
`review_state: "unreviewed"`, and the unchanged machine `result_digest` as its
export digest. A corrected JSON export is a derived envelope with
`view: "corrected"`, the
review state, parent result digest, correction-set ID and digest, a corrected
transcript projection, and its own `export_digest`. It is distinct from the
machine result and does not replace or acquire the machine `result_digest`.
The correction-set digest covers its ID, timestamp, parent, review state, and
all targets and replacement data using stable JSON encoding.
Corrected TXT, SRT, and VTT select replacement text while retaining the machine
segments' IDs, ordering, speakers, and timing intervals. The variant-level text
is rebuilt from the corrected segments in source order.

Human review records that a person handled the correction set. It does not
establish truth, authenticity, honesty, semantic accuracy, consent, custody, or
producer identity.

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
  time. Unavailable values remain `null`. Creation time is observational and
  does not participate in result identity.
- SHA-256 digests establish byte identity only. They do not establish
  authenticity, consent, custody or continuity, or semantic correctness.

Schema version 1 permits additive fields. Consumers must ignore unknown fields.
Removing a field, changing its meaning or type, or tightening previously valid
values requires a new integer schema version.

## Batch result and exit status

A batch machine result contains `schema_version`, an ordered `files` array of
`TranscriptResult` records, and summary counts by file status. Processing
continues after a file failure. Completed outputs remain preserved on failure
or cancellation.

The CLI exits 0 when every requested file completes, 1 when any file is partial,
failed, or cancelled, and 2 for command-line usage errors. It writes diagnostics
to stderr and reserves stdout for the requested machine result.

## Export transaction receipts

Each completed transcript result is exported as a per-source transaction. Mun
renders every requested projection into a mode-0700 staging directory beside
the destinations, hashes and validates staged content, refuses any existing
destination unless overwrite was explicitly requested, and commits in sorted
destination-path order. The staging directory is removed after success,
pre-commit failure, or cancellation.

Mun atomically replaces each individual file, but does not claim multi-file
filesystem atomicity. A receipt named `<output-base>.receipt.json` records
`completed`, `cancelled`, `failed_before_commit`, or `partial_commit`, the staged
artifact hashes and sizes, and exact committed and uncommitted destination
paths. `partial_commit` means at least one destination was committed before a
later commit failed. A workflow-boundary interruption is reported as cancelled
with exit status 1 after its receipt has been flushed; previously completed
sources remain committed.

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
  splitting, or line wrapping. Machine view retains machine text; corrected
  view selects a validated correction overlay. Cues retain source order and
  may overlap. A speaker label, when present, prefixes cue text as
  `[speaker_1] `. With English translation requested, Mun writes `.original`
  and `.en` files only for variants that contain timed segments.

JSON, JSONC, and Markdown each contain all transcript variants in one file.
Requesting a renderer that cannot represent available data fails before commit,
so no projection from that source reaches a final path. Existing files are never
overwritten without explicit permission.
