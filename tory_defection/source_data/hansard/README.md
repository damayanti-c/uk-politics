# Speech Analysis Snowflake Flow

## Rebuild all_speeches_extended.csv from Snowflake (preferred)

If you do not want to keep a committed local copy, regenerate on demand:

```bash
conda run -n snowflake_env python tory_defection/source_data/hansard/build_all_speeches_extended_from_snowflake.py --overwrite
```

## Upload Hansard Data

This uploads:
- `source_data/hansard/all_speeches_extended.csv` (as `INGEST_TYPE='speech_row'`)
- `source_data/hansard/*.xml` (as `INGEST_TYPE='xml_chunk'`)

Target:
- `GOOGLE_AI_EIR.SPEECH_ANALYSIS.HANSARD`

Run:

```bash
conda run -n snowflake_env python tory_defection/source_data/hansard/upload_hansard_to_snowflake.py
```

Optional reset:

```bash
conda run -n snowflake_env python tory_defection/source_data/hansard/upload_hansard_to_snowflake.py --truncate-table
```

## Fetch Hansard For Analysis

`analysis/final_model/training_speech_analysis.py` and `test_speech_analysis.py` now support:

- Local (default): `HANSARD_SOURCE=local`
- Snowflake: `HANSARD_SOURCE=snowflake`

Example:

```bash
set HANSARD_SOURCE=snowflake
conda run -n snowflake_env python tory_defection/analysis/final_model/training_speech_analysis.py
```

Optional local override path:

```bash
set HANSARD_DATA_PATH=C:\path\to\all_speeches_extended.csv
```
