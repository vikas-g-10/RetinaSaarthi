# Messidor-2 inspection

This inspection did not modify the original archive parts or [`messidor-2.csv`](../datasets/messidor2/messidor-2.csv). The only dataset-side files created were the requested extracted images under [`datasets/messidor2/extracted/`](../datasets/messidor2/extracted/).

## CSV structure and possible labels

The exact first 10 data rows and header were:

```text
left;right
20051109_57451_0400_PP.png;20051109_57843_0400_PP.png
20051109_59864_0400_PP.png;20051109_59751_0400_PP.png
20051109_60031_0400_PP.png;20051109_59969_0400_PP.png
20051110_33960_0400_PP.png;20051109_60136_0400_PP.png
20051110_34071_0400_PP.png;20051110_34013_0400_PP.png
20051110_35279_0400_PP.png;20051110_35239_0400_PP.png
20051110_35926_0400_PP.png;20051110_35332_0400_PP.png
20051110_38111_0400_PP.png;20051110_37992_0400_PP.png
20051110_38194_0400_PP.png;20051110_38156_0400_PP.png
20051110_38280_0400_PP.png;20051110_38239_0400_PP.png
```

The raw CSV has 875 lines including its header and 874 data rows. Its parsed fields are `left` and `right `; the trailing space in the second header is present in the file. Each column contains 874 distinct filename values, every value occurs once, and neither column is a numeric DR-grade field. Therefore this CSV does not expose a grading scale. The values appear to be image filenames only; meanings that cannot be verified from this file are not interpreted here.

**Source:** [`datasets/messidor2/messidor-2.csv`](../datasets/messidor2/messidor-2.csv), [`reports/inspect_messidor2.py`](inspect_messidor2.py).

## Archive parts, integrity, and extraction

The archive parts were:

| Part | Size in bytes |
|---|---:|
| `IMAGES.zip.001` | 734003200 |
| `IMAGES.zip.002` | 734003200 |
| `IMAGES.zip.003` | 734003200 |
| `IMAGES.zip.004` | 253175547 |

The numbering is consecutive from `.001` through `.004`. The first three parts have the same segment size and the final part is smaller, as expected for a final segment. The parts were concatenated in numeric order into a temporary file outside the dataset directory. The original parts were not overwritten.

Commands used:

```text
mkdir -p datasets/messidor2/extracted
cat datasets/messidor2/IMAGES.zip.001 datasets/messidor2/IMAGES.zip.002 datasets/messidor2/IMAGES.zip.003 datasets/messidor2/IMAGES.zip.004 > /tmp/vscode-agent-codex-sandbox-QhfVxS/messidor2_IMAGES.zip
7z t /tmp/vscode-agent-codex-sandbox-QhfVxS/messidor2_IMAGES.zip
7z x /tmp/vscode-agent-codex-sandbox-QhfVxS/messidor2_IMAGES.zip -odatasets/messidor2/extracted -y
```

The combined archive size was 2455185147 bytes. The integrity test output was:

```text
Testing archive: /tmp/vscode-agent-codex-sandbox-QhfVxS/messidor2_IMAGES.zip
--
Path = /tmp/vscode-agent-codex-sandbox-QhfVxS/messidor2_IMAGES.zip
Type = zip
Physical Size = 2455185147

Everything is Ok

Folders: 1
Files: 1748
Size:       2467041260
Compressed: 2455185147
```

Extraction reported the same archive totals and `Everything is Ok`:

```text
Everything is Ok
Files: 1748
Size:       2467041260
Compressed: 2455185147
```

The first attempt to write the temporary combined archive directly under `/tmp` was rejected by the execution environment as read-only; it did not alter any source file. The successful command used the writable temporary workspace shown above.

**Source:** archive listing and `file` output for [`datasets/messidor2/`](../datasets/messidor2/), plus the exact `7z t` and `7z x` outputs above.

## Extracted images and CSV correspondence

The extracted folder contains 1,748 images: 690 JPG files and 1,058 PNG files. All 1,748 images opened and verified successfully.

The width/height distribution is:

```text
1440x960=528; 2240x1488=616; 2304x1536=604
```

The independent correspondence check output was:

```text
csv_rows=874 csv_referenced_images=1748 csv_unique_images=1748
extracted_images=1748 extracted_unique_names=1748
images_opened_ok=1748 open_errors=0
exact_csv_not_in_extracted=[]
exact_extracted_not_in_csv=[]
casefold_csv_not_in_extracted=[]
casefold_extracted_not_in_csv=[]
```

Thus every extracted image has a CSV entry and every CSV-referenced image exists after extraction. There were no unmatched files in either direction.

**Source:** [`datasets/messidor2/extracted/IMAGES/`](../datasets/messidor2/extracted/IMAGES/), [`datasets/messidor2/messidor-2.csv`](../datasets/messidor2/messidor-2.csv), and [`reports/inspect_messidor2.py`](inspect_messidor2.py).

## Possible patient-level leakage

The CSV has 874 explicit rows with columns named `left` and `right`; each row therefore visibly represents a left/right filename pair. Both filename columns are unique, and the extracted set has 1,748 unique filenames. In 524 of the 874 rows, the two filenames share their first underscore-delimited token. The filenames and CSV contain no explicit patient identifier, and the shared token is not verified here as a patient ID. Therefore the data shows paired left/right entries, but it does not establish patient identity or prove that multiple rows belong to the same patient.

```text
csv_rows_with_left_right_same_first_filename_token=524
csv_rows_with_left_right_pairs=874
```

**Source:** [`datasets/messidor2/messidor-2.csv`](../datasets/messidor2/messidor-2.csv) and [`reports/inspect_messidor2.py`](inspect_messidor2.py).

## Comparison with the APTOS label scale

APTOS uses numeric classes `0` through `4` in [`datasets/aptos/train.csv`](../datasets/aptos/train.csv). The supplied APTOS convention is `0 No DR`, `1 Mild`, `2 Moderate`, `3 Severe`, and `4 Proliferative`. Messidor-2’s inspected CSV contains no grade column and no numeric label values, so no Messidor-2 grading scale can be compared or mapped from the inspected files.

Possible next decisions, without remapping or merging classes now:

1. Obtain and verify the official Messidor-2 grade labels, then preserve their native values and document any defensible mapping separately.
2. Treat Messidor-2 as an unlabeled external image set for quality, preprocessing, or distribution checks only.
3. Do not use Messidor-2 for supervised grading evaluation until a label file and its scale are verified.

No option has been selected in this report.

**Source:** [`datasets/aptos/train.csv`](../datasets/aptos/train.csv), [`datasets/messidor2/messidor-2.csv`](../datasets/messidor2/messidor-2.csv), and the CSV inspection output above.

## Status

Messidor-2 extraction and file-integrity inspection are complete. Image opening, dimensions, and bidirectional CSV matching are complete. The grading-label inspection is not complete because the available CSV is a left/right filename-pair table rather than a verified DR-label table. Patient-level linkage also remains unverified because no patient identifier is present.
