# Supported Encodings

| Encoder          | Key        | Description                                                                 |
|------------------|------------|-----------------------------------------------------------------------------|
| Integer          | `integer`  | Assigns each unique element an integer in ascending order.                  |
| One Hot          | `one_hot`  | Assigns each unique element an indicator column.                            |
| Hashing Encoding | `hash`     | Hashes categorical variables into `n` (defaulted to 8) columns. Recommended when dealing with high cardinality in one_hot context. |
