# VibeChannel Schema

This file defines the format for VibeChannel conversations.

## Folder Structure

```yaml
root: /
channels: subfolders (e.g., general/, random/, dev/)
messages: markdown files inside channel folders
```

## Filename Convention

```yaml
pattern: "{timestamp}-{sender}-{id}.md"
timestamp:
  format: "%Y%m%dT%H%M%S"
  example: "20250115T103045"
sender:
  format: "lowercase alphanumeric, no spaces"
  example: "lucas"
id:
  length: 6
  charset: "a-z0-9"
  example: "a3f8x2"
```

## Message Format

```yaml
from: string        # Sender identifier
date: datetime      # ISO 8601 format
reply_to: string    # Optional: filename of parent message
tags: [array]       # Optional: categorization tags
```

## Rendering Preferences

```yaml
rendering:
  sort_by: date
  order: ascending
  group_by: date
  timestamp_display: relative
```
