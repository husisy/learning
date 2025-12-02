# Agent Instructions for VibeChannel

This branch contains conversations following the VibeChannel protocol.

**IMPORTANT:** Read `schema.md` for the complete format specification.

## Quick Start

1. Each channel is a subfolder (e.g., general/, random/)
2. Each message is a `.md` file: `{timestamp}-{sender}-{id}.md`
3. Use YAML frontmatter + markdown body

## Example Message

```markdown
---
from: lucas
date: 2025-01-15T10:30:45Z
---

Your message content here.
```
