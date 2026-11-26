"""Write docs/commit_log.md: commit hashes + ISO timestamps (for the report)."""
import subprocess
from pathlib import Path

out = subprocess.run(
    ["git", "log", "--reverse", "--date=iso-strict",
     "--pretty=format:| `%h` | %ad | %s |"],
    capture_output=True, text=True, check=True).stdout
text = ("# Commit log\n\n| Hash | Timestamp | Message |\n|---|---|---|\n" + out + "\n\n"
        "Note: a commit cannot contain its own hash, so when this file ships inside the last commit "
        "it lists only the earlier ones; run `git log --date=iso-strict` for the final hash and timestamp.\n")
Path("docs").mkdir(exist_ok=True)
Path("docs/commit_log.md").write_text(text, encoding="utf-8")
print(text)
