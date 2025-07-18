# Checkout

This command allows to restore a specific project revision in the project
tree or to restore separate revisions of sources. A revision can be a commit
hash, branch, tag, or any [relative reference in the Git format](https://git-scm.com/book/en/v2/Git-Tools-Revision-Selection).

This command has multiple forms:
``` bash
1) datum checkout <revision>
2) datum checkout [--] <source1> ...
3) datum checkout <revision> [--] <source1> <source2> ...
```

1 - Restores a revision and all the corresponding sources in the
working directory. If there are conflicts between modified files in the
working directory and the target revision, an error is raised, unless
`--force` is used.

2, 3 - Restores only selected sources from the specified revision.
The current revision is used, when not set.

"--" can be used to separate source names and revisions:
- `datum project checkout name` - will look for revision "name"
- `datum project checkout -- name` - will look for source "name" in the current
  revision

Usage:
``` bash
datum project checkout [-h] [-f] [-p PROJECT_DIR] [rev] [--] [sources [sources ...]]
```

Parameters:
- `--force` - Allows to overwrite unsaved changes in case of conflicts
- `-p, --project` (string) - Directory of the project to operate on
  (default: current directory).
- `-h, --help` - Print the help message and exit.

Examples:
- Restore the previous revision:
  `datum project checkout HEAD~1`
  <br>

- Restore the saved version of a source in the working tree
  `datum project checkout -- source-1`
  <br>

- Restore a previous version of a source
  `datum project checkout 33fbfbe my-source`
  <br>
