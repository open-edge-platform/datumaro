# Contributing to Datumaro

## Welcome! 🌟

We appreciate any contribution to [Datumaro](https://github.com/open-edge-platform/datumaro),
whether it's in the form of a Pull Request, Feature Request or general comments/issue that you found.
For feature requests and issues, please feel free to create a GitHub Issue in this repository.

## Table of Contents

- [Security](#security)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Sign your work](#sign-your-work)
- [License](#license)

## Security

To ensure our codebase remains secure, we leverage GitHub Actions for continuous security scanning (on pre-commit, PR and periodically) with the following tools:

- [CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql): static analysis tool to check Python, Rust code and GitHub Actions workflows
- [Semgrep](https://github.com/semgrep/semgrep): static analysis tool to check Python code; ML-specific Semgrep rules developed by [Trail of Bits](https://github.com/trailofbits/semgrep-rules?tab=readme-ov-file#python) are used
- [Bandit](https://github.com/PyCQA/bandit): Static analysis tool to check Python code
- [Zizmor](https://github.com/woodruffw/zizmor): Static analysis tool to check GitHub Actions workflows
- [Trivy](https://github.com/aquasecurity/trivy): Check misconfigurations and detect security issues in dependencies

| Tool    | Pre-commit | PR-checks | Periodic |
| ------- | ---------- | --------- | -------- |
| CodeQL  |            | ✅        | ✅       |
| Semgrep |            |           | ✅       |
| Bandit  | ✅         | ✅        | ✅       |
| Zizmor  | ✅         | ✅        | ✅       |
| Trivy   |            |           | ✅       |

<details>
<summary>Suppressing False Positives</summary>

If necessary, to suppress _false_ positives, add inline comment with specific syntax.
Please also add a comment explaining _why_ you decided to disable a rule or provide a risk-acceptance reason.

#### Bandit

Findings can be ignored inline with `# nosec BXXX` comments.

```python
import subprocess # nosec B404 # this is actually fine
```

[Details](https://bandit.readthedocs.io/en/latest/config.html#exclusions) in Bandit docs.

#### Zizmor

Findings can be ignored inline with `# zizmor: ignore[rulename]` comments.

```yaml
uses: actions/checkout@v3 # zizmor: ignore[artipacked] this is actually fine
```

[Details](https://woodruffw.github.io/zizmor/usage/#with-comments) in Zizmor docs.

#### Semgrep

Findings can be ignored inline with `# nosemgrep: rule-id` comments.

```python
    # nosemgrep: python.lang.security.audit.dangerous-system-call.dangerous-system-call # this is actually fine
    r = os.system(' '.join(command))
```

[Details](https://semgrep.dev/docs/ignoring-files-folders-code) in Semgrep docs.

</details>

Read additional details in the [`Security Policy`](security.md)

## How to Contribute

### Contribute Code Changes

If you'd like to help improve Datumaro, pick one of the issues listed in [GitHub
Issues](https://github.com/open-edge-platform/datumaro/issues) and submit
a [Pull Request](https://github.com/open-edge-platform/datumaro/pulls) to address it.
Note: Before you start working on it, please make sure the change hasn’t already been implemented.

### Report Bugs

If you encounter a bug, please open an issue in [`Github Issues`](https://github.com/open-edge-platform/datumaro/issues).
Be sure to include all the information requested in the bug report template to help us understand and resolve the issue
quickly.

### Suggest Enhancements

Intel welcomes suggestions for new features and improvements. Follow these steps to make a suggestion:

- Check if there's already a similar suggestion in [`Github Issues`](https://github.com/open-edge-platform/datumaro/issues).
- If not, please open a new issue and provide the information requested in the feature request template.

### Submit Pull Requests

Before submitting a pull request, ensure you follow these guidelines:

- Fork the repository and create your branch from `develop`.
- Follow the [`Development Guidelines`](#development-guidelines) in this document.
- Test your changes thoroughly.
- Document your changes (in code, readme, etc.).
- Submit your pull request, detailing the changes and linking to any relevant issues.
- Wait for a review. Intel will review your pull request as soon as possible and provide you with feedback.
  You can expect a merge once your changes are validated with automatic tests and approved by maintainers.

## Development Guidelines

### Prerequisites

- Python (3.9+)

To set up your development environment, please follow the steps below.

0. Because Datumaro has some C++ and Rust implementations to improve Python performance, you should install C++ compiler (`apt-get install build-essential`) and a [Rust toolchain](https://www.rust-lang.org/tools/install) in your system to build the binary extensions.

1. Fork the [repo](https://github.com/open-edge-platform/datumaro).

2. Clone the forked repo.
   ```bash
   git clone <forked_repo>
   ```
3. Optionally, install a virtual environment (recommended):

   ```bash
   python -m pip install virtualenv
   python -m virtualenv venv
   . venv/bin/activate
   ```

4. Install Datumaro with the following optional dependencies:

   ```bash
   cd /path/to/the/cloned/repo/
   pip install -e .[tf,tfds,torch]
   ```

5. Install development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

6. Set up pre-commit hooks in the repo. See [Code style](#code-style).

   ```bash
   pre-commit install
   pre-commit run
   ```

7. Create your branch based off the `develop` branch and make changes.

8. Verify your code by running unit tests and integration tests. See [Testing](#testing)

   ```bash
   pytest -v
   ```

   or

   ```bash
   python -m pytest -v
   ```

9. Push your changes.

Now you are ready to create a PR(Pull Request) and get review.

### Code style

Try to be readable and consistent with the existing codebase.

The project uses Black for code formatting and isort for sorting import statements.
You can find corresponding configurations in `pyproject.toml` in the repository root.
No trailing whitespaces, at most 100 characters per line.

Datumaro includes a Git pre-commit hook, `.pre-commit-config.yaml` that can help you follow the style requirements. To install, make sure isort and black are installed on your system, then run `pre-commit run`.

### Testing

It is expected that all Datumaro functionality is covered and checked by
unit tests. Tests are placed in the `tests/unit/` directory. Additional
pre-generated files for tests can be stored in the `tests/assets/` directory.
CLI tests are separated from the core tests, they are stored in the
`tests/integration/cli/` directory.

Currently, we use [`pytest`](https://docs.pytest.org/) for testing.

To run tests use:

```bash
pytest -v
```

or

```bash
python -m pytest -v
```

## Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise
have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

```
Signed-off-by: Joe Smith <joe.smith@email.com>
```

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

## License

Datumaro is licensed under the terms in [LICENSE](LICENSE). By contributing to the project, you agree
to the license and copyright terms therein and release your contribution under these terms.
