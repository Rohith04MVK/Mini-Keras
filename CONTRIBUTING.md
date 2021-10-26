# Contributing Guide

This project is fully open-sourced and will be automatically deployed whenever commits are pushed to `main` branch, so these are the guidelines to keep everything clean and in working order.

Note that contributions may be rejected on the basis of a contributor failing to follow these guidelines

## Rules

1. **No force-pushes** or modifying the Git history in any way.
2. If you have direct access to the repository **Create a branch for your changes** and create a pull request for that branch. If not, create a branch on a for of the repository and create a pull request from there.
   - It's common practice for repository to reject direct pushes to `main`, so make branching a habit!
   - If PRing from your own fork, **ensure that "Allow edits from maintainers" is checked**. This gives permission for maintainers to commit changes directly to your fork, speeding up the review process.
3. **Adhere to the prevailing code style** which we enforce using [`flake8`](https://flake8.pycqa.org/en/latest/index.html) and [`pre-commit`](https://pre-commit.com/).
   - Run `flake8` and `pre-commit` against your code before you push it.
   - [Git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) are a powerful git feature for executing custom scripts when certain important git actions occur. The pre-commit hook is the first hook executed during the commit process and can be used to check the code being commited & abort the commit if issues, such as linting failures are detected. While git hooks can seem daunting to configure, the `pre-commit` framework abstracts this process away from you and is provided as a dev dependency for this project. Run `pipenv run precommit` when setting up the project and you'll never have to worry about commiting your code that fails linting.
4. **Make great commits**. A well structured git log is key to a project's maintainability; it efficiently provides insight into when and why things were done for future maintainers of the project.
   - Commits should be as narrow in scope as possible. Commits that span hundreds of lines across multiple unrelated functions and/or files are very hard for maintainers to follow. After about a week they'll probably be hard for you to follow too.
   - Avoid making minor commits for fixing typos or linting errors. Since you've already set up a `pre-commit` hook to run the linting pipeline before a commit, you shouldn't be commiting linting issues anyway.
   - A more in-depth guide to writing great commit messages can be found in Chris Beam's [_How to Write a Git Commit Message_](https://chris.beams.io/posts/git-commit/)
5. **Avoid frequent pushes to the main repository**. This goes for PRs opened against your fork as well. Try to batch your commits until you've finished working for that session, or you've reached a point where collaborators need your commits to continue their own work. This also provides you the opportunity to ammend commits for minor changes rather than having to commit them on their own because you've already pushed.
   - This includes merging main into your branch. Try to leave merging from main for after your PR passes review: a maintainer will bring your PR up to date before merging. Exceptions to this include: resolving merge conflicts, needing something that was pushed to main for your branch, or something was pushed to main that could potentially affect the functionality of what you're doing
6. **Don't fight the framework**. Every framework has its laws, but the frameworks we've picked out have been carefully chosen for their particular merits. If you can avoid it, please resist reimplementing swathes or framework logic - the work has already been done for you!
7. If someone is working on an issue or pull request. **do not open your own pull request for the same task**. Instead, collaborate with the author(s) of the existing pull request. Duplicate PRs opened without communicating with the other author(s) and/or repository authors will be closed. Communication is key, and there's no point in two separate implementations of the same thing.
   - One option is to fork the other contributor's repository and submit your changes to their branch with your own pull request. We suggest following these guidelines when interacting with their repository as well.
   - The author(s) of inactive PRs and claimed issues will be pinged after a week of inactivity for an update. Continued inactivity may result in the issue being released back to the community and/or PR closure.
8. **Work as a team** and collaborate whenever possible. Keep things friendly and help each other out - these are shared projects and nobody likes to have their feet trodded on.
9. All static content, such as images or audio, **must be licensed for open public use**.
   - Static content must be hosted by a service designed to do so. Failing to do so is known as "leeching" and is frowned upon, as it generates extra bandwidth to the host without providing benefit. It would be best if appropriately licensed content is added to the repository itself.

Above all, the needs of our community should come before the wants of an individual. Work together, build solutions to problems and try to do so in a way that people can learn from easily. Abuse of our trust may result in the loss of your Contributor role.

## Changes to this Arrangement

All projects evolve over time, and this contribution guide is no different. This document is open to pull requests or changes by contributors. If you believe you have something valuable to add or change, please don't hesitate to do so in a PR.

## Type Hinting

[PEP 484](https://www.python.org/dev/peps/pep-0484/) formally specifies type hints for Python functions, added to the Python Standard Library in version 3.5. Type hints are recognized by most modern code editing tools and provide useful insight into both the input and output types of a function, preventing the user from having to go through the codebase to determine these types.

For example:

```py
import typing as t


def foo(input_1: int, input_2: t.Dict[str, str]) -> bool:
    ...
```

Tell us that `foo` accepts an `int` and a `dict` with `str` keys and values, and returns a `bool`.

All functions declarations should be type hinted in code contributed to this repository

## AutoDoc Formatting Directive

Many documentation packages provide support for automatic documentation generation from the codebase's docstrings. These tools utilize special formatting directives to enable richer formatting in the generated documentation.

For example:

```py
import typing as t


def foo(bar: int, baz: t.Optional[t.Dict[str, str]] = None) -> bool:
    """
    Does some things with some stuff.

    :param bar: Some input
    :param baz: Optional, some dictionary with string keys and values

    :return: Some boolean
    """
    ...
```

Since we don't utilize automatic documentation generation, use of this syntax should not be used in the code contributed here. Should the purpose and type of the input variables not be easily discernable from the variable name and type annotation a prose explanation can be used. Explicit references to variables, function, classes, etc. should be wrapped with backticks (`` ` ``)

For example, the above docstring would become:

```py
import typing as t


def foo(bar: int, baz: t.Optional[t.Dict[str, str]] = None) -> bool:
    """
    Does some things with some stuff.

    This function takes an index, `bar` and checks for its presence in the database `baz`, passed as a dictionary. Returns `False` if `baz` is not passed.
    """
    ...
```

### Work in Progress (WIP) PRs

Github [provides a PR feature](https://github.blog/2019-02-14-introducing-draft-pull-requests/) that allows the PR author to mark it as a WIP. This provides both a visual and functional indicator that the contents of the PR are in a draft state and not yet ready for formal review.

This feature should be utilized in place of the traditional method of prepending `[WIP]` to the PR title.

As stated earlier, **ensure that "Allow edits from maintainers" is checked**. This gives permission for maintainers to commit changes directly to your fork, speeding up the review process.
