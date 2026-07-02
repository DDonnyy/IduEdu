import pathlib
import re
import tomllib

VERSION_PATTERN = re.compile(r'VERSION = "([^"]+)"')


def main() -> None:
    pyproject_version = tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
    version_file = pathlib.Path("iduedu/_version.py")
    match = VERSION_PATTERN.search(version_file.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"VERSION line not found in {version_file}")

    version_file_value = match.group(1)
    if pyproject_version != version_file_value:
        raise SystemExit(f"pyproject.toml {pyproject_version} != {version_file} {version_file_value}")

    print(f"Versions match: {pyproject_version}")


if __name__ == "__main__":
    main()
