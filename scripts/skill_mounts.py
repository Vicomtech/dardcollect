#!/usr/bin/env python3
"""Agent-host skill mounts: one gate for every host the harness claims.

WHY THIS EXISTS
    An agent host discovers skills only from the directories it scans; a harness
    that claims several hosts but wires one mount has a portability claim its
    code does not honour. Adopted from the ai-harness-eng harness (2026-10-06):
    the harness claimed Kilo, pi, Claude Code, Codex and GitHub Copilot support
    while only one host actually loaded a skill, so pi, Claude Code and Codex
    discovered nothing and no gate could see it. This module owns both halves of
    the fix: the registry shape (canonical tree + one mount per host + the prose
    claim that must match) and the file-by-file mirror comparison.

    `scripts/host_surfaces.json` is the registry. The canonical tree is
    `.agents/skills/`, the cross-client Agent Skills directory read directly by
    Kilo Code, pi, Codex and GitHub Copilot, so those four cannot drift; only
    Claude Code needs a mirror (`.claude/skills/`), because it does not read
    `.agents/skills/`. A mount equal to the canonical path is the source itself
    and needs no copy; every distinct mount is compared file by file.

USAGE
    python scripts/skill_mounts.py [--registry host_surfaces.json] [--root .]
    python scripts/skill_mounts.py --sync | --check   # write mounts | errors only

Exit-code contract: 0 clean, 1 errors, 2 misconfigured registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REGISTRY_FILE = "host_surfaces.json"


def _norm_bytes(path: Path) -> bytes:
    """File bytes with CRLF folded to LF: a checkout artifact is not drift."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def _dir_hashes(root: Path) -> dict[str, str]:
    """Map POSIX relative path -> SHA256 of newline-normalised bytes."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            out[rel] = hashlib.sha256(_norm_bytes(path)).hexdigest()
    return out


def _err(file_label: str, kind: str, message: str, fix: str) -> str:
    """Remediation-injecting error string, in the repository's message shape."""
    return f"[{kind}] {file_label}: {message} -> {fix}"


def load_registry(registry_path: Path) -> tuple[dict | None, str | None]:
    """Return (data, error). A present-but-unreadable registry is an error."""
    if not registry_path.is_file():
        return None, f"{registry_path} is absent"
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return None, f"{registry_path} cannot be parsed: {exc}"
    if not isinstance(data, dict):
        return None, f"{registry_path} is not a JSON object"
    return data, None


def _host_mounts(data: dict) -> dict[str, list[str]]:
    """Host -> mount list from the registry's hostMounts map."""
    source = data.get("hostMounts")
    if not isinstance(source, dict):
        return {}
    return {str(host): [str(m) for m in list(mounts or [])] for host, mounts in source.items()}


def _mount_path_errors(host: str, mount: str) -> str | None:
    """A mount must be project-relative and stay inside the project."""
    p = Path(mount)
    if p.is_absolute() or mount.startswith("/") or mount.startswith("\\"):
        return _err(
            mount,
            "skill_mount_invalid",
            f"host '{host}' declares the absolute mount '{mount}'",
            "use a project-relative mount directory; an absolute path cannot be "
            "ported to another machine or project",
        )
    if ".." in p.parts:
        return _err(
            mount,
            "skill_mount_invalid",
            f"host '{host}' declares the escaping mount '{mount}'",
            "use a project-relative mount directory that stays inside the project",
        )
    return None


def structural_errors(data: dict, root: Path) -> list[str]:
    """Registry shape: canonical name, per-host wiring, valid mount paths."""
    errors: list[str] = []
    canonical = str(data.get("canonical") or "")
    if not canonical:
        errors.append(
            _err(
                str(root),
                "skill_registry_canonical",
                "the registry declares no 'canonical' tree",
                'add "canonical": ".agents/skills" so the single edited tree is named',
            )
        )
    hosts = _host_mounts(data)
    if not hosts:
        errors.append(
            _err(
                str(root),
                "empty_domain",
                "the registry declares no host mounts, so no host was verified",
                "add hostMounts (host -> mount directories); an empty registry is not a pass",
            )
        )
        return errors
    for host, mounts in sorted(hosts.items()):
        if not mounts:
            errors.append(
                _err(
                    host,
                    "skill_host_unmounted",
                    f"host '{host}' is claimed but declares no mount, so it discovers no skill",
                    f"add the directory '{host}' really scans, or remove the host "
                    f"from the registry and from the instructions that claim it",
                )
            )
        for mount in mounts:
            bad = _mount_path_errors(host, mount)
            if bad:
                errors.append(bad)
    return errors


def _split_hosts(sentence: str) -> list[str]:
    """Host names out of a prose list: 'A, B and C' -> [A, B, C]."""
    text = sentence.replace(" and ", ", ")
    return [part.strip() for part in text.split(",") if part.strip()]


def _claim_sentence_hosts(claim: dict, root: Path) -> tuple[set[str] | None, str | None]:
    """Host names in the claim sentence, or (None, reason) when unreadable."""
    doc = root / str(claim["doc"])
    if not doc.is_file():
        return None, f"the host-claim document '{claim['doc']}' is absent"
    text = doc.read_text(encoding="utf-8", errors="replace")
    start = str(claim.get("start") or "")
    if not start or start not in text:
        return None, f"the host-claim sentence start is not in '{claim['doc']}'"
    chunk = text.split(start, 1)[1]
    end = str(claim.get("end") or "")
    if end and end in chunk:
        chunk = chunk.split(end, 1)[0]
    return set(_split_hosts(chunk)), None


def claim_errors(data: dict, root: Path) -> list[str]:
    """The host names in the claim document must equal the registry's hosts."""
    claim = data.get("hostClaim")
    if not isinstance(claim, dict) or not claim.get("doc"):
        return []
    listed, reason = _claim_sentence_hosts(claim, root)
    if reason:
        return [
            _err(
                str(claim["doc"]),
                "skill_host_claim_mismatch",
                f"{reason}, so the prose claim cannot be compared with the wiring",
                "restore or fix the document, or drop hostClaim from the registry",
            )
        ]
    declared = set(_host_mounts(data))
    if listed == declared:
        return []
    return [
        _err(
            str(claim["doc"]),
            "skill_host_claim_mismatch",
            f"the host list in '{claim['doc']}' is {sorted(listed or [])} but the "
            f"registry wires {sorted(declared)}",
            "make them equal: every claimed host needs a mount in the registry, "
            "and every registry host must be named in the claim sentence",
        )
    ]


def _skill_dirs(tree: Path) -> list[Path]:
    """Immediate subdirectories that are skill directories."""
    if not tree.is_dir():
        return []
    return sorted(d for d in tree.iterdir() if d.is_dir())


def _name_diff(canon_names: list[str], mount_names: list[str], display: str) -> list[str]:
    errors: list[str] = []
    for name in sorted(set(canon_names) - set(mount_names)):
        errors.append(
            _err(
                f"canonical/{name}",
                "skill_mirror_missing",
                f"canonical skill '{name}' has no mirror in {display}/, so the "
                f"hosts wired there never load it",
                f"copy it from the canonical tree into {display}/{name}/ verbatim, or run --sync",
            )
        )
    for name in sorted(set(mount_names) - set(canon_names)):
        errors.append(
            _err(
                f"{display}/{name}",
                "skill_mirror_extra",
                f"mirror skill '{name}' in '{display}' has no canonical twin, so "
                f"its source is unknown",
                "move it under the canonical tree first, or remove it",
            )
        )
    return errors


def _file_diff(canon_skill: Path, mount_skill: Path, name: str, display: str) -> list[str]:
    """Every file of a skill pair must match by path and by content."""
    errors: list[str] = []
    canon_files = _dir_hashes(canon_skill)
    mount_files = _dir_hashes(mount_skill)
    for rel in sorted(set(canon_files) - set(mount_files)):
        errors.append(
            _err(
                f"{display}/{name}/{rel}",
                "skill_mirror_missing",
                f"'{rel}' is missing from skill '{name}' in '{display}', so that "
                f"host loads an incomplete skill",
                f"copy {name}/{rel} from the canonical tree to "
                f"{display}/{name}/{rel} verbatim, or run --sync",
            )
        )
    for rel in sorted(set(mount_files) - set(canon_files)):
        errors.append(
            _err(
                f"{display}/{name}/{rel}",
                "skill_mirror_extra",
                f"'{rel}' exists only in the '{display}' copy of skill '{name}'",
                "move it under the canonical tree first, or remove it",
            )
        )
    for rel in sorted(set(canon_files) & set(mount_files)):
        if canon_files[rel] != mount_files[rel]:
            errors.append(
                _err(
                    f"{display}/{name}/{rel}",
                    "skill_mirror_drift",
                    f"'{rel}' of skill '{name}' drifted between the canonical "
                    f"tree and '{display}', so the host wired there loads stale "
                    f"content",
                    f"copy {name}/{rel} from the canonical tree over "
                    f"{display}/{name}/{rel} verbatim, or run --sync",
                )
            )
    return errors


def _pair_errors(canonical: Path, mount_dir: Path, display: str) -> list[str]:
    if not mount_dir.is_dir():
        return [
            _err(
                display,
                "empty_domain",
                f"the mount point '{display}' is absent",
                f"recreate '{display}' as a mirror of the canonical tree (git "
                f"checkout, or run --sync); the hosts wired there discover skills "
                f"only inside it",
            )
        ]
    canon_names = [d.name for d in _skill_dirs(canonical)]
    mount_names = [d.name for d in _skill_dirs(mount_dir)]
    errors = _name_diff(canon_names, mount_names, display)
    for name in sorted(set(canon_names) & set(mount_names)):
        errors.extend(_file_diff(canonical / name, mount_dir / name, name, display))
    return errors


def mount_errors(registry_path: Path, root: Path) -> list[str]:
    """Full gate: registry shape, host claim, and every declared mount."""
    data, error = load_registry(Path(registry_path))
    if error:
        return [
            _err(
                str(registry_path),
                "empty_domain",
                f"the skill-mount registry is unusable: {error}",
                "restore or repair the registry; without it no mount is verified "
                "and the claimed hosts load unchecked",
            )
        ]
    errors = structural_errors(data or {}, Path(root)) + claim_errors(data or {}, Path(root))
    canonical = Path(root) / str((data or {}).get("canonical") or "")
    if not canonical.is_dir():
        errors.append(
            _err(
                str(canonical),
                "empty_domain",
                "the canonical skill tree is absent",
                "restore the canonical tree (git checkout); without it no mirror can be verified",
            )
        )
        return errors
    if not _skill_dirs(canonical):
        errors.append(
            _err(
                str(canonical),
                "empty_domain",
                "the canonical skill tree holds no skill directories",
                "restore the skills (git checkout); an empty tree means no "
                "frontmatter or mirror was inspected",
            )
        )
        return errors
    seen: set[str] = set()
    for _host, mounts in sorted(_host_mounts(data or {}).items()):
        for mount in mounts:
            target = Path(root) / mount
            key = str(target.resolve()) if target.exists() else str(target)
            if key in seen:
                continue
            seen.add(key)
            errors.extend(_pair_errors(canonical, target, mount))
    return errors


def frontmatter_tree_errors(tree: Path, checker, rel_fn) -> list[str]:
    """Run a frontmatter checker over the canonical tree, one entry per skill.

    `checker` is called as `checker(skill_md_path, label)` and returns error
    strings; the YAML contract itself lives in `skill_frontmatter.py` so both
    gates report the same way and the validator stays thin.
    """
    if not tree.is_dir():
        return [
            _err(
                "skills",
                "empty_domain",
                "the canonical skill tree is absent, so no frontmatter was inspected",
                "restore the canonical tree (git checkout); an uninspected tree is not a pass",
            )
        ]
    skills = _skill_dirs(tree)
    if not skills:
        return [
            _err(
                "skills",
                "empty_domain",
                "the canonical skill tree holds no skill directories",
                "restore the skills (git checkout)",
            )
        ]
    errors: list[str] = []
    for skill in skills:
        skill_md = skill / "SKILL.md"
        if not skill_md.exists():
            errors.append(
                _err(
                    str(rel_fn(skill)),
                    "skill_md_missing",
                    f"skill directory '{skill.name}' has no SKILL.md, so it is not "
                    f"a skill the spec defines and it is silently absent from every "
                    f"client that packages skills",
                    "add SKILL.md with `name` and `description` frontmatter, or "
                    "remove the directory",
                )
            )
            continue
        errors.extend(checker(skill_md, str(rel_fn(skill_md))))
    return errors


def sync_mounts(registry_path: Path, root: Path) -> list[str]:
    """Regenerate every declared mount from the canonical tree.

    The gate stays the authority: this only performs the copy the gate would
    otherwise fail on, so a human never hand-copies mounts.
    """
    data, error = load_registry(Path(registry_path))
    if error:
        raise SystemExit(f"cannot sync: {error}")
    root = Path(root)
    canonical = root / str((data or {}).get("canonical") or "")
    if not canonical.is_dir():
        raise SystemExit(f"cannot sync: canonical tree {canonical} is absent")
    canon_res = canonical.resolve()
    written: list[str] = []
    for _host, mounts in sorted(_host_mounts(data or {}).items()):
        for mount in mounts:
            target = root / mount
            target_res = target.resolve()
            if target_res == canon_res:
                continue  # the mount IS the canonical tree: nothing to copy
            if target_res in canon_res.parents:
                raise SystemExit(
                    f"cannot sync: mount '{mount}' is an ancestor of the canonical tree"
                )
            if target.is_dir() and _dir_hashes(target) == _dir_hashes(canonical):
                continue  # already in sync: a no-op rewrite is not drift
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(canonical, target)
            written.append(mount)
    return sorted(set(written))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent-host skill mount gate")
    parser.add_argument("--registry", default=REGISTRY_FILE)
    parser.add_argument("--root", default=".")
    parser.add_argument("--sync", action="store_true", help="regenerate every mount")
    parser.add_argument("--check", action="store_true", help="quiet: errors only")
    args = parser.parse_args(argv)

    registry = Path(args.root) / args.registry
    if not registry.is_file():
        alt = Path(args.root) / "scripts" / args.registry
        registry = alt if alt.is_file() else registry
    if args.sync:
        for mount in sync_mounts(registry, Path(args.root)):
            print(f"synced {mount}")
        return 0
    errors = mount_errors(registry, Path(args.root))
    if errors:
        for err in errors:
            print(f"  ERROR {err}", file=sys.stderr)
        return 1
    if not args.check:
        print(f"OK: every declared mount mirrors the canonical tree ({registry}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
