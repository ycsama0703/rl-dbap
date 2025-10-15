from __future__ import annotations
import argparse
from pathlib import Path


def copy_head(in_path: Path, out_path: Path, limit: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            n += 1
            if n >= limit:
                break
    return n


def main():
    ap = argparse.ArgumentParser(description='Create tiny JSONL samples (first N lines) for safe Git pushes')
    ap.add_argument('--in', dest='inp', type=str, required=True,
                    help='Input file or directory containing .jsonl files')
    ap.add_argument('--out-dir', type=str, default='artifacts/samples',
                    help='Output directory to write sampled jsonl files')
    ap.add_argument('--limit', type=int, default=50, help='Number of lines per file')
    ap.add_argument('--glob', type=str, default='*.jsonl', help='Glob pattern when input is a directory')
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)

    files: list[Path]
    if inp.is_file():
        files = [inp]
    else:
        files = sorted(inp.glob(args.glob))

    total_files = 0
    total_lines = 0
    for fp in files:
        rel_name = fp.name
        outp = out_dir / rel_name
        n = copy_head(fp, outp, args.limit)
        total_files += 1
        total_lines += n
        print(f'[sample] {fp} -> {outp} ({n} lines)')

    print(f'[sample] done: files={total_files}, lines={total_lines}, out_dir={out_dir}')


if __name__ == '__main__':
    main()

