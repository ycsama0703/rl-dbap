import json
from pathlib import Path

root = Path('artifacts/distill_data')
processed = root / 'processed'
processed.mkdir(parents=True, exist_ok=True)

RAW_TRAIN = root / 'raw' / 'teacher_outputs_banks.jsonl'
RAW_TEST = root.parent / 'test' / 'test_banks.jsonl'

def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open('r', encoding='utf-8') if line.strip()]

def sanitize(text: str) -> str:
    return ' '.join((text or '').strip().split())

def extract_teacher(row):
    system = sanitize(row.get('system', ''))
    user = sanitize(row.get('prompt', ''))
    assistant = sanitize(row.get('raw_output') or row.get('answer', ''))
    if not assistant:
        msgs = row.get('messages', [])
        assistants = [m for m in msgs if m.get('role') == 'assistant']
        assistant = sanitize(assistants[0].get('content', '') if assistants else '')
    return system, user, assistant

def extract_test(row):
    msgs = row.get('messages', [])
    system = sanitize(next((m.get('content', '') for m in msgs if m.get('role') == 'system'), ''))
    user = sanitize(next((m.get('content', '') for m in msgs if m.get('role') == 'user'), ''))
    assistants = [m for m in msgs if m.get('role') == 'assistant']
    assistant = sanitize(next((m.get('content', '') for m in assistants if m.get('loss')), assistants[0].get('content', '') if assistants else ''))
    return system, user, assistant

def write_split(records, extractor, src_path: Path, tgt_path: Path):
    src_lines, tgt_lines = [], []
    for row in records:
        system, user, assistant = extractor(row)
        if not assistant:
            continue
        src_lines.append(' '.join(filter(None, [system, user])))
        tgt_lines.append(assistant)
    src_path.write_text('\n'.join(src_lines) + '\n', encoding='utf-8')
    tgt_path.write_text('\n'.join(tgt_lines) + '\n', encoding='utf-8')
    print(f"[ok] {src_path.name}: {len(src_lines)} samples")

def main():
    train_records = load_jsonl(RAW_TRAIN)
    test_records = load_jsonl(RAW_TEST)

    write_split(train_records, extract_teacher,
                processed / 'train.source', processed / 'train.target')

    if not (processed / 'val.source').exists():
        (processed / 'val.source').write_bytes((processed / 'train.source').read_bytes())
        (processed / 'val.target').write_bytes((processed / 'train.target').read_bytes())
        print('[ok] val: copied from train')

    write_split(test_records, extract_test,
                processed / 'test.source', processed / 'test.target')

if __name__ == '__main__':
    main()
