import json
from pathlib import Path

raw_train = Path('artifacts/distill_data/raw/teacher_outputs_banks.jsonl')
raw_test = Path('artifacts/test/test_banks.jsonl')
out_dir = Path('artifacts/gkd_data')
out_dir.mkdir(parents=True, exist_ok=True)

def build_train_eval():
    records = []
    with raw_train.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            prompt = (rec.get('prompt') or '').strip()
            answer = (rec.get('raw_output') or rec.get('answer') or '').strip()
            if not prompt or not answer:
                continue
            records.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': answer},
                ]
            })
    # split into train/eval (last 200 for eval)
    eval_size = min(200, len(records))
    train_split = records[:-eval_size] if eval_size else records
    eval_split = records[-eval_size:] if eval_size else []

    with (out_dir / 'train.jsonl').open('w', encoding='utf-8') as f:
        for rec in train_split:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    with (out_dir / 'eval.jsonl').open('w', encoding='utf-8') as f:
        for rec in eval_split:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(train_split), len(eval_split)

def build_test():
    count = 0
    with (out_dir / 'test.jsonl').open('w', encoding='utf-8') as f_out:
        with raw_test.open('r', encoding='utf-8') as f_in:
            for line in f_in:
                rec = json.loads(line)
                msgs = rec.get('messages', [])
                user_msg = next((m.get('content', '') for m in msgs if m.get('role') == 'user'), '').strip()
                assistants = [m for m in msgs if m.get('role') == 'assistant']
                assistant_msg = ''
                for m in assistants:
                    if m.get('loss') is True:
                        assistant_msg = (m.get('content') or '').strip()
                        break
                if not assistant_msg and assistants:
                    assistant_msg = (assistants[0].get('content') or '').strip()
                if not user_msg or not assistant_msg:
                    continue
                rec_out = {
                    'messages': [
                        {'role': 'user', 'content': user_msg},
                        {'role': 'assistant', 'content': assistant_msg},
                    ]
                }
                f_out.write(json.dumps(rec_out, ensure_ascii=False) + '\n')
                count += 1
    return count

train_count, eval_count = build_train_eval()
test_count = build_test()
print(f'train samples: {train_count}')
print(f'eval samples: {eval_count}')
print(f'test samples: {test_count}')
