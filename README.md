RL-DBAP: Prompted Holdings Prediction �� Quick Guide

����
- Ŀ�꣺�Ѽ��ȳֲ��������ת�ɡ���ʷ�����͡�Prompts����� SFT ����GRPO ǿ��ѵ���������� MAE/IC ��ָ�ꡣ
- ������ڣ��ϸ�ģ����ֲ������`src/cli/build_history_prompts.py`����

һ����ͨ���� clone �����⣩
- ��ȡ���벢����Ŀ¼��
  - `git clone <YOUR_REPO_URL> rl-dbap && cd rl-dbap`
- ��������������Windows PowerShell����
  - `python -m venv .venv && ./.venv/Scripts/Activate.ps1`
  - ��װ PyTorch����ѡ��һ����
    - CUDA 12.1��`pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
    - CPU��`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
  - ����������`pip install -r requirements.txt`
- ��ѡ��ms-swift Դ��/�̶�·����װ
  - Դ�밲װ��`pip install git+https://github.com/modelscope/ms-swift.git`
  - �̶�·����װ��
    - `cd /workspace/rl-dbap` �ĳ����Լ���·��
    - `rm -rf ./ms-swift` ɾ��ԭʼ���ļ���
    - `git clone https://github.com/modelscope/ms-swift.git ms-swift` ֱ��clone����
- ��ѡ����ǰ���ػ���ģ�ͣ�����/����ʱ��
  - `pip install "huggingface_hub[cli]"`
  - `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct --local-dir-use-symlinks False`
- ����׼����
  - `python -m src.cli.prepare_data --config configs/data.yaml`
- ���� prompts��
  - SFT��`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sft --date-end 2016-12-31`
  - GRPO��`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpo --date-start 2017-01-01 --date-end 2018-12-31`
  - Test��`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01`
- ת����
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl`
  - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`��Ĭ�ϲ�������ʧ `<think>` ʾ�����ɼ� `--no-think-example` �رգ�
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

?���������ɵ�һ�������ݼ�����������ˣ�
- ˵����Ϊĳ�� firm type һ������������ prompts��SFT/GRPO/Test������ת���������ݼ���SFT/GRPO/Test����Ĭ��ʱ���з֣�SFT �� 2016-12-31��GRPO 2017-01-01�C2018-12-31��Test �� 2019-01-01��
- ʾ����Banks������ t=0 ��ÿ�ָ�� 1000 ������
  - `python -m src.cli.build_type_datasets --type banks --per-type-limit 1000 --exclude-zero-holding-t`
- ��ѡ������
  - �޸�ʱ���з֣�`--sft-end 2016-12-31 --grpo-start 2017-01-01 --grpo-end 2018-12-31 --test-start 2019-01-01`
  - ���� t=0��ʹ�� `--include-zero-holding-t`��Ĭ���ų���
  - SFT ת���������� `<think>`��`--sft-with-think`
  - GRPO ���������� `<think>` ʾ����`--grpo-no-think-example`

Banks ����ֲֳ�����1000 ����ʾ������
- Ŀ�꣺�ϸ�ʱ���з֣���� Banks ��ȡÿ���ָ�� 1000 ���� t ʱ�ֲֲ̳�Ϊ 0 ��������
- ���� prompts����������˿��أ���
  - SFT���� 2016-12-31����
    - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sft --include-types banks --per-type-limit 1000 --date-end 2016-12-31 --exclude-zero-holding-t`
  - GRPO��2017-01-01�C2018-12-31����
    - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpo --include-types banks --per-type-limit 1000 --date-start 2017-01-01 --date-end 2018-12-31 --exclude-zero-holding-t`
  - Test���� 2019-01-01����
    - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --include-types banks --per-type-limit 1000 --date-start 2019-01-01 --exclude-zero-holding-t`
- ת��Ϊ�������ݼ���
  - SFT��`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft/banks.jsonl --out artifacts/sft/sft_train_banks.jsonl`
  - GRPO��`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo/banks.jsonl --out artifacts/grpo/grpo_banks.jsonl`
  - Test��`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test/banks.jsonl --out artifacts/test/test_banks.jsonl`
- ����У�飨PowerShell����
  - ������`Get-Content artifacts/prompts_hist_sft/banks.jsonl | Measure-Object | % Count`
  - ���㣺`Get-Content artifacts/prompts_hist_sft/banks.jsonl | % { $o = $_ | ConvertFrom-Json; if ([double]$o.holding_t -eq 0) { throw 'found zero in SFT' } }`
  - ע������������ʱ��Ͱ����ʱ����������� 1000�������㴰�����岻�㣬��������
- ѵ����
  - SFT��`powershell .\scripts\sft.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train.jsonl" -OutputDir "outputs/sft_qwen2.5_7b"`
- GRPO��`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
- ���⣺
- Base��`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path None --out_dir artifacts/eval_base`
- SFT��`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
- GRPO��`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`

������������
- �������� prompts��`python scripts/export_infer_prompts.py --in artifacts/sft/test.jsonl --out-dir artifacts/test --stem test`������ base/GRPO ���� `id/system/prompt` JSONL��
- �������ԣ�`python scripts/infer_grpo.py --base_model Qwen/Qwen2.5-7B-Instruct --checkpoint output/grpo_qwen2.5_7b/<run>/checkpoint-1000 --jsonl artifacts/test/test_prompts_grpo.jsonl --index 0`��`--checkpoint None` �ɶԱ�ԭʼģ�ͣ�ͬ��֧�� `--prompt`/`--prompt_file`��
- ��������`python scripts/batch_infer.py --jsonl artifacts/test/test_prompts_grpo.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --checkpoint output/grpo_qwen2.5_7b/<run>/checkpoint-1000 --labels artifacts/sft/test.jsonl --out_jsonl artifacts/test/grpo_outputs.jsonl --out_csv artifacts/test/grpo_outputs.csv --plot_dir artifacts/test/grpo_plots --progress_log_steps 100`
  - �ű��Զ����� `<think>/<answer>`����ȡ `holding_tp1`��������ʵ��ǩ������� JSONL/CSV���ṩ `--plot_dir` ��������ɲв�ֱ��ͼ��Ԥ��Ա�ɢ���������� MAE/RMSE ���ߡ�
  - ��������� `--progress_log_steps` չʾ������������ MAE/RMSE�������ش���������

������װ������˵����
- Windows��PowerShell���������ġ�һ����ͨ����
- Linux/macOS��
  - `python3 -m venv .venv && source .venv/bin/activate`
  - ���� https://pytorch.org ��װ����� `torch/torchvision/torchaudio`
  - `pip install -r requirements.txt`
- ms-swift��`pip install ms-swift` �� `pip install git+https://github.com/modelscope/ms-swift.git`
- ����У�飺`swift --help`��`swift sft --help` �����С�

����ģ�����������
- ���ߣ��״������Զ���ȡ��
- ���ߣ�ʹ�� huggingface_hub ���ص� `models/Qwen2.5-7B-Instruct` ����ѵ��/���������а� `-Model/--base_model` ��Ϊ��Ŀ¼��
- ���棺������ `HF_HOME` �����ظ����ء�

������ Prompt ����
- ׼�����ݣ�`python -m src.cli.prepare_data --config configs/data.yaml`
- �ϸ�ģ�� Prompts��
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist --per-type-limit 1000 --time-bins 10 --cap-per-pair 3 --seed 42 --use-tqdm`
- �Զ���˾��ӳ�䣺�� `data/ticker_mapping.csv` ��ȡ���� `Ticker: {permno}` �滻Ϊ `Ticker: {TICKER��PERMNO} | Company: {COMNAM}`��

ת��Ϊѵ��/���⼯
- SFT��`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl [--with-think]`
- GRPO��`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`��Ĭ�ϲ�������ʧ `<think>` ʾ�����ɼ� `--no-think-example` �رգ�
- Test��`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

ѵ����������ϸ����
- SFT��LoRA���������ġ�һ����ͨ����
- GRPO��LoRA����`scripts/grpo.sh` �� `scripts/grpo.ps1` �̶�ʹ�� `contract_holdings`+`mse_holdings` �� 0.4/0.6 ���Ͻ������ⲿ���·���̶�Ϊ `src/plugins/grpo/holdings_plugin.py`��
- �������`metrics.csv`��`pred_detail.csv`��`residual_hist.png`��`ic_by_quarter.png`��

ģ��ְ��һ��
- `src/cli/prepare_data.py`�����뵽���ȡ����ɱ�ǩ����� parquet ���
- `src/cli/build_history_prompts.py`�������������ϸ�ģ�� t-3..t ������ prompt���Զ���˾��ӳ�䣩
- `src/prompts/sampler.py`������������ֲ�ʱ��Ͱ����
- `src/prompts/builder.py`���ϸ�ģ�� prompt ƴװ��ʮ���Ƹ�ʽ�������Լ
- `src/cli/prompts_to_sft.py`���� prompts תΪ SFT chat ��ʽ����ѡ `<think>` ������
- `src/cli/prompts_to_grpo.py`���� prompts תΪ GRPO ���ݣ�messages + labels��
- `src/plugins/grpo/holdings_plugin.py`���Զ��影����`mse_holdings`��`contract_holdings`��`external_holdings`��
- `scripts/sft.ps1`��`scripts/grpo.ps1`��ѵ���ű���ms?swift CLI��
- `src/cli/run_eval.py`��`src/backends/hf_infer.py`������������

��ʾ
- �����������ѡȡ messages �� `loss=True` �� assistant ��Ϊ��ǩ/������󣬺��� `<think>`��

һ�������Ͷ�������� SFT+GRPO �ܵ�
- Linux/macOS��
  - `bash scripts/run_per_type.sh -t banks -m "Qwen/Qwen2.5-7B-Instruct" -sft_end 2016-12-31 -grpo_start 2017-01-01 -grpo_end 2018-12-31 -g 4 -l 512`
  - �Զ�ִ�У����� per?type SFT prompts �� ת SFT jsonl �� ���� SFT������ per?type GRPO prompts �� ת GRPO jsonl �� �� SFT ������Ϊ������� GRPO��
- Windows/PowerShell��
  - `powershell .\scripts\run_per_type.ps1 -Type banks -Model "Qwen/Qwen2.5-7B-Instruct" -SftEnd "2016-12-31" -GrpoStart "2017-01-01" -GrpoEnd "2018-12-31" -NumGenerations 4 -MaxCompletionLen 512`
  - ͬ������ִ�� per?type �� SFT �� GRPO��

���ѵ������ѵ���������ѱ���
- �������Ѵ��� per?type �� SFT �� GRPO ���ݼ����� `artifacts/sft/sft_train_banks.jsonl`��`artifacts/grpo/grpo_banks.jsonl`����
- Linux/macOS��
  - `bash scripts/train_per_type.sh -t banks -m "Qwen/Qwen2.5-7B-Instruct"`
- Windows/PowerShell��
  - `powershell .\scripts\train_per_type.ps1 -Type banks -Model "Qwen/Qwen2.5-7B-Instruct"`
- ˵�����ű�Ĭ�϶�ȡ `artifacts/sft/sft_train_<TYPE>.jsonl` �� `artifacts/grpo/grpo_<TYPE>.jsonl`�������� SFT ����� `outputs/sft_<TYPE>`������Ը� SFT ������Ϊ������� GRPO������� `outputs/grpo_<TYPE>`��

GRPO �н� SFT ��ϵ���ѵ�����䣩
- �� SFT LoRA ������ GRPO��
  - PowerShell��`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512 -Adapters "outputs/sft_qwen2.5_7b"`
  - Bash��`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 4 -l 512 -a outputs/sft_qwen2.5_7b`
- GRPO �ϵ���ѵ��
  - PowerShell��`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512 -ResumeFrom "outputs/grpo_qwen2.5_7b/checkpoint-1000"`
  - Bash��`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 4 -l 512 -r outputs/grpo_qwen2.5_7b/checkpoint-1000`

��Ͷ�������ͣ�per?type��ѵ��
- ������ϣ����׽��ͬͶ�������͵Ĳ��죬ÿ��ѵ����ʹ�õ�һ���͵����ݣ�SFT �� GRPO �Կɣ���
- ���裨�� `banks` Ϊ������
  1) ֻΪ���������� prompts�������з�ʱ�䣩
     - SFT��`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sft --include-types banks --date-end 2016-12-31`
     - GRPO��`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpo --include-types banks --date-start 2017-01-01 --date-end 2018-12-31`
  2) ��ת�������͵� jsonl Ϊѵ����
     - SFT��`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft/banks.jsonl --out artifacts/sft/sft_train_banks.jsonl`
     - GRPO��`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo/banks.jsonl --out artifacts/grpo/grpo_banks.jsonl`
  3) ʹ�� per?type ��������ѵ��
     - Windows/PS��SFT����`powershell .\scripts\sft.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train_banks.jsonl" -OutputDir "outputs/sft_banks"`
     - Windows/PS��GRPO����`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo_banks.jsonl" -OutputDir "outputs/grpo_banks" -NumGenerations 4 -MaxCompletionLen 512`
     - Linux/macOS��SFT����`bash scripts/sft.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/sft/sft_train_banks.jsonl -o outputs/sft_banks`
     - Linux/macOS��GRPO����`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo_banks.jsonl -o outputs/grpo_banks -g 4 -l 512`
  4) ���������⣨��ѡ���� banks��
     - ���ɲ��Լ����� banks����`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test_banks --include-types banks --date-start 2019-01-01`
     - ת����`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test_banks/banks.jsonl --out artifacts/sft/test_banks.jsonl`
     - ���⣺`python -m src.cli.run_eval --test_path artifacts/sft/test_banks.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_banks --out_dir artifacts/eval_sft_banks`

˵��
- `--include-types` �����Զ��ŷָ����ļ��� stem���� `banks,mutual_funds`����������ʱ����һ����
- ת���׶���� `--in` ָ��Ŀ¼����ϲ�Ŀ¼��ȫ�� jsonl������� per?type ѵ��ʱ����ذ� `--in` ָ�򵥸����͵� jsonl �ļ�·����

ѵ����Python ִ�� �� .sh ������
- Python ִ�У�SFT����
  - `python -m swift.cli.sft --model "Qwen/Qwen2.5-7B-Instruct" --train_type lora --dataset artifacts/sft/sft_train.jsonl --torch_dtype bfloat16 --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32 --target_modules all-linear --logging_steps 20 --save_steps 500 --save_total_limit 2 --max_length 2048 --output_dir outputs/sft_qwen2.5_7b --system "You are a quantitative portfolio manager. Respond with valid JSON only."`
- Python ִ�У�GRPO����
- `python -m swift.cli.rlhf --rlhf_type grpo --model "Qwen/Qwen2.5-7B-Instruct" --external_plugins src/plugins/grpo/holdings_plugin.py --reward_funcs contract_holdings mse_holdings --reward_weights 0.4 0.6 --train_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear --torch_dtype bfloat16 --dataset artifacts/grpo/grpo.jsonl --load_from_cache_file true --max_completion_length 512 --num_train_epochs 1 --per_device_train_batch_size 1 --learning_rate 1e-6 --gradient_accumulation_steps 8 --logging_steps 5 --save_steps 100 --save_total_limit 2 --max_length 2048 --output_dir outputs/grpo_qwen2.5_7b --warmup_ratio 0.05 --dataset_num_proc 2 --num_generations 4 --temperature 0.9 --beta 0.04 --log_completions true`����ֱ�ӵ��� Python CLI�������ֶ����뽱����ϣ��ű��汾�����ã�
- Linux/macOS��.sh ������banks����Ϊ������
  - SFT��`bash scripts/sft.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/sft/sft_train_banks.jsonl -o outputs/sft_qwen2.5_7b`
  - GRPO��`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo_banks.jsonl -o outputs/grpo_qwen2.5_7b -g 4 -l 512`���� `-v` ���� vLLM colocate��
  - macOS ���� `bash` �ɸ��� `zsh` ִ���������
- Windows��PowerShell��������ʹ������ `.ps1` �ű� `scripts/sft.ps1` �� `scripts/grpo.ps1`��

��¼������˵��
- ԭʼ������¼����������ϸ�Ķ��������������������������嵥�������ڲֿ��ļ���`_RESTORE_README.md`���� README �����ṹ��������������֣�δɾ���κ���ʷ��Ϣ��
- �����ҽ� `_RESTORE_README.md` ������ֱ�Ӻϲ����� README ��Ϊ����ϸ�桱�½ڣ����֪���һ�����ϲ�����������ԭ���䡣
Manual Reward Scoring
- Manually inspect reward scores for sampled completions aligned with the GRPO dataset:
- `python -m src.cli.score_rewards \
  `  --dataset artifacts/grpo/grpo_banks.jsonl \
  `  --completions outputs/grpo_qwen2.5_7b/v1-20251018-073116/completions.jsonl \
  `  --external_plugins src/plugins/grpo/holdings_plugin.py \
  `  --reward_funcs contract_holdings mse_holdings \
  `  --reward_weights 0.4 0.6 \
  `  --completion_field completion.0 \
  `  --limit 5`
- Notes: `--completion_field completion.0` selects the first sampled completion when the completions JSONL stores a list. Add `--strict_answer_only` to score only the <answer>...</answer> body. Adjust the completions path to your actual run directory.

��ֵ���ȵ� GRPO ѵ��
- Ŀ�꣺�� HoldingsDeltaORM����ֵ���ȣ������Ż���ContractHoldingsORM ������ʽ/�߽綵�ס�
- PowerShell��Windows����
  - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 8 -MaxCompletionLen 96 -Temperature 0.3 -StopWords "}"`
- Bash��Linux/macOS����
  - `bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 8 -l 96 -T 0.3 -S "}"`
  - ��ʾ��
  - �ű���д�� `contract_holdings`+`mse_holdings`��0.4/0.6�������������������ֱ���޸� `scripts/grpo.sh` / `scripts/grpo.ps1` �еĳ��������� `contract_holdings` �ֻ��������ı����Ƿ� `<think>��</think>`��ȱʧ���� 0 �֡�
  - �� JSON ������� `-S/--StopWords '}'`����ʹ�� XML �����飬��Ϊ `'</answer>'` ����������ģ��һ�¡�
  - �¶Ƚ��� 0.3�C0.6�������߳������`MaxCompletionLen` �趨Ϊ���������� JSON ����С�㹻�����Լ��ٽضϡ�

�� SFT ������������ / �� GRPO �ϵ���ѵ
- Bash��Linux/macOS����
  - �������� SFT LoRA ���������滻 `<SFT_ADAPTER_DIR>` Ϊ��� SFT ������Ŀ¼������ `outputs/sft_qwen2.5_7b/v2-20251021-083409/checkpoint-189`��
    - `bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo_banks.jsonl -o outputs/grpo_qwen2.5_7b -g 8 -l 96 -T 0.3 -S "}" -a <SFT_ADAPTER_DIR>`
  - ������ GRPO ������������滻 `<GRPO_CKPT_DIR>` Ϊ��Ķϵ�·�������� `outputs/grpo_qwen2.5_7b/checkpoint-1000`��
    - `bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 8 -l 96 -T 0.3 -S "}" -r <GRPO_CKPT_DIR>`
  - ˵������δ��ʽ���� `-r`���ű��������Ŀ¼���Զ�������µ� `checkpoint-*` ��Ϊ�ϵ㡣
- PowerShell��Windows����
  - ���� SFT LoRA��
    - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 8 -MaxCompletionLen 96 -Temperature 0.3 -StopWords "}" -Adapters <SFT_ADAPTER_DIR>`
  - �ϵ���ѵ��
    - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 8 -MaxCompletionLen 96 -Temperature 0.3 -StopWords "}" -ResumeFrom <GRPO_CKPT_DIR>`
