嚜燎L-DBAP: Prompted Holdings Prediction ??Quick Guide

憪???
- ?拆???瘞剖次??摰喲雿瑞疏?瞉??????皛溶??仿?嚚琿瞍ompts?掃?祇?SFT ??拇??GRPO 撖桀?撖脩????掃???芸? MAE/IC 蝏?撖??乒?- ?敶?瘞怠????元蝚???脣?梢?`src/cli/build_history_prompts.py`??
???券??
- Python 3.10+????颱? parquet ?餅蝺菟??
1) ???祇??????格氖?砍???摰?+ ?Ｙ???拆??- ?啣???甇onfigs/data.yaml`
- ?拇??甇ython -m src.cli.prepare_data --config configs/data.yaml`
- 瘚Ⅶ??data/processed/panel_quarter.parquet/*.parquet`

2) ?Ｙ???衣憒胼?瞉?Prompts
- 瘨?詨旬???鬼敶????撌駁??`t-3..t`?蝴撣寥???瘚?蝚瘛枚RICT OUTPUT CONTRACT?交??砍???典洩蝜璊方租皝?`<answer>` 瘨?蝎圈??`{"holding_delta": <float>}` ??`{"holding_tp1": <float>}`??憓? ????靘瞏芰????????芰???- ?蝻?? - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist --per-type-limit 1000 --time-bins 10 --cap-per-pair 3 --seed 42 --use-tqdm`
- ??剝?甇rtifacts/prompts_hist/{type}.jsonl`

?脣?梁??急???ata Sampling??- ?拍?餌?璆敶?瘞剖? `(type, mgrno, permno)` ?箸?蝪剝?摨舐?瘚蝜甈??蝜????湛蜀畾?`t-3,t-2,t-1,t` 蝏?敶???憒?蝝啁??亦?拙蝝??- ???圈??ˋ憒捆蝝啁菔??⊥????批?砍鋡恍戭嗥???`qid_t`????id??隞???????? `B=time_bins` 瘨?璊?摮???????[3,12]??蝝滬?餌???璉踵?憍??批撖?????- 憒滓?湧撊???per_type_limit` ?艾???萇恐??瞏??祇????蝬??踹?蝻?蝝?撅潛??蝴?⊥???????交???皝圈??- ?萇?桅?憭貊??蝝圈?撅潛契 `(mgrno,permno)` ?典?仿?嚚鼓??祇??祆噤瘞砍? `cap_per_pair` 瘨?蝝?????蝡渡?儭賢?/??蝡湧嚗瘨租?梢?瘨??游?Ｕ?祆????+ ??皞?菜韐⊿瞈神?怠?蝝?砍?瞉嗆鬼?梢???- ??皞?癸????脤?撣桃敦瘚?楊??`--seed` ?亙???`numpy.random.default_rng`??敶脫噤撊??掉蝡湧???蝻???- 蝏?敶噤??祆?摨Ｙ蝏?蝝圈?亦??`holding_t1`?掉蝝啁?嚗 `label_delta=holding_t1?剜?olding_t` 瘨漆???斤?皜???颲典???賣童頝冽??撌望噤颲函蝏蝚什???皝圈?詨???- ??剝?????  - `--per-type-limit` 憪????∠市?啣鋡恍戭祆???皝唳?憍的??撣寥?1000??000????  - `--time-bins` ??璉輸???撣桃??箝撏?8??2????  - `--cap-per-pair` 憪????`(mgrno,permno)` ??祆?憍的??撣寥?2??????  - `--include-types/--exclude-types` ?碧?蝏怨租?琿?瘝?-date-start/--date-end` ?碧???璉輸?冽曾??--max-files/--head` ?Ｕ蝪祈??色?餌?????  - ?拇?摰喲?甇?-use-tqdm` ??`--progress-every` ??皛臬砟?寧?????剝瞈??湛蛹??
3) ??撋?????撣寥?兩皛???蝡琿?撅潛??脣?敶瞈?
- ?詨?璊?蝧?憡?蝝?
  - SFT??--date-end 2016-12-31` ??剝??`artifacts/prompts_hist_sft`
  - GRPO??--date-start 2017-01-01 --date-end 2018-12-31` ??剝??`artifacts/prompts_hist_grpo`
  - Test??--date-start 2019-01-01` ??剝??`artifacts/prompts_hist_test`

4) ????賜撖?- SFT??python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl`
- GRPO??python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
- Test??python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

5) ??蝎?
- SFT???戭嗥?LoRA??蝝圈???`scripts/sft.ps1`?掃??`--dataset` ?詨??`artifacts/sft/sft_train.jsonl`??- GRPO?鬼憓賜?撗敦
  - `.\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
  - 璁阮??`--reward_funcs contract_holdings external_holdings format`

6) 瞈憪喲??交??s-swift??- `contract_holdings`???詨祚敹?蝏橘蜈蝝滬?拿??蝝?/1??  - ?????契瘨?`<answer>??/answer>` ??瞏⊿?撅潛???瞏⊿??剖膚??JSON ?萇????  - ?蝸?????遴蝞???祆??曄??撟??砍??敦`{"holding_delta": <float>}` ??`{"holding_tp1": <float>}`?閒?剔?敹???  - ??祉?蝜璊斤?瑽賊?雿賜??霂??貊??? 瘚???祇?撣桃敢蝏蝴?????????芰???e/E??蝝梢??船瘚???祆?敶??????  - 蝏橘蜇瞏恍?甇olding_tp1 ??0`??憳ａ?芰殿 `holding_t`?掃??`holding_delta ??-holding_t`??  - 憍文??渡??怠瞈憪?1.0?掃???0.0??斗?摨Ｗ???詨祚敹???蝚?憍????- `external_holdings`?????瑟噤撊??偉?箇遝?+ ????  - 璉啣蟡??拆??撜啣???    - 璉啣蟡?pred?鬥蝝剝??`holding_delta`????甈 `holding_tp1 ??holding_t`?皜園??holding_t????    - ?拆??target?鬥蝝剝??`label_delta`????甈 `label_tp1 ??holding_t`??    - ??璅?`e = pred ??target`??瘣圈??抒???`r = target`??  - ?脣?撉??批妊 R_mag???假 Huber?掛?捂?賣ㄓ?食蝝??    - Huber ?寧??甇?蚜uber(e;c) = 0.5 e^2 (|e|?殉?)??c(|e|??.5c) (|e|>c)`??    - ????c ?典???儭潛敦璁阮??`c = k_mag 頝?EMA_雿?|e|)`???函??`robust_mode ??{mad, iqr}` 瘚?楊??`k頝烘AD` / `k頝涅QR`??    - 銴唳?蝡湧??批? [0,1]??R_mag = 1 ??min(?拇Huber / (0.5 c^2), 1)`??  - ????批妊 R_dir??璉日??唳??元?祇?憭敦
    - ????怠???s = (pred / c_dir) 頝?sign(target)`??c_dir = k_dir 頝?EMA_雿?|target|)`????MAD/IQR????    - 撉蝎阡?喳???R_dir = sigmoid(隡?(s ??m))`?掃?暹??隡??碧?摰?摰喲??亦盔??5??蝝 瘨????????蝝?璅遴 0????  - ?祈租???敦`R = w_mag 頝?R_mag + w_dir 頝?R_dir`?蝎舐??`w_mag=0.6, w_dir=0.4`????  - ?????剖憔??wargs??蝝躬k_mag, k_dir, ema_lambda, alpha, margin, w_mag, w_dir, robust_mode`???湧?扳榆??EMA ??砌蔆鈭??假??摰喲??
7) ???摨Ｙ打??- ?Ｙ?迄戭剔?蝝砟曏遴??蝺?瘞砍瘚蝝?戭軋 2019+??蝝?
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01 --use-tqdm`
  - ??撏脫?曏輻?憡游洵??chat ??蝝⊿??sistant ?蝎瑞菔????????    - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`
- ?拇???祉打??  - Base??璉?LoRA??蝝?
    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path None --out_dir artifacts/eval_base`
  - SFT LoRA??    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
  - GRPO LoRA??    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
- ?蟡湔??憓輸??  - `metrics.csv`??恍?怠滂?葬AE?虞MSE?虞2?劂MAPE%?腹C?虞ankIC?虞ecall/Precision/NDCG@50??  - `pred_detail.csv`??祆?梢??y_true/y_pred/quarter/valid??  - `residual_hist.png`?馳ic_by_quarter.png`
  - `compare.txt`???折?抒 `--post_csv_for_compare`??蝺剝??寞???Base ??MAE ??唳瘨?95% CI??
??祉蝚瘨拿
- `--use-tqdm` ??`--progress-every N` ?菜撋瘨拿??--date-start/--date-end`?馳--max-files/--head` ???拚?瑞?????
?甈Ｙ菔??
- ?Ｙ???脣?梢?甇rc/cli/build_history_prompts.py`?馳src/prompts/sampler.py`?馳src/prompts/builder.py`
- ??撋????src/cli/prepare_data.py`?馳src/dataio/*`
- SFT??src/cli/prompts_to_sft.py`
- GRPO??src/cli/prompts_to_grpo.py`
- ???甇rc/cli/time_split_sft_jsonl.py`?馳src/cli/split_sft_jsonl.py`
- ???甇rc/cli/run_eval.py`?馳src/backends/hf_infer.py`?馳src/evaluation/metrics.py`
- 瞈憪喲?甇s-swift/examples/train/grpo/plugin/plugin.py`??contract_holdings`?馳external_holdings`??
?拇??SFT/GRPO?????????縞蝝?
- ???SFT??oRA??  - `swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset artifacts/sft/sft_train.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --logging_steps 20 \
    --save_steps 500 \
    --save_total_limit 2 \
    --max_length 2048 \
    --output_dir outputs/sft_qwen2.5_7b \
    --system "You are a quantitative portfolio manager. Respond with valid JSON only."`
- ???GRPO??oRA??  - `.\\scripts\\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`

- ???SFT???拚?詨╪?急偉?潛?斤?
  - ?Ｙ??mini ??撋??    - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sftmini --include-types banks --date-end 2016-12-31 --per-type-limit 100 --time-bins 6 --cap-per-pair 2 --head 200000 --use-tqdm`
    - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sftmini --out artifacts/sft/sft_train_mini.jsonl --limit 1000`
  - ??蝎????    - `swift sft --model Qwen/Qwen2.5-7B-Instruct --train_type lora --dataset artifacts/sft/sft_train_mini.jsonl --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 2e-4 --lora_rank 8 --lora_alpha 16 --target_modules all-linear --max_length 1024 --logging_steps 10 --save_steps 200 --save_total_limit 2 --output_dir outputs/sft_debug`

- ???GRPO???拚?詨╪?急偉?潛?斤?
  - ?Ｙ??mini ??撋??    - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpomini --include-types banks --date-start 2017-01-01 --date-end 2018-12-31 --per-type-limit 100 --time-bins 6 --cap-per-pair 2 --head 200000 --use-tqdm`
    - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpomini --out artifacts/grpo/grpo_mini.jsonl --limit 1500`
  - ??蝎????    - `.\\scripts\\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo_mini.jsonl" -OutputDir "outputs/grpo_debug" -NumGenerations 2 -MaxCompletionLen 256`

瞈∪?蝬????交??∠?色瘞??- 瘚Ⅶ?那皝芷??  - SFT??outputs/sft_qwen2.5_7b` ??`outputs/sft_debug` ?撟?`adapter_config.json`?馳adapter_model.safetensors` 蝏???  - GRPO??outputs/grpo_qwen2.5_7b` ??`outputs/grpo_debug` ?撟?checkpoint 瘨滯璉抵?璊潑?- ?蝜??喳蝙??  - SFT?限oss 憪??嗆?戭格疝/??菜???Saving state at step ...` 憪??園?喳???  - GRPO?偃?戭芸? `contract_holdings`/`external_holdings` 瞈憪喲?瘨??掃???瘞喟???0 瘨?畾Ｘ??餅撏?摮珞mag/R_dir ?批?祇蝚??乒?- ??皝啁??魂??  - `SFT:  (Get-ChildItem artifacts/prompts_hist_sft*.jsonl | Get-Content).Count`
  - `GRPO: (Get-ChildItem artifacts/prompts_hist_grpo*.jsonl | Get-Content).Count`

蝏?憪砟曏遴?
- ??撏?24GB??10/3090/4090??蝝圈??曾?箝蝒?7B LoRA?FT/GRPO??蝝??桀拿 1024??048??- 16GB?鬥蝎??7B LoRA?掃蝻??潛 `--max_length` ????768/1024?掛?券?`lora_rank`=4??- 8??2GB?鬼?潮?亦憒胼喟琿?? `Qwen/Qwen2.5-1.5B-Instruct` ??`Qwen/Qwen2.5-3B-Instruct`??蝝寅RPO ??`-NumGenerations` ?韐?1????
RL-DBAP: Prompted Holdings Prediction — 完整说明（UTF‑8）

概览
- 目标：把季度持仓面板数据转成“历史窗口型”Prompts，完成 SFT 热身与 GRPO 强化训练，并评估 MAE/IC 等指标。
- 核心入口：严格模板与分层采样（`src/cli/build_history_prompts.py`）。

1) 环境与数据
- Python 3.10+；数据以 parquet 提供。
- 准备数据（对齐到季度 + 生成标签）：
  - 配置：`configs/data.yaml`
  - 运行：`python -m src.cli.prepare_data --config configs/data.yaml`
  - 产物：`data/processed/panel_quarter.parquet/*.parquet`

2) 生成严格模板 Prompts
- 严格模板：历史 `t-3..t`、推理指令与 STRICT OUTPUT CONTRACT；`<answer>` 输出以下二选一（≤6 小数，非科学计数法）：
  - `{ "holding_delta": <float> }`（首选）
  - `{ "holding_tp1": <float> }`
- 构建示例（全量）：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist --per-type-limit 1000 --time-bins 10 --cap-per-pair 3 --seed 42 --use-tqdm`
- 输出：`artifacts/prompts_hist/{type}.jsonl`

3) 采样策略（Data Sampling）
- 连续窗口：按 `(type, mgrno, permno, date)` 排序，仅保留严格连续季度的 `t-3,t-2,t-1,t` 窗口（断档跳过）。
- 分层时间桶：对每个投资者类型，对 `qid_t`（季度 id）做分位数切分为 `B=time_bins` 个时间桶（自动夹在 [3,12]），确保时间均匀覆盖。
- 桶内配额：`per_type_limit` 在各桶之间均匀分配（余数前置），保证每段时间都有样本。
- 对对儿上限：同一 `(mgrno,permno)` 的窗口在全局最多取 `cap_per_pair` 个；桶内采用“轮转 + 随机打乱”挑选，提升多样性。
- 随机性与可复现：使用 `--seed` 固定 `numpy.random.default_rng`；可复现同一划分结果。
- 标签与符号：若存在 `holding_t1`，计算 `label_delta = holding_t1 − holding_t` 与符号；缺失标签不影响样本生成。
- 关键参数：
  - `--per-type-limit` 每类型样本上限（建议 1000~10000）
  - `--time-bins` 时间分桶数（建议 8~12）
  - `--cap-per-pair` 每 `(mgrno,permno)` 全局上限（建议 2~3）
  - 其他：`--include/exclude-types`、`--date-start/--date-end`、`--max-files/--head`、`--use-tqdm`、`--progress-every`

4) 数据划分（同分布，不重叠）
- 按时间三段：
  - SFT：`--date-end 2016-12-31` → `artifacts/prompts_hist_sft`
  - GRPO：`--date-start 2017-01-01 --date-end 2018-12-31` → `artifacts/prompts_hist_grpo`
  - Test：`--date-start 2019-01-01` → `artifacts/prompts_hist_test`

5) 转数据格式（SFT/GRPO/Test）
- SFT：
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl`
  - 可选加入非监督 `<think>`：`--with-think`
- GRPO：
  - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
- Test：
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

6) 训练
- SFT（LoRA 示例）：
  - `scripts/sft.ps1`（将 `--dataset` 指向 `artifacts/sft/sft_train.jsonl`）
  - 或：`powershell .\scripts\sft.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train.jsonl" -OutputDir "outputs/sft_qwen2.5_7b"`
- GRPO（LoRA 示例）：
  - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
  - 默认奖励：`--reward_funcs contract_holdings external_holdings format`

7) 奖励函数（ms‑swift）
- contract_holdings（格式契约，硬约束，0/1）：
  - 仅允许一个 `<answer>...</answer>` 区块，且区块内只含 JSON 对象。
  - 仅允许两种键之一且出现一次：`{"holding_delta": <float>}` 或 `{"holding_tp1": <float>}`；键小写；≤6 小数；禁止科学计数法；无多余逗号与字段。
  - 约束：`holding_tp1 ≥ 0`；若提供 `holding_t`，则 `holding_delta ≥ -holding_t`。
  - 满足全部规则记 1.0，否则 0.0（用于抑制格式走样与越界）。
- external_holdings（数值型复合：量级 + 方向）：
  - 预测/目标：
    - 预测 pred：优先 `holding_delta`；否则 `holding_tp1 − holding_t`（需 `holding_t`）。
    - 目标 target：优先 `label_delta`；否则 `label_tp1 − holding_t`。
    - 误差 `e = pred − target`；目标幅值 `r = target`。
  - 量级奖励 R_mag（自适应 Huber，方差无关）：
    - Huber：`ℓ_Huber(e;c) = 0.5 e^2 (|e|≤c)；c(|e|−0.5c) (|e|>c)`。
    - 阈值 c 的稳健尺度：默认 `c = k_mag · EMA_λ(|e|)`；或 `robust_mode ∈ {mad, iqr}` 使用 `k·MAD` / `k·IQR`。
    - 归一化到 [0,1]：`R_mag = 1 − min(ℓ_Huber / (0.5 c^2), 1)`。
  - 方向奖励 R_dir（无方差替代）：
    - `s = (pred / c_dir) · sign(target)`，`c_dir = k_dir · EMA_λ(|target|)`（或 MAD/IQR）。
    - `R_dir = sigmoid(α (s − m))`；α 控陡峭度（默认 5），m 为正向边际（默认 0）。
  - 总奖励：`R = w_mag · R_mag + w_dir · R_dir`（默认 `w_mag=0.6, w_dir=0.4`）。
  - 可调超参（kwargs）：`k_mag, k_dir, ema_lambda, alpha, margin, w_mag, w_dir, robust_mode`；EMA 状态在进程内持久化。

8) 评估与测试
- 生成测试集（建议较晚年份，例如 2019+）：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01 --use-tqdm`
  - 转换为评测用 chat 格式（assistant 含绝对标签）：
    - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`
- 运行评测：
  - Base（无 LoRA）：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path None --out_dir artifacts/eval_base`
  - SFT（LoRA）：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
  - GRPO（LoRA）：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
- 评测产物：
  - `metrics.csv`（覆盖率、MAE、RMSE、R2、sMAPE%、IC、RankIC、Recall/Precision/NDCG@50）
  - `pred_detail.csv`（逐样本 y_true/y_pred/quarter/valid）
  - `residual_hist.png`、`ic_by_quarter.png`

9) Mini 调试（可选）
- SFT mini：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sftmini --include-types banks --date-end 2016-12-31 --per-type-limit 100 --time-bins 6 --cap-per-pair 2 --head 200000 --use-tqdm`
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sftmini --out artifacts/sft/sft_train_mini.jsonl --limit 1000`
- GRPO mini：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpomini --include-types banks --date-start 2017-01-01 --date-end 2018-12-31 --per-type-limit 100 --time-bins 6 --cap-per-pair 2 --head 200000 --use-tqdm`
  - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpomini --out artifacts/grpo/grpo_mini.jsonl --limit 1500`
- 训练：`.\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo_mini.jsonl" -OutputDir "outputs/grpo_debug" -NumGenerations 2 -MaxCompletionLen 256`

10) 备注
- 评测解析优先选取 messages 中 `loss=True` 的 assistant 作为标签/对齐对象，忽略 `<think>`。
- Ticker → 公司名映射：`build_history_prompts` 生成时自动从 `data/ticker_mapping.csv` 替换。
