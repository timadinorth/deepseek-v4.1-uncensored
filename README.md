# DeepSeek V4.1 Flash — attention-r3-s45

[English](#english) | [Русский](#russian)

<a id="english"></a>

## English

A ready-to-apply weight patch for an experimental **DeepSeek V4.1 Flash** variant, plus scripts to build a separate checkpoint. The weight file contains **78 replacement tensors**. It is neither a complete model nor an additive delta.

| Parameter | Value |
|---|---|
| Base model | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Exact Hugging Face revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Variant | `attention-r3-s45` |
| Edit scope | `attn.wo_b`, layer indices 1–39 |
| Existing candidate parameters | Strength 4.5; rank-3 approximation of the update |
| Patch | `weights/hf-weight-patch.safetensors` |
| Patch size | 1,637,383,960 bytes, approximately 1.64 GB |
| Patch SHA256 | `e28611217f1a40f7244b33e20c9d0a0fcfc9a579d3add4639d323cd5f8675062` |

Use this exact **V4.1 Flash** revision. This patch is not intended for V4 Flash, V4 Flash 0731, V4 Pro, or an API model with a similar name. The scripts check the actual original tensor values, not just the directory name.

### Repository contents

```text
weights/                 Replacement weights, original manifest, and SHA256
scripts/verify_patch.py   Patch verification without PyTorch
scripts/download_base.py Download the pinned base model
scripts/apply_patch.py   Build a separate checkpoint on CPU
scripts/start_sglang.sh  Serve the resulting model on 4 GPUs
base-model-files.json    Sizes and SHA256 hashes of all 48 original shards
docs/EVALUATION.md       Evaluation protocol and limitations
tests/                  Small tensor packaging and application tests
```

This patch does not edit the experts, router, Engram, vision, or DSpark weights. It replaces selected attention output projections and their block scales. The scope and original tensor hashes are recorded in `weights/bundle.json`.

### 1. Download the repository and weights

The large weight file uses [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

```bash
git lfs install
git clone https://github.com/timadinorth/deepseek-v4.1-uncensored.git
cd deepseek-v4.1-uncensored
git lfs pull
python3 scripts/verify_patch.py
```

If `git lfs` is unavailable, install it using your operating system's package manager, for example `brew install git-lfs` on macOS. A source ZIP or a clone with LFS disabled may contain a small pointer instead of the 1.64 GB weight file. The verifier detects this and asks you to run `git lfs pull`.

### 2. Prepare the environment and storage

The application script runs on CPU and requires PyTorch with `float8_e8m0fnu` support and safetensors. Patch verification uses only Python's standard library. Downloading the base model requires huggingface_hub. The closest match to the tested server environment is this pinned Linux amd64 image:

```text
lmsysorg/sglang@sha256:c1633485e2ef3a8562fc93b2e3b686abb31c4499300bf5ce13ee5d491b65c386
```

It contains SGLang `0.0.0.dev1+gda64c5cbb` and PyTorch 2.13.0+cu130. An arbitrary `pip install sglang` does not reproduce this runtime. For a GPU.ai container-native instance, select this image when creating the instance. On a Linux VM with Docker, you can enter the environment as follows:

```bash
mkdir -p /workspace
docker run --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/repo:ro -v /workspace:/workspace \
  -w /repo -it --entrypoint bash \
  lmsysorg/sglang@sha256:c1633485e2ef3a8562fc93b2e3b686abb31c4499300bf5ce13ee5d491b65c386
```

Run the following commands from the repository root inside this environment. For a separate CPU environment, install the dependencies in `requirements.txt`; these provide the checkpoint assembly tools, not a replacement for the tested SGLang runtime.

The 48 original shards occupy **510.30 GB**. Assembly rewrites 39 shards into separate files, requiring another **288.28 GB**; unchanged files use hardlinks by default. Allow approximately 799 GB for both checkpoints, plus the runtime, caches, and free space. The experiment used a 1 TB disk. `--copy-unchanged` requires more storage because unchanged shards are also copied.

Serving was tested on **4×H200 SXM with NVLink**, with Engram offloaded to CPU memory. The instance had a RAM limit of approximately 1.3 TiB; the minimum required RAM was not measured separately. The assembler holds the current shard and patch in memory, so the patch size alone is not a sufficient RAM estimate.

### 3. Download the base model

```bash
python3 scripts/download_base.py \
  --output /workspace/models/DeepSeek-V4.1-Flash
```

The script always uses the pinned revision, not `main`. A fresh download requires 550 GB of free space. After downloading, it checks the sizes of all 48 shards. Full SHA256 verification additionally reads all 510 GB:

```bash
python3 scripts/download_base.py \
  --output /workspace/models/DeepSeek-V4.1-Flash \
  --verify-shards
```

### 4. Apply the existing patch

```bash
python3 scripts/apply_patch.py \
  --base /workspace/models/DeepSeek-V4.1-Flash \
  --output /workspace/models/DeepSeek-V4.1-Flash-candidate-attn-r3-s45
```

The script verifies the patch SHA256, tensor names, shapes/dtypes, and hashes of every original tensor being replaced **before creating output shards**. It then rebuilds the affected files using the replacement weights and scales and verifies the serialized values. An existing output directory, or an output directory inside the base checkpoint, is rejected.

The resulting checkpoint includes `abliteration-manifest.json` and `patch-application.json`. Successful completion prints `Checkpoint exported`. If interrupted, a partially created output directory remains; retry with a new directory or deliberately remove the incomplete copy.

Do not add the patch to the base weights: it contains **replacement values**, not a delta. Do not modify unchanged hardlinked shards in place, because they are shared with the base checkpoint. For different filesystems or physically independent copies, add `--copy-unchanged` and provide the additional storage.

### 5. Serve the modified model

```bash
bash scripts/start_sglang.sh \
  /workspace/models/DeepSeek-V4.1-Flash-candidate-attn-r3-s45
```

Defaults: TP4/EP4, context 262144, total token pool 1048576, four requests, DSpark block size 5, chunked prefill 2048, and private pinned CPU Engram. Thinking is enabled by default. The equivalent explicit command is:

```bash
bash scripts/start_sglang.sh \
  /workspace/models/DeepSeek-V4.1-Flash-candidate-attn-r3-s45 \
  262144 1048576 dspark 2048
```

The API listens on `127.0.0.1:30000`. Its OpenAI-compatible base URL is `http://127.0.0.1:30000/v1`, and its model ID is `deepseek-v41-eval`. Access from another machine requires SSH forwarding. The launcher runs in the foreground; use tmux or your own process management for a persistent session.

Check the server after loading:

```bash
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/get_model_info
curl --fail http://127.0.0.1:30000/server_info
```

The reported model_path must point to the modified checkpoint. Hot reloading quantized weights failed in the tested preview build, so switching checkpoints required a new serving process. The model's native 1M context limit does not mean this repository has validated 1M operation: the actual long-context check used 250079 input tokens with a 256k context configuration.

### Results and limitations

On the same 30 questions from each benchmark, the original and modified models achieved identical aggregate scores: MMLU-Pro 26/30, GPQA 24/30, and GSM8K 30/30. These are small subsets, and individual errors differ. See [docs/EVALUATION.md](docs/EVALUATION.md) for details and an explanation of historical bundle labels.

The repository name does not establish the absence of all refusals or the preservation of every capability. Private responses, reasoning traces, API keys, and cloud instance details are not included. The scripts neither create nor delete GPU instances and do not configure TTL or auto-termination.

The base model is released under MIT; its copyright notice is preserved in [LICENSE](LICENSE). This is a third-party experimental patch, not an official DeepSeek release.

### Test the scripts

In an environment with PyTorch and safetensors:

```bash
python3 -m unittest discover -s tests -v
```

The tests use small synthetic shards, require no GPU, and do not download the base model.

---

<a id="russian"></a>

## Русский

Готовый patch экспериментально изменённых весов **DeepSeek V4.1 Flash** и скрипты для сборки отдельного checkpoint. Файл весов здесь — **78 замещающих тензоров**, а не полная модель и не дельта для сложения.

| Параметр | Значение |
|---|---|
| Исходная модель | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Точная HF-ревизия | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Вариант | `attention-r3-s45` |
| Область правки | `attn.wo_b`, слои с индексами1–39 |
| Параметры существующего кандидата | strength4.5, приближение поправки rank3 |
| Patch | `weights/hf-weight-patch.safetensors` |
| Размер patch | 1 637 383 960 байт, около1.64GB |
| SHA256 patch | `e28611217f1a40f7244b33e20c9d0a0fcfc9a579d3add4639d323cd5f8675062` |

Использовать именно эту ревизию **V4.1 Flash**. Этот patch не предназначен для V4 Flash, V4 Flash0731, V4 Pro или API-модели с похожим именем. Проверяются реальные исходные значения заменяемых тензоров, а не только название каталога.

### Что лежит в репозитории

```text
weights/                 Готовые веса, исходный manifest и SHA256
scripts/verify_patch.py  Проверка patch без PyTorch
scripts/download_base.py Загрузка закреплённого исходника
scripts/apply_patch.py   Сборка отдельного checkpoint на CPU
scripts/start_sglang.sh  Запуск готовой модели на 4 GPU
base-model-files.json    Размеры и SHA256 всех 48 исходных shards
docs/EVALUATION.md       Протокол и ограничения выполненных проверок
tests/                  Малые тесты упаковки и применения тензоров
```

Experts, router, Engram, vision и DSpark не редактируются этим patch. Новые тензоры заменяют выбранные выходные проекции attention и их block scales. Детали области и хэши исходных тензоров сохранены в `weights/bundle.json`.

### 1. Скачать репозиторий вместе с весами

Для большого файла используется [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

```bash
git lfs install
git clone https://github.com/timadinorth/deepseek-v4.1-uncensored.git
cd deepseek-v4.1-uncensored
git lfs pull
python3 scripts/verify_patch.py
```

Если `git lfs` не установлен, сначала установите его средствами своей ОС, например `brew install git-lfs` на Mac. ZIP исходников или clone с отключённым LFS может содержать небольшой pointer вместо1.64GB весов; verifier это обнаруживает и просит выполнить `git lfs pull`.

### 2. Подготовить окружение и место

Скрипт применения работает на CPU и требует PyTorch с поддержкой `float8_e8m0fnu` и safetensors. Проверка patch использует только стандартную библиотеку Python. Для загрузки исходника нужен huggingface_hub. Самый близкий к проверенному серверу вариант — закреплённый Linux amd64 образ:

```text
lmsysorg/sglang@sha256:c1633485e2ef3a8562fc93b2e3b686abb31c4499300bf5ce13ee5d491b65c386
```

Он содержит SGLang `0.0.0.dev1+gda64c5cbb` и PyTorch2.13.0+cu130. Обычный произвольный `pip install sglang` не воспроизводит этот runtime. На GPU.ai container-native можно выбрать этот образ при создании инстанса. Для Linux VM с Docker пример входа в окружение:

```bash
mkdir -p /workspace
docker run --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/repo:ro -v /workspace:/workspace \
  -w /repo -it --entrypoint bash \
  lmsysorg/sglang@sha256:c1633485e2ef3a8562fc93b2e3b686abb31c4499300bf5ce13ee5d491b65c386
```

Следующие команды выполняются из корня репозитория внутри этого окружения. Для отдельного CPU-окружения можно установить зависимости из `requirements.txt`; это только инструменты сборки, не замена проверенного SGLang runtime.

Исходные48 shards занимают **510.30GB**. При сборке отдельно переписываются39 shards, ещё **288.28GB**; незатронутые файлы по умолчанию используют hardlink. Нужны примерно799GB под оба checkpoint плюс runtime, cache и запас. В эксперименте использовался диск1TB. При `--copy-unchanged` потребуется больше места, поскольку неизменённые shards тоже копируются.

Serving проверен на **4×H200 SXM с NVLink** и CPU-offload Engram. У использованного инстанса лимит RAM был около1.3TiB; минимальный необходимый объём RAM отдельно не измерялся. Сборщик держит в памяти содержимое обрабатываемого shard и patch, поэтому одного объёма patch для оценки RAM недостаточно.

### 3. Скачать исходную модель

```bash
python3 scripts/download_base.py \
  --output /workspace/models/DeepSeek-V4.1-Flash
```

Скрипт всегда использует закреплённую ревизию, не `main`. Для чистой загрузки требует550GB свободного места, после загрузки проверяет размеры48 shards. Полная проверка их SHA256 дополнительно читает все510GB:

```bash
python3 scripts/download_base.py \
  --output /workspace/models/DeepSeek-V4.1-Flash \
  --verify-shards
```

### 4. Применить готовый patch

```bash
python3 scripts/apply_patch.py \
  --base /workspace/models/DeepSeek-V4.1-Flash \
  --output /workspace/models/DeepSeek-V4.1-Flash-candidate-attn-r3-s45
```

Скрипт проверяет SHA256 patch, список тензоров, shapes/dtypes и хэши всех заменяемых исходных тензоров **до создания выходных shards**. Затем пересобирает затронутые файлы, подставляя готовые weight и scale, и проверяет записанные значения. Существующий output или output внутри исходника отклоняется.

Итоговые документы в checkpoint: `abliteration-manifest.json` и `patch-application.json`. При успешном завершении выводится `Checkpoint exported`. При прерывании частично созданный output сохраняется; для повторной сборки нужен новый каталог либо осознанная очистка этой неполной копии.

Не прибавляйте patch к исходнику: он содержит **замещающие значения**, не дельту. Нельзя менять незатронутые hardlinked shards на месте — они общие с исходником. Для разных файловых систем или независимых физических копий добавьте `--copy-unchanged` и обеспечьте дополнительное место.

### 5. Запустить изменённую модель

```bash
bash scripts/start_sglang.sh \
  /workspace/models/DeepSeek-V4.1-Flash-candidate-attn-r3-s45
```

Настройки по умолчанию: TP4/EP4, context262144, общий token pool1048576, четыре запроса, DSpark block5, chunked prefill2048, private pinned CPU Engram. Thinking включён по умолчанию. Полная форма с теми же параметрами:

```bash
bash scripts/start_sglang.sh \
  /workspace/models/DeepSeek-V4.1-Flash-candidate-attn-r3-s45 \
  262144 1048576 dspark 2048
```

API слушает `127.0.0.1:30000`, OpenAI-compatible base URL — `http://127.0.0.1:30000/v1`, model ID — `deepseek-v41-eval`. Для доступа с другой машины нужен SSH-forward. Launcher работает в foreground; для постоянной сессии используйте tmux или собственное управление процессом.

Проверка после загрузки:

```bash
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/get_model_info
curl --fail http://127.0.0.1:30000/server_info
```

В model_path должен быть изменённый checkpoint. В использованной preview-сборке горячая замена quantized весов дала ошибку; для смены checkpoint запускали новый процесс. Заводской предел модели1M не означает, что режим1M проверен этим репозиторием: реально проверен запрос на250079 входных токенов при context256k.

### Результаты и ограничения

На одинаковых30 вопросах каждого теста исходная и изменённая модели получили одинаковые суммарные результаты: MMLU-Pro26/30, GPQA24/30, GSM8K30/30. Это небольшие выборки, отдельные ошибки различаются. Подробности и исторические метки bundle — в [docs/EVALUATION.md](docs/EVALUATION.md).

Название репозитория не означает доказанного отсутствия всех отказов или сохранения всех способностей. Приватные ответы, reasoning, API-ключи и данные облачного инстанса не включены. Скрипты не создают и не удаляют GPU-инстансы, не устанавливают TTL или auto-terminate.

Исходная модель опубликована по MIT; её copyright notice сохранён в [LICENSE](LICENSE). Это сторонний экспериментальный patch, не официальный релиз DeepSeek.

### Проверка скриптов

В окружении с PyTorch и safetensors:

```bash
python3 -m unittest discover -s tests -v
```

Тесты используют небольшие синтетические shards, не требуют GPU и не скачивают исходную модель.
