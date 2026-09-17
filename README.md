# DeepSeek V4.1 Flash — attention-r3-s45

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

## Что лежит в репозитории

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

## 1. Скачать репозиторий вместе с весами

Для большого файла используется [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

```bash
git lfs install
git clone https://github.com/timadinorth/deepseek-v4.1-uncensored.git
cd deepseek-v4.1-uncensored
git lfs pull
python3 scripts/verify_patch.py
```

Если `git lfs` не установлен, сначала установите его средствами своей ОС, например `brew install git-lfs` на Mac. ZIP исходников или clone с отключённым LFS может содержать небольшой pointer вместо1.64GB весов; verifier это обнаруживает и просит выполнить `git lfs pull`.

## 2. Подготовить окружение и место

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

## 3. Скачать исходную модель

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

## 4. Применить готовый patch

```bash
python3 scripts/apply_patch.py \
  --base /workspace/models/DeepSeek-V4.1-Flash \
  --output /workspace/models/DeepSeek-V4.1-Flash-candidate-attn-r3-s45
```

Скрипт проверяет SHA256 patch, список тензоров, shapes/dtypes и хэши всех заменяемых исходных тензоров **до создания выходных shards**. Затем пересобирает затронутые файлы, подставляя готовые weight и scale, и проверяет записанные значения. Существующий output или output внутри исходника отклоняется.

Итоговые документы в checkpoint: `abliteration-manifest.json` и `patch-application.json`. При успешном завершении выводится `Checkpoint exported`. При прерывании частично созданный output сохраняется; для повторной сборки нужен новый каталог либо осознанная очистка этой неполной копии.

Не прибавляйте patch к исходнику: он содержит **замещающие значения**, не дельту. Нельзя менять незатронутые hardlinked shards на месте — они общие с исходником. Для разных файловых систем или независимых физических копий добавьте `--copy-unchanged` и обеспечьте дополнительное место.

## 5. Запустить изменённую модель

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

## Результаты и ограничения

На одинаковых30 вопросах каждого теста исходная и изменённая модели получили одинаковые суммарные результаты: MMLU-Pro26/30, GPQA24/30, GSM8K30/30. Это небольшие выборки, отдельные ошибки различаются. Подробности и исторические метки bundle — в [docs/EVALUATION.md](docs/EVALUATION.md).

Название репозитория не означает доказанного отсутствия всех отказов или сохранения всех способностей. Приватные ответы, reasoning, API-ключи и данные облачного инстанса не включены. Скрипты не создают и не удаляют GPU-инстансы, не устанавливают TTL или auto-terminate.

Исходная модель опубликована по MIT; её copyright notice сохранён в [LICENSE](LICENSE). Это сторонний экспериментальный patch, не официальный релиз DeepSeek.

## Проверка скриптов

В окружении с PyTorch и safetensors:

```bash
python3 -m unittest discover -s tests -v
```

Тесты используют небольшие синтетические shards, не требуют GPU и не скачивают исходную модель.
