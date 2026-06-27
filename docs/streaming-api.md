# Потоковая транскрибация (WebSocket `/v1/realtime`)

ai2npu предоставляет потоковую транскрибацию «на лету» поверх WebSocket по
адресу `ws://127.0.0.1:9555/v1/realtime`. Протокол — **подмножество OpenAI
Realtime** (раздел транскрипции), что позволяет заменять провайдера без
изменения клиента (ai2npu / облако OpenAI / Speaches).

Доступ только с loopback-адреса, без аутентификации.

## Предусловия

- В конфиге включена секция `[streaming]` (`enabled = true`).
- Есть включённая Whisper-модель в `[[models]]` (`type = "whisper"`).
- Доступны OpenVINO runtime и устройство NPU.
- Рядом с `ai2npu.exe` лежит совместимый `onnxruntime.dll` (1.22.x) — кладётся
  автоматически сборкой дистрибутива (`prepare-dist.ps1`). Он используется
  `ort` для запуска VAD и должен опережать `System32\onnxruntime.dll` в поиске
  DLL (каталог exe ищется раньше System32).

> Модель Silero V5 встроена в крейт `voice_activity_detector`, отдельный файл не
> нужен. Поле `vad_model_path` сейчас не используется (зарезервировано).

Пример конфигурации:

```toml
[streaming]
enabled = true
vad_model_path = "models/silero_vad.onnx"
default_min_silence_ms = 400
default_max_segment_ms = 30000
max_input_buffer_sec = 30
```

## Одна сессия одновременно

NPU — однопользовательский ресурс, поэтому активна **ровно одна** стриминговая
сессия. Второе параллельное подключение получает событие `error` с кодом
`streaming_busy`, после чего сокет закрывается. Сессия освобождается при закрытии
сокета (в т.ч. при разрыве): текущий декод фразы доводится до конца кооперативно,
новые сегменты не запускаются (нет «убийства» инференса).

## Формат кадров

Все кадры — **текстовые** JSON-сообщения. Аудио передаётся внутри
`input_audio_buffer.append` как base64. Бинарные WebSocket-кадры **не
поддерживаются** (ответ — `error: invalid_request`).

Аудио: PCM16 (s16le), моно. Частота дискретизации задаётся в
`transcription_session.update` (`sample_rate`); сервер ресэмплит в 16 кГц mono.
Если `sample_rate` не передан, по умолчанию используется 24000 Гц.

### Клиент → сервер

`transcription_session.update` — настройка сессии (можно слать повторно):

```json
{
  "type": "transcription_session.update",
  "session": {
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
      "model": "openai/whisper-large-v3-turbo",
      "language": "ru",
      "prompt": "Команды: Запиши Открой Telegram"
    },
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "silence_duration_ms": 400
    },
    "sample_rate": 16000,
    "max_segment_ms": 30000,
    "word_timestamps": false
  }
}
```

- `input_audio_format` — только `"pcm16"` (иначе `error`).
- `turn_detection.type` — только `"server_vad"` (иначе `error`).
- `silence_duration_ms` — пауза тишины для закрытия сегмента.
- `prompt` — кондиционирование Whisper; следующий сегмент дополнительно получает
  текст уже подтверждённых фраз. **На NPU `prompt` сейчас игнорируется** (см.
  ниже): статический пайплайн NPU зависает в `generate()` при заданном
  `initial_prompt`, поэтому контекст-кондиционирование нейтрализуется на границе
  устройства, как и `word_timestamps`. Обвязка протокола сохранена для
  совместимости и будущих устройств.
- `word_timestamps` — расширение ai2npu (см. ниже), по умолчанию `false`.

`input_audio_buffer.append` — кусок аудио:

```json
{ "type": "input_audio_buffer.append", "audio": "<base64 pcm16>" }
```

`input_audio_buffer.commit` — принудительно закрыть текущий буфер речи и
отправить его на декод (например, по таймауту на стороне клиента):

```json
{ "type": "input_audio_buffer.commit" }
```

### Сервер → клиент

- `transcription_session.created` `{ "session_id": <u64> }` — сессия готова.
- `transcription_session.updated` — настройки применены.
- `input_audio_buffer.speech_started` `{ "audio_start_ms", "item_id" }` — начало
  фразы по VAD.
- `input_audio_buffer.speech_stopped` `{ "audio_end_ms", "item_id" }` — конец
  фразы.
- `input_audio_buffer.committed` `{ "item_id" }` — буфер фразы отправлен на декод.
- `conversation.item.input_audio_transcription.completed`:

  ```json
  {
    "type": "conversation.item.input_audio_transcription.completed",
    "item_id": "item_0",
    "transcript": "Запиши сообщение",
    "words": [ { "text": "Запиши", "start_ms": 0, "end_ms": 500 } ]
  }
  ```

  Финальный результат одной VAD-фразы. Поле `words` присутствует только при
  включённом `word_timestamps` (расширение ai2npu, не часть стандарта OpenAI).
- `error` `{ "error": { "code", "message" } }`.

Гарантии: события `...completed` упорядочены по фразам (`item_id`), не теряются и
не дублируются; `speech_started`/`speech_stopped` обрамляют каждую фразу. Одна
VAD-фраза = один Realtime-«item».

## Партиалы (`delta`) — Phase 2, в реализации

> Контракт зафиксирован (взят из OpenAI Realtime); реализация ведётся в ветке
> `phase2-streaming-delta` по плану
> `docs/superpowers/plans/2026-06-27-streaming-delta-partials.md`. До мерджа в
> `main` сервер партиалы **не** эмитит.

`conversation.item.input_audio_transcription.delta` — промежуточный результат
фразы по микропаузе VAD:

```json
{
  "type": "conversation.item.input_audio_transcription.delta",
  "item_id": "item_0",
  "content_index": 0,
  "delta": "Запиши"
}
```

- Семантика **append-only**: `delta` содержит только новый текст, дописываемый к
  ранее полученному для того же `item_id`. Авторитетный текст — в `...completed`.
- Включается `streaming.partial_silence_ms > 0` (порог микропаузы, `< min_silence_ms`);
  `0` (по умолчанию) — партиалы выключены.
- Каждый партиал — полный прогон Whisper на NPU (нет инкрементального декодинга
  на статическом пайплайне); очередь сериализует, порядок `delta` перед
  `completed` сохранён.
- В Phase 2 в `...completed` добавляется поле `content_index: 0` (паритет с OpenAI).

## Коды ошибок

`openvino_unavailable`, `npu_unavailable`, `model_not_found`, `model_load_failed`,
`inference_failed`, `streaming_busy`, `invalid_request`.

## Расширения ai2npu (вне стандарта OpenAI Realtime)

- `session.sample_rate` — частота входного PCM16.
- `session.max_segment_ms` — принудительная нарезка длинной речи без пауз.
- `session.word_timestamps` + `words` в `completed` — пословные таймстампы.

> Примечание: пословные таймстампы на NPU пока возвращаются пустыми. OpenVINO
> GenAI требует включать `word_timestamps` на этапе конструирования
> `WhisperPipeline` (декомпозиция cross-attention SDPA), что несовместимо с одним
> общим NPU-пайплайном и ограничением single-context. Обвязка протокола готова и
> протестирована; нативное заполнение `words` — follow-up.

## Контракт с фоновой программой / оркестратором

1. Подключиться к `ws://127.0.0.1:9555/v1/realtime`.
2. Отправить `transcription_session.update`, затем `input_audio_buffer.append`
   (base64 PCM16) по мере захвата звука.
3. Читать события сервера; первый `...completed` / первое слово — команда, далее
   контекст. Использовать `speech_started`/`speech_stopped`, при необходимости
   `word_timestamps` и `input_audio_buffer.commit` по таймауту.
4. Соблюдать правило «одна сессия»: не открывать второй стрим параллельно.

Поскольку контракт совместим с транскрипцией OpenAI Realtime, клиент заменяем
(ai2npu / облако OpenAI / Speaches).
