import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import os
import json
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Импортируем ваши модули (судя по вкладкам в вашем редакторе)
from config.model_config import QwenConfig
from modeling.modeling_qwen import FlaxQwenForCausalLM
from training.checkpoint import load_and_shard_weights
from utils.arg import parse_args
from scripts.tg_notifier import send_telegram_notification
send_telegram_notification("success")
args = parse_args()
# =====================================================================
# ШАГ 1: ЗАГРУЗКА И НАСТРОЙКА КОНФИГУРАЦИИ И ТОКЕНИЗАТОРА
# =====================================================================
print("[Шаг 1] Инициализация конфигурации и токенизатора...")
model_dir = snapshot_download(
    repo_id="Qwen/Qwen3-14B",
    allow_patterns=["*.json", "*.safetensors"]
)

config = QwenConfig(model_dir)
if args.num_hidden_layers is not None:
    config.num_hidden_layers = args.num_hidden_layers
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
    weight_map = json.load(f)["weight_map"]

# Проверяем наличие QK-norm
q_norm_key_test = "model.layers.0.self_attn.q_norm.weight"
k_norm_key_test = "model.layers.0.self_attn.k_norm.weight"
config.use_qk_norm = q_norm_key_test in weight_map and k_norm_key_test in weight_map

# =====================================================================
# ШАГ 2: НАСТРОЙКА TPU ШАРДИНГА
# =====================================================================
print("[Шаг 2] Настройка сетки устройств TPU...")
devices = np.array(jax.devices())
num_devices = len(devices)
devices_mesh = devices.reshape(1, num_devices)
mesh = Mesh(devices_mesh, ('data', 'tensor'))

# Определяем маски для распределения тензоров по чипам TPU
sharding_repl_1d = NamedSharding(mesh, P(None))
sharding_repl_2d = NamedSharding(mesh, P(None, None))
sharding_col = NamedSharding(mesh, P(None, 'tensor'))
sharding_row = NamedSharding(mesh, P('tensor', None))

# =====================================================================
# ШАГ 3: ЗАГРУЗКА ВЕСОВ НА TPU
# =====================================================================
# ВНИМАНИЕ: Для теста скорости генерации можете поставить config.num_hidden_layers = 1,
# чтобы не ждать загрузку всех 40 слоев. Для финального запуска оставьте как есть.
print(f"[Шаг 3] Загрузка и нарезка весов для {config.num_hidden_layers} слоев...")
tpu_params = load_and_shard_weights(
    model_dir=model_dir,
    weight_map=weight_map,
    config=config,
    sharding_repl_1d=sharding_repl_1d,
    sharding_repl_2d=sharding_repl_2d,
    sharding_col=sharding_col,
    sharding_row=sharding_row,
    has_qk_norm=config.use_qk_norm
)

# =====================================================================
# ШАГ 4: ИНИЦИАЛИЗАЦИЯ KV-КЭША И ФУНКЦИЙ JIT
# =====================================================================
print("\n[Шаг 4] Подготовка функций генерации и KV-кэша...")
model = FlaxQwenForCausalLM(config=config)

# 1. Подготавливаем фиктивные данные для создания кэша
dummy_ids = jax.device_put(jnp.ones((1, 1), dtype=jnp.int32), sharding_repl_2d)
dummy_pos = jax.device_put(jnp.zeros((1, 1), dtype=jnp.int32), sharding_repl_2d)

# 2. Холостой прогон для инициализации ТОЛЬКО кэша
# Передаем уже загруженные tpu_params. Flax увидит, что 'cache' не передан,
# создаст его с помощью jnp.zeros и вернет в initial_vars.
_, initial_vars = model.apply(
    tpu_params,          # Наши реальные веса
    dummy_ids, 
    dummy_pos, 
    use_cache=True, 
    mutable=['cache']    # Разрешаем создать кэш
)
kv_cache = initial_vars['cache'] # Забираем готовые пустые матрицы кэша

# 3. Функция 1: Обработка промпта (Prefill)
@jax.jit
def prefill_step(weights, cache, input_ids, position_ids):
    logits, mutated_vars = model.apply(
        {'params': weights, 'cache': cache},
        input_ids, position_ids,
        use_cache=True,
        mutable=['cache']
    )
    return logits, mutated_vars['cache']

# 4. Функция 2: Обработка ОДНОГО нового токена (Decode)
@jax.jit
def decode_step(weights, cache, input_ids, position_ids):
    logits, mutated_vars = model.apply(
        {'params': weights, 'cache': cache},
        input_ids, position_ids,
        use_cache=True,
        mutable=['cache']
    )
    return logits, mutated_vars['cache']

# =====================================================================
# ШАГ 5: ЗАПУСК ГЕНЕРАЦИИ ТЕКСТА
# =====================================================================
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
seq_len = input_ids.shape[1]

# 1. PREFILL: Проглатываем весь текст сразу
tpu_input_ids = jax.device_put(jnp.array(input_ids, dtype=jnp.int32), sharding_repl_2d)
tpu_position_ids = jax.device_put(jnp.arange(seq_len, dtype=jnp.int32)[None, :], sharding_repl_2d)

print(f"\nОбработка промпта: '{prompt}'...")
logits, current_cache = prefill_step(tpu_params["params"], kv_cache, tpu_input_ids, tpu_position_ids)

# Берем логиты самого последнего слова промпта и предсказываем следующее
next_token_id = int(jnp.argmax(logits[0, -1, :]))
generated_tokens = [next_token_id]
print(f"Первый сгенерированный токен: {tokenizer.decode([next_token_id])}")

# 2. DECODE LOOP: Генерируем следующие 10 слов
current_pos = seq_len
MAX_NEW_TOKENS = 10

print("Продолжение генерации (Decode):", end=" ", flush=True)

for i in range(MAX_NEW_TOKENS):
    # На вход подаем только 1 слово (последнее предсказанное)
    step_input_id = jax.device_put(jnp.array([[next_token_id]], dtype=jnp.int32), sharding_repl_2d)
    step_position = jax.device_put(jnp.array([[current_pos]], dtype=jnp.int32), sharding_repl_2d)
    
    # Очень быстрый JIT-проход!
    logits, current_cache = decode_step(tpu_params, current_cache, step_input_id, step_position)
    
    # Выбираем следующее слово
    next_token_id = int(jnp.argmax(logits[0, -1, :]))
    
    # Если модель выдала токен конца текста — останавливаемся
    if next_token_id == 151645: # <|im_end|> для Qwen
        break
        
    generated_tokens.append(next_token_id)
    print(tokenizer.decode([next_token_id]), end="", flush=True)
    
    current_pos += 1

print("\n\nГенерация завершена!")