# generate.py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import os
import json
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
# Импортируем ваши новые модули
from config.model_config import QwenConfig
from modeling.modeling_qwen import FlaxQwenForCausalLM
from training.checkpoint import load_and_shard_weights
from scripts.tg_notifier import send_tg
from utils.arg import parse_args
# =====================================================================
# 1. ЗАГРУЗКА И НАСТРОЙКА КОНФИГУРАЦИИ И ТОКЕНИЗАТОРА
# =====================================================================
args = parse_args()
send_tg()
print("[Шаг 1] Инициализация конфигурации...")
# Путь, куда HF скачивает модель (мы узнаем его после первого запуска snapshot_download)
# Для теста укажем путь к кэшу или скачаем заново:
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

# Проверяем QK-norm
q_norm_key_test = "model.layers.0.self_attn.q_norm.weight"
k_norm_key_test = "model.layers.0.self_attn.k_norm.weight"
config.use_qk_norm = q_norm_key_test in weight_map and k_norm_key_test in weight_map

# =====================================================================
# 2. НАСТРОЙКА TPU ШАРДИНГА
# =====================================================================
print("[Шаг 2] Настройка сетки устройств TPU...")
devices = np.array(jax.devices())
num_devices = len(devices)
devices_mesh = devices.reshape(1, num_devices)
mesh = Mesh(devices_mesh, ('data', 'tensor'))

# Определяем маски для разной размерности тензоров
sharding_repl_1d = NamedSharding(mesh, P(None))
sharding_repl_2d = NamedSharding(mesh, P(None, None))       # Нужна для embed_tokens
sharding_repl_3d = NamedSharding(mesh, P(None, None, None))
sharding_col = NamedSharding(mesh, P(None, 'tensor'))
sharding_row = NamedSharding(mesh, P('tensor', None))

# =====================================================================
# 3. ЗАГРУЗКА ВЕСОВ
# =====================================================================
# Для начала проверьте на 1 или 2 слоях (измените config.num_hidden_layers = 1 для быстрого теста)
# Если все хорошо, верните config.num_hidden_layers = 40 (из конфига)
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
# 4. ПОДГОТОВКА ТЕКСТОВОГО ВВОДА
# =====================================================================
print("[Шаг 4] Токенизация входного текста...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="np")
input_ids_np = inputs["input_ids"] # Имеет форму [1, seq_len]
seq_len = input_ids_np.shape[1]

# Переносим токены на TPU
tpu_input_ids = jax.device_put(jnp.array(input_ids_np, dtype=jnp.int32), sharding_repl_2d)

# Создаем position_ids и маску
position_ids_np = np.arange(seq_len, dtype=np.int32)[None, :]
tpu_position_ids = jax.device_put(jnp.array(position_ids_np), sharding_repl_2d)

# =====================================================================
# 5. КОМПИЛЯЦИЯ И ИНФЕРЕНС С ПОЛНОЙ МОДЕЛЬЮ
# =====================================================================
print("[Шаг 5] Сборка модели и запуск инференса...")
model = FlaxQwenForCausalLM(config=config)

@jax.jit
def inference_step(weights, input_ids, position_ids):
    return model.apply(weights, input_ids, position_ids)

# Делаем проход модели (forward pass)
logits = inference_step(tpu_params, tpu_input_ids, tpu_position_ids)
logits.block_until_ready()

# Берем логиты последнего предсказанного токена
next_token_logits = logits[0, -1, :] # Форма [vocab_size]

# Находим самый вероятный токен (Greedy Search)
next_token_id = int(jnp.argmax(next_token_logits))
predicted_word = tokenizer.decode([next_token_id])

print("\n" + "=" * 60)
print(f"Входной промпт: '{prompt}'")
print(f"ID предсказанного токена: {next_token_id}")
print(f"Декодированное слово: '{predicted_word}'")
print("=" * 60)