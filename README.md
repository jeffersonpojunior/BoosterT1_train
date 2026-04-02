# Booster RL Tasks

Repositório de tarefas de reinforcement learning para o robô Booster T1, usando [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).

Inclui o framework [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) para motion tracking adaptado ao Booster T1.

Testado com IsaacLab 0.54.3 e Isaac Sim 5.1.0.

---

## Requisitos

- Ubuntu 22.04 ou 24.04
- GPU NVIDIA com driver >= 525 (testado com RTX 5060 Ti, driver 590)
- CUDA 12+
- [uv](https://docs.astral.sh/uv/) instalado
- ~20 GB de espaço em disco

---

## Instalação rápida

```bash
# 1. Crie uma pasta raiz e clone este repositório
mkdir ~/booster_train && cd ~/booster_train
git clone https://github.com/robocin/BoosterT1_train.git BoosterT1_train

# 2. Rode o script de instalação
cd BoosterT1_train
./install.sh
```

O script faz automaticamente:
- Cria um venv Python 3.11 com `uv` em `../.venv/`
- Clona o IsaacLab e o booster_assets
- Instala o Isaac Sim (~10GB — requer aceitar o EULA da NVIDIA)
- Instala todos os pacotes Python necessários

> Na **primeira execução** após a instalação, o Isaac Sim baixa extensões adicionais do Kit (~alguns minutos). Isso é esperado.

---

## Instalar uv (se necessário)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # ou abra um novo terminal
```

---

## Estrutura de pastas após a instalação

```
booster_train/          ← pasta raiz
├── .venv/              ← ambiente virtual (criado pelo script)
├── IsaacLab/           ← clonado pelo script
├── booster_assets/     ← clonado pelo script
└── BoosterT1_train/    ← este repositório
```

---

## Verificar instalação

```bash
source ../.venv/bin/activate

# Teste básico
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('Isaac Sim OK'); app.close()"

# Listar ambientes disponíveis
python scripts/list_envs.py
```

Você deve ver os ambientes do T1:
```
Booster-T1-Locomotion-Flat-v0
Booster-T1-Locomotion-Rough-v0
Booster-T1-Locomotion-v0-Play
Booster-T1-Dance-v0
...
```

---

## Treinar

```bash
source ../.venv/bin/activate
cd ~/booster_train/BoosterT1_train

# Terreno plano (mais rápido para convergir)
python scripts/rsl_rl/train.py --task Booster-T1-Locomotion-Flat-v0 --headless

# Terreno rugoso com curriculum
python scripts/rsl_rl/train.py --task Booster-T1-Locomotion-Rough-v0 --headless

# Especificar GPU
python scripts/rsl_rl/train.py --task Booster-T1-Locomotion-Flat-v0 --headless --device cuda:0
```

Logs e checkpoints salvos em:
```
logs/rsl_rl/t1_locomotion/<data_hora>/
```

---

## Retomar treino de onde parou

Use `--resume` com `--load_run` para continuar a partir do último checkpoint:

```bash
# Retomar o run mais recente (pega o último checkpoint automaticamente)
python scripts/rsl_rl/train.py \
    --task Booster-T1-Locomotion-Flat-v0 \
    --headless \
    --resume \
    --load_run <data_hora>

# Retomar de um checkpoint específico
python scripts/rsl_rl/train.py \
    --task Booster-T1-Locomotion-Flat-v0 \
    --headless \
    --resume \
    --load_run <data_hora> \
    --checkpoint model_<iter>.pt
```

Substitua `<data_hora>` pelo nome da pasta do run (ex: `2025-01-15_10-30-00`) em `logs/rsl_rl/t1_locomotion/`.

---

## Visualizar política treinada

```bash
python scripts/rsl_rl/play.py \
    --task Booster-T1-Locomotion-v0-Play \
    --checkpoint logs/rsl_rl/t1_locomotion/<run>/model_<iter>.pt
```

---

## Tarefas disponíveis

| Task ID | Descrição |
|---------|-----------|
| `Booster-T1-Locomotion-Flat-v0` | Locomoção T1, terreno plano |
| `Booster-T1-Locomotion-Rough-v0` | Locomoção T1, terreno rugoso + curriculum |
| `Booster-T1-Locomotion-v0-Play` | Visualização (1 env, sem perturbações) |
| `Booster-T1-Dance-v0` | Dança T1 (motion tracking, requer NPZ) |
| `Booster-K1-MJ_Dance_004-v0` | Dança K1 (motion tracking) |

---

## Preparar dados de motion (BeyondMimic)

```bash
python scripts/mimic/csv_to_npz.py \
    --headless \
    --input_file=<PATH_TO_BOOSTER_ASSETS>/motions/<ROBOT>/<MOTION>.csv \
    --input_fps=<FPS> \
    --output_file=<PATH_TO_BOOSTER_ASSETS>/motions/<ROBOT>/<MOTION>.npz \
    --output_fps=50 \
    --robot=<k1|t1>
```

---

## Deploy

Após treinar e exportar o modelo, use o [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) para rodar em MuJoCo ou no robô real.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pxr'`**
→ Normal fora do contexto do SimulationApp. Os scripts de treino inicializam o SimulationApp automaticamente.

**`Unable to expose 'isaacsim.simulation_app' API: Extension not found`**
→ Aviso na primeira execução enquanto as extensões são baixadas. Pode ignorar se o script terminar com sucesso.

**`isaacsim[all]` falha com conflito de dependências**
→ Conflitos de `starlette` e `numpy` entre `isaacsim` e `isaaclab` são esperados e não afetam o funcionamento.

**EULA prompt no `pip install isaacsim[all]`**
→ Digite `yes` para aceitar a licença NVIDIA Omniverse.

---

## Agradecimentos

- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): framework BeyondMimic para motion tracking de humanoides.
