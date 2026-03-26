# Fixes para Compatibilidade BoosterT1_train → booster_deploy

## Contexto

O modelo treinado neste repositório não roda no `booster_deploy` porque o vetor de
observação que o ambiente de treino gera **não é igual** ao que a `LocomotionPolicy`
do `booster_deploy` constrói em runtime. O modelo exportado precisa aceitar exatamente
o mesmo input que o deploy vai fornecer.

---

## Diagnóstico: o que está errado

### Vetor de observação por frame

| Componente | Treino atual | booster_deploy (T1WalkControllerCfg) | Compatível? |
|---|---|---|---|
| `base_lin_vel` | removido no flat env | ausente | ✅ |
| `base_ang_vel` | 3D | 3D | ✅ |
| `projected_gravity` | 3D | 3D | ✅ |
| `velocity_commands` | **4D** (`heading_command=True`) | **3D** (vx, vy, vyaw) | ❌ |
| `joint_pos` | **23 joints** (inclui Head) | **21 joints** (sem Head) | ❌ |
| `joint_vel` | **23 joints** | **21 joints** | ❌ |
| `last_action` | **23D** | **21D** | ❌ |
| **Total por frame** | **79** | **72** | ❌ |

### History stacking

| | Treino atual (rsl_rl padrão) | booster_deploy padrão |
|---|---|---|
| Frames empilhados | **1** (sem history) | **10** |
| Input total da rede | **79** | **720** |

O modelo treinado esperaria 79 valores, mas recebe 720 → erro de shape na inferência.

---

## Mudanças necessárias

### Repo 1: BoosterT1_train

**Arquivo:** `source/train/tasks/manager_based/kickao/robots/T1/env_cfg.py`

#### Mudança 1 — `heading_command=False` em `CommandsCfg`

`heading_command=True` faz `generated_commands` retornar 4 valores `[vx, vy, vyaw, heading]`.
O deploy só usa 3 `[vx, vy, vyaw]`.

```python
# ANTES
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,          # <-- gera 4D
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

# DEPOIS
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,          # desativa heading envs
        heading_command=False,         # <-- gera 3D: [vx, vy, vyaw]
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
```

#### Mudança 2 — Excluir joints da cabeça em `ActionsCfg`

O treino usa `[".*"]` que inclui `Head_Pitch` e `Head_Yaw` (23 joints).
O deploy controla apenas 21 joints (sem cabeça).

```python
# ANTES
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],            # <-- inclui Head_Pitch, Head_Yaw
        scale=T1_ACTION_SCALE,
        use_default_offset=True,
    )

# DEPOIS
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["(?!.*Head).*"],  # <-- exclui Head_Pitch e Head_Yaw
        scale=T1_ACTION_SCALE,
        use_default_offset=True,
    )
```

> **Nota:** com `joint_names=["(?!.*Head).*"]` o espaço de ação passa a ter 21
> dimensões. A ordem dos joints no vetor de observação (`joint_pos_rel`,
> `joint_vel_rel`, `last_action`) segue a ordem do URDF. Verifique que a ordem
> bate com `policy_joint_names` no `T1WalkControllerCfg` do booster_deploy
> (listada abaixo na seção de verificação).

#### Mudança 3 — Verificar `T1KickFlatEnvCfg` herda da classe correta

No arquivo atual, `T1KickFlatEnvCfg` herda de `T1LocomotionEnvCfg` e
`T1KickPlayEnvCfg` herda de `T1LocomotionFlatEnvCfg`, mas essas classes não estão
definidas nem importadas no arquivo. Provavelmente deveriam herdar de `T1KickEnvCfg`.

```python
# VERIFICAR / CORRIGIR
class T1KickFlatEnvCfg(T1KickEnvCfg):   # era T1LocomotionEnvCfg
    ...

class T1KickPlayEnvCfg(T1KickFlatEnvCfg):  # era T1LocomotionFlatEnvCfg
    ...
```

---

### Repo 2: booster_deploy

**Arquivo:** `booster_deploy/tasks/locomotion/__init__.py`

Criar uma nova task apontando para o modelo treinado neste repo, com
`actor_obs_history_length=1` (porque o rsl_rl padrão não empilha history).

```python
# Adicionar ao final do arquivo

@configclass
class T1KickWalkCfg(T1WalkControllerCfg):
    """Policy de locomotion treinada no BoosterT1_train (sem history stacking)."""
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_kick.pt"  # nome do arquivo exportado
        self.policy.actor_obs_history_length = 1            # modelo sem history

register_task("t1_kick", T1KickWalkCfg())
```

---

## Fluxo completo após as mudanças

### Passo 1 — Treinar
```bash
# Na raiz do BoosterT1_train
python scripts/rsl_rl/train.py --task Booster-T1-Kick-Flat-v0
```
Checkpoints salvos em: `logs/rsl_rl/t1_locomotion/<data>/`

### Passo 2 — Exportar o modelo TorchScript
```bash
python scripts/rsl_rl/play.py --task Booster-T1-Kick-v0-Play --headless
```
Modelo exportado em: `logs/rsl_rl/t1_locomotion/<run>/exported/t1_locomotion_<run>.pt`

### Passo 3 — Copiar para o booster_deploy
```bash
cp logs/rsl_rl/t1_locomotion/<run>/exported/t1_locomotion_<run>.pt \
   booster_deploy/tasks/locomotion/models/t1_kick.pt
```

### Passo 4 — Testar em simulação MuJoCo
```bash
cd booster_deploy
python scripts/deploy.py --task t1_kick --mujoco
```

### Passo 5 — Rodar no robô real
```bash
python scripts/deploy.py --task t1_kick --net <IP_DO_ROBO>
```

---

## Verificação: ordem dos joints

O deploy usa `real2sim_joint_map` para reordenar os joints. A ordem esperada pelo
`T1WalkControllerCfg` é:

```
Índice  Joint
  0     Left_Shoulder_Pitch
  1     Right_Shoulder_Pitch
  2     Waist
  3     Left_Shoulder_Roll
  4     Right_Shoulder_Roll
  5     Left_Hip_Pitch
  6     Right_Hip_Pitch
  7     Left_Elbow_Pitch
  8     Right_Elbow_Pitch
  9     Left_Hip_Roll
 10     Right_Hip_Roll
 11     Left_Elbow_Yaw
 12     Right_Elbow_Yaw
 13     Left_Hip_Yaw
 14     Right_Hip_Yaw
 15     Left_Knee_Pitch
 16     Right_Knee_Pitch
 17     Left_Ankle_Pitch
 18     Right_Ankle_Pitch
 19     Left_Ankle_Roll
 20     Right_Ankle_Roll
```

O `booster_deploy` faz o remapeamento automaticamente via `real2sim_joint_map`,
então a **ordem da rede** (ordem do URDF no treino) não precisa ser idêntica à lista
acima — o mapeamento cuida disso. O que **precisa** ser igual é a **lista de nomes**:
todos os 21 joints acima devem existir no URDF e ser controlados pelo treino
(sem Head_Pitch e Head_Yaw).

---

## Resumo das mudanças

| Arquivo | O que muda | Por quê |
|---|---|---|
| `env_cfg.py` — `CommandsCfg` | `heading_command=False`, remover `heading_control_stiffness` e `heading` range | velocity_commands precisa ser 3D |
| `env_cfg.py` — `ActionsCfg` | `joint_names=["(?!.*Head).*"]` | excluir Head_Pitch e Head_Yaw (23→21 joints) |
| `env_cfg.py` — herança | `T1KickFlatEnvCfg(T1KickEnvCfg)` e `T1KickPlayEnvCfg(T1KickFlatEnvCfg)` | corrigir herança quebrada |
| `booster_deploy/tasks/locomotion/__init__.py` | Nova task `t1_kick` com `actor_obs_history_length=1` | modelo exportado sem history stacking |
