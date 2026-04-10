# K1 Retarget

Kimodo G1 モーション生成 → Booster K1 リターゲットパイプライン。
AMP (Adversarial Motion Priors) ベースの locomotion policy 学習用モーションデータの生成・変換・可視化ツール群。

## 前提条件

- kimodo がインストール済み (`uv pip install -e ./kimodo`)
- booster_assets が `kimodo_ws/booster_assets/` に配置済み
- `mujoco`, `scipy` がインストール済み

```bash
# kimodo_ws/.venv を activate
source /path/to/kimodo_ws/.venv/bin/activate
```

## ディレクトリ構成

```
k1_retarget/
├── README.md
├── robot/                     # K1 MuJoCo モデル (XML, URDF, meshes)
├── scripts/
│   ├── g1_to_k1.py            # G1 qpos CSV → K1 qpos CSV 変換
│   ├── batch_generate.py       # locomotion モーション一括生成
│   ├── batch_lateral.py        # 横移動モーション一括生成
│   ├── tweak_foot_lift.py      # kimodo constraint で足上げブースト
│   ├── gen_small_yaw_pivot.py  # 小yaw超信地旋回モーション生成
│   ├── fix_arms.py             # 腕をニュートラルポーズに固定
│   ├── compute_motion_stats.py # 速度・方向メタデータ計算
│   ├── visualize_k1.py         # K1 モーション MuJoCo 再生
│   ├── visualize_g1.py         # G1 モーション MuJoCo 再生
│   └── visualize_all.py        # ディレクトリ内モーション一括再生
└── motions_k1/
    ├── g1/                    # G1 生成モーション (NPZ + CSV)
    ├── k1/                    # K1 リターゲット済み (CSV, 139 clips)
    └── k1_fixed/              # 腕固定版 (AMP学習用, 139 clips)
```

## パイプライン

### 1. G1 モーション生成 → K1 リターゲット (単体)

```bash
# G1 モーション生成 (CWD を /tmp にする必要あり: kimodo namespace 衝突回避)
cd /tmp && python -m kimodo.scripts.generate \
  "a person walking forward" \
  --model kimodo-g1-rp --output /path/to/output

# K1 にリターゲット
python scripts/g1_to_k1.py output.csv output_k1.csv
```

### 2. locomotion モーション一括生成

```bash
# プロンプト一覧を確認
python scripts/batch_generate.py --dry-run

# 全生成 (37プロンプト × 2サンプル = 74クリップ)
python scripts/batch_generate.py --output-dir motions_k1 --samples-per-prompt 2

# 特定カテゴリのみ
python scripts/batch_generate.py --output-dir motions_k1 --filter turn
```

### 3. 横移動モーション追加生成

```bash
# 足をクロスしない横移動に特化したプロンプト
python scripts/batch_lateral.py --output-dir motions_k1 --samples-per-prompt 3
```

### 4. 超信地旋回モーションの足上げブースト

kimodo が生成する超信地旋回は足のリフトが小さくシャッフル気味になる。
`tweak_foot_lift.py` は kimodo の end-effector constraint 機能を利用して、
既存モーションの swing peak フレームで膝の曲げを増やし、足をより高く上げた
モーションを再生成する。

```bash
# 単体: 既存 G1 NPZ をシードにして足上げブースト版を生成
python scripts/tweak_foot_lift.py \
  motions_k1/g1/turn_in_place_left/turn_in_place_left_00.npz \
  --prompt "a person turning around in place to the left" \
  --duration 4.0 --seed 1642 \
  --output-name turn_in_place_v2_left_00

# パラメータ調整:
#   --knee-boost-rad 0.4   # 膝曲げ追加量 (rad, default 0.4 ≈ 23°)
#   --constraint-weight 4.0 # constraint 追従強度 (default 4.0)
```

仕組み: G1 NPZ の `foot_contacts` からスイング区間を検出 → ピークフレームで
膝の axis-angle X成分に boost を加えた pose を `left-foot` / `right-foot`
constraint として保存 → kimodo `--input_folder` で separated CFG
(text_weight=2.0, constraint_weight=4.0) を使って再生成。

### 5. 小 yaw 超信地旋回の一括生成

AMP で小さい yaw command に追従させるため、ゆっくり回転するステッピングモーションが必要。
2パス方式: baseline G1 生成 → 足上げブースト → K1 リターゲット → 腕固定。

```bash
# プロンプト一覧を確認
python scripts/gen_small_yaw_pivot.py --dry-run

# 全生成 (6プロンプト × 2サンプル = 12クリップ)
python scripts/gen_small_yaw_pivot.py

# 特定カテゴリのみ
python scripts/gen_small_yaw_pivot.py --filter gentle

# パラメータ調整
python scripts/gen_small_yaw_pivot.py --knee-boost-rad 0.3 --samples-per-prompt 3
```

### 6. 腕の固定 (AMP 学習用)

生成モーションの腕が不自然な場合、ニュートラルポーズに固定する。
AMP のディスクリミネータには脚の動きだけ学習させたい場合に有用。

```bash
# 全クリップの腕を固定 (別ディレクトリに出力)
python scripts/fix_arms.py motions_k1/k1/ --output-dir motions_k1/k1_fixed/

# 特定ファイルのみ
python scripts/fix_arms.py motions_k1/k1/lateral*.csv --output-dir motions_k1/k1_fixed/

# プレビュー (書き込みなし)
python scripts/fix_arms.py motions_k1/k1/ --preview lateral_left_shuffle_05_00
```

### 7. 速度・方向メタデータ計算

```bash
python scripts/compute_motion_stats.py motions_k1/k1_fixed/
```

出力: 各クリップの平均速度 (m/s)、移動方向 (deg)、ヨーレート (deg/s) 等を JSON で保存。

## 可視化

### 単体再生

```bash
# K1 モーション
python scripts/visualize_k1.py motions_k1/k1_fixed/walk_forward_normal_00.csv --loop

# G1 モーション (元データ確認)
python scripts/visualize_g1.py motions_k1/g1/walk_forward_normal_00.csv --loop
```

### 一括再生 (ウィンドウを閉じると次へ)

```bash
# K1 全モーション
python scripts/visualize_all.py motions_k1/k1_fixed/

# フィルタ付き
python scripts/visualize_all.py motions_k1/k1_fixed/ --filter lateral
python scripts/visualize_all.py motions_k1/k1_fixed/ --filter turn_in_place_v2
python scripts/visualize_all.py motions_k1/k1_fixed/ --filter pivot_
python scripts/visualize_all.py motions_k1/k1_fixed/ --filter jog

# G1 で確認
python scripts/visualize_all.py motions_k1/g1/ --g1 --filter walk_forward
```

## モーションデータの概要

### yaw rate 分布 (超信地旋回系)

| カテゴリ | クリップ | yaw rate |
|---|---|---|
| 小 yaw | `pivot_quarter_*`, `pivot_gentle_*` | 12 ~ 22°/s |
| 中 yaw | `pivot_slow_*` | 20 ~ 35°/s |
| 大 yaw | `turn_in_place_v2_*` | 42 ~ 45°/s |
| 元データ (v1) | `turn_in_place_left/right_*` | ~45°/s (足上げ不足) |

### 速度分布 (全体)

| カテゴリ | クリップ数 | 速度範囲 |
|---|---|---|
| 静止 (standing) | ~7 | < 0.1 m/s |
| 歩行 (walking) | ~100+ | 0.1 ~ 1.0 m/s |
| 高速 (fast) | ~16 | > 1.0 m/s |
| 旋回 (turning) | ~30+ | yaw > 10°/s |

## リターゲットの仕組み

### G1 → K1 ジョイントマッピング

| 部位 | G1 (29 DoF) | K1 (22 DoF) | 備考 |
|------|-------------|-------------|------|
| 脚 (片側 6) | hip pitch/roll/yaw, knee, ankle pitch/roll | 同一 | 直接マッピング |
| 腰 (3) | waist yaw/roll/pitch | なし | K1 にない (破棄) |
| 肩 (片側) | pitch, roll, yaw | pitch, roll | yaw は K1 にない |
| 肘 (片側) | elbow | elbow pitch | 直接マッピング |
| 手首 (片側 3) | wrist roll/pitch/yaw | elbow yaw | shoulder_yaw → elbow_yaw |
| 頭 | なし | yaw, pitch | G1 にない (0固定) |

### 補正

- **Root XYZ**: 体格比 (K1/G1 ≈ 0.70) でスケーリング
- **Shoulder roll**: ±π/2 オフセット (G1: 腕下げ休止, K1: T-pose 休止)
- **Joint limits**: K1 XML の関節可動域でクランプ

### 足上げブースト (tweak_foot_lift.py)

kimodo の end-effector keyframe constraint を使って足のリフト量を増やす:

1. 既存 G1 NPZ から `foot_contacts` でスイング区間を検出
2. ピークフレームの `local_rot_mats` を取得
3. 膝関節 (G1 joint 4/11) の axis-angle X成分に boost を加算 (default +0.4 rad)
4. `left-foot` / `right-foot` constraint として `constraints.json` に書き出し
5. kimodo `--input_folder` + separated CFG で再生成 (constraint が足位置を拘束)
6. g1_to_k1.py で K1 にリターゲット

### 既知の制限

- **肘の向き**: K1 の肩が 2DoF (pitch+roll) のため、roll オフセット後の上腕 twist を制御できない。肘が若干内向きになる。
- **腕のモーション**: 生成モーションの腕が不自然になりがち。AMP 学習用には `fix_arms.py` で腕固定推奨。
- **横移動の足クロス**: kimodo が横移動で足をクロスさせる傾向がある。自然な横ステップは少数 (~3/48)。
- **超信地旋回の足上げ**: 素の kimodo 生成ではシャッフル気味 (足リフト ~3cm)。`tweak_foot_lift.py` で改善 (→ 8-13cm)。

## AMP 学習での使い方

```
motions_k1/k1_fixed/  ← このディレクトリの CSV を参照データとして使用
```

K1 qpos CSV フォーマット: 29列 = `[root_xyz(3), root_quat_wxyz(4), joint_angles(22)]`

```python
# タスク報酬: 速度コマンド追従
r_task = exp(-||v_actual - v_command||^2)

# スタイル報酬: AMP ディスクリミネータ (全クリップから学習)
r_style = discriminator(state_transition)

# 総報酬
r = w_task * r_task + w_style * r_style
```

速度コマンドの範囲設定には `motion_stats.json` を参照:
- 速度: 0 ~ 1.5 m/s
- 方向: 全方位 (-180° ~ +180°)
- ヨーレート: 最大 ~180°/s
