# Guide to the HPT Model

## Step 1 - Registering a New Embodiment

For each new embodiment, navigate to [`./external/rldb/rldb/utils.py`](./external/rldb/rldb/utils.py).

In the `Embodiment` Enum:

```python
class EMBODIMENT(Enum):
    EVE_RIGHT_ARM = 0
    EVE_LEFT_ARM = 1
    EVE_BIMANUAL = 2
    ARIA_RIGHT_ARM = 3
    ARIA_LEFT_ARM = 4
    ARIA_BIMANUAL = 5
    MY_NEW_EMBODIMENT = 6 # Assign an arbitrary integer ID
```

---

## Step 2 - Modify Train YAML

Navigate to [`./egomimic/hydra_configs/train.yaml`](./egomimic/hydra_configs/train.yaml). Replace the `data_schematic` with the relevant keys for your embodiment. The current `data_schematic` looks like:

```yaml
data_schematic: # Dynamically fill in these shapes from the dataset
  _target_: rldb.utils.DataSchematic
  schematic_dict:
    eve_right_arm:
      front_img_1: # Batch key
        key_type: camera_keys # Key type
        lerobot_key: observations.images.front_img_1 # Dataset key
      right_wrist_img:
        key_type: camera_keys
        lerobot_key: observations.images.right_wrist_img
      joint_positions:
        key_type: proprio_keys
        lerobot_key: observations.state.joint_positions
      actions_joints:
        key_type: action_keys
        lerobot_key: actions_joints
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
    aria_right_arm:
      front_img_1:
        key_type: camera_keys
        lerobot_key: observations.images.front_img_1
      ee_pose:
        key_type: proprio_keys
        lerobot_key: observations.state.ee_pose
      actions_cartesian:
        key_type: action_keys
        lerobot_key: actions_cartesian
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
  viz_img_key: 
    eve_right_arm:
      front_img_1
    aria_right_arm:
      front_img_1
```

Modify this to your new embodiment:

```yaml
data_schematic: 
  _target_: rldb.utils.DataSchematic
  schematic_dict:
    my_new_embodiment:
      front_img_1: # Batch key
        key_type: camera_keys # Key type
        lerobot_key: observations.images.front_img_1 # Dataset key
      right_wrist_img:
        key_type: camera_keys
        lerobot_key: observations.images.right_wrist_img
      joint_positions:
        key_type: proprio_keys
        lerobot_key: observations.state.joint_positions
      actions_joints:
        key_type: action_keys
        lerobot_key: actions_joints
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
    aria_right_arm:
      front_img_1:
        key_type: camera_keys
        lerobot_key: observations.images.front_img_1
      ee_pose:
        key_type: proprio_keys
        lerobot_key: observations.state.ee_pose
      actions_cartesian:
        key_type: action_keys
        lerobot_key: actions_cartesian
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
  viz_img_key: 
    my_new_embodiment:
      front_img_1
    aria_right_arm:
      front_img_1
```

If your embodiment has additional observations, add them under `my_new_embodiment`:

```yaml
      my_new_obs: # Name of new observation
        key_type: key_type # Replace with "image_keys" or "proprio_keys" depending on type
        lerobot_key: my.new_key.stored.in.dataset # Stored dataset key
```

For action spaces, follow the same structure, but set `key_type: action_keys`.

---

## Step 3 - Modify Your Model YAML

Navigate to [`./egomimic/hydra_configs/model/hpt_cotrain.yaml`](./egomimic/hydra_configs/model/hpt_cotrain.yaml) or an alternate HPT YAML file. Modify the `domains`:

```yaml
domains: ["my_new_embodiment", "aria_right_arm"] # List of domains
```

Modify `stem_specs` to add new observation modalities:

```yaml
      my_new_obs_1:
        _target_: egomimic.models.hpt_nets.MLPPolicyStem
        input_dim: 512 # ResNet output feature dim
        output_dim: 512
        widths: [512]
        specs:
          random_horizon_masking: false
          cross_attn:
            crossattn_latent: 16
            crossattn_heads: 8
            crossattn_dim_head: 64
            crossattn_modality_dropout: 0.1
            modality_embed_dim: 512
      state_my_new_obs_2:
        _target_: egomimic.models.hpt_nets.MLPPolicyStem
        input_dim: 10 # Input dimension
        output_dim: 512 # HPT embed_dim
        widths: [512]
        specs:
          random_horizon_masking: false
          cross_attn:
            crossattn_latent: 16
            crossattn_heads: 8
            crossattn_dim_head: 64
            crossattn_modality_dropout: 0.1
            modality_embed_dim: 512
```

Add the required image encoder under `encoder_specs`:

```yaml
    my_new_obs_1:
      _target_: egomimic.models.hpt_nets.ResNet
      output_dim: 512
      num_of_copy: 1
```

---

## Step 4 (Optional) - Flexible Action Head System

Define action heads in `head_specs`:

```yaml
 head_specs:
    my_new_embodiment: 
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 7 # Default action head output dim
      widths: [256, 512]
      tanh_end: false
      dropout: true    
   aria_right_arm: 
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 3
      widths: [256, 512]
      tanh_end: false
      dropout: true    
    shared:
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 3
      widths: [256, 512]
      tanh_end: false
      dropout: true
    my_new_embodiment_actions_auxiliary:
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 10 # Auxiliary head output dim
      widths: [256, 512]
      tanh_end: false
      dropout: true
```

Specify shared and auxiliary action keys under `robomimic_model`:

```yaml
  shared_ac_key: "actions_shared" # Key for shared actions
  auxiliary_ac_keys:
    my_new_embodiment: ["actions_auxiliary"] # Auxiliary action keys
```

**Note:** These action keys must match their `batch_key` inside the `data_schematic` in `train.yaml`.

