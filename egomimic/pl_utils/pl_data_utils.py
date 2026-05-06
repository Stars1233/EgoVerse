import logging
import random
from typing import Literal

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from termcolor import cprint
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class RLDBModule(LightningDataModule):
    """
    Deprecated and is not supported by trainHydra.py
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_kwargs,
        valid_dataloader_kwargs,
    ):
        cprint(
            "RLDBModule is deprecated and is not supported by trainHydra.py. Use MultiDataModuleWrapper instead",
            "red",
        )

        super().__init__()
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.valid_dataloader_kwargs = valid_dataloader_kwargs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, **self.train_dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, shuffle=False, **self.valid_dataloader_kwargs
        )


class MultiDataModuleWrapper(LightningDataModule):
    """
    New functionality for dictionary based multi embodiment loading using CombinedLoader.

    Uses hydra to instantiate DataLoader objects and then wraps them in a combined loader
    """

    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict,
        train_dataloader_params: dict,
        valid_dataloader_params: dict,
        collate_max_length=128,
        model_name="google/paligemma-3b-mix-224",
        sampling_mode: Literal["first", "random"] = "random",
        annotation_key=None,
        use_tokenizer=False,
        default_prompt="",
        proprio_keys: list[str] | None = None,
        discrete_state_input: bool = False,
        state_num_bins: int = 256,
        proprio_mode: Literal["original", "specific", "language"] = "original",
    ):
        """
        Args:
            train_datasets: dictionary of train datasets
            valid_datasets: dictionary of valid datasets
            train_dataloader_params: dictionary of train dataloader parameters
            valid_dataloader_params: dictionary of valid dataloader parameters
            model_name: name of the model to use for the tokenizer
            sampling_mode: "first" to sample the first prompt from the list of prompts, "random" to sample a random prompt from the list of prompts
            annotation_key: key of the annotation to use for the collate function
            use_tokenizer: whether to use the tokenizer to tokenize the prompts
            default_prompt: default prompt to use if the annotation key is not found
            collate_max_length: maximum length of the tokenized prompts
            proprio_keys: union of per-sample state keys to concat for the
                ``discrete_state_input`` path. Keys missing from a given
                embodiment's batch are skipped.
            discrete_state_input: if True, splice discretized proprio into the
                prompt as ``Task: ..., State: <bins>;\\nAction: `` (pi0.5 style).
            state_num_bins: number of bins to discretize each state dim into.
            proprio_mode: "original" for the upstream pi0.5 template
                (``Task: ..., State: ...``), "specific" to additionally splice
                ``Embodiment:`` and ``Control mode:`` descriptors into the
                prompt, or "language" to tokenize only the raw prompt with no
                ``State:`` block (pre-pi0.5 behaviour; takes precedence over
                ``discrete_state_input``).
        """
        super().__init__()
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        if use_tokenizer:
            self.collate_fn = build_tokenized_collate(
                max_length=collate_max_length,
                model_name=model_name,
                sampling_mode=sampling_mode,
                annotation_key=annotation_key,
                default_prompt=default_prompt,
                proprio_keys=proprio_keys,
                discrete_state_input=discrete_state_input,
                state_num_bins=state_num_bins,
                proprio_mode=proprio_mode,
            )
        else:
            self.collate_fn = annotation_collate

    def train_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config train_dataloader_params."
                )
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=True,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")

    def val_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.valid_datasets.items():
            dataset_params = self.valid_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config valid_dataloader_params."
                )
            dataset_params = dict(dataset_params)
            shuffle = dataset_params.pop("shuffle", False)
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=shuffle,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")


class DualDataModuleWrapper(LightningDataModule):
    """
    Same as DataModuleWrapper but there are two train datasets and two valid datasets
    """

    """
    Deprecated and is not supported by trainHydra.py
    """

    def __init__(
        self,
        train_dataset1,
        valid_dataset1,
        train_dataset2,
        valid_dataset2,
        train_dataloader_params,
        valid_dataloader_params,
        collate_max_length=128,
        model_name="google/paligemma-3b-mix-224",
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        cprint(
            "DualDataModuleWrapper is deprecated and is not supported by trainHydra.py. Use MultiDataModuleWrapper instead",
            "red",
        )

        super().__init__()
        self.train_dataset1 = train_dataset1
        self.valid_dataset1 = valid_dataset1
        self.train_dataset2 = train_dataset2
        self.valid_dataset2 = valid_dataset2
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = build_tokenized_collate(
            max_length=collate_max_length,
            model_name=model_name,
        )

    def train_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.train_dataset1,
            collate_fn=self.collate_fn,
            **self.train_dataloader_params,
        )
        new_dataloader2 = DataLoader(
            dataset=self.train_dataset2,
            collate_fn=self.collate_fn,
            **self.train_dataloader_params,
        )
        return [new_dataloader1, new_dataloader2]

    ## to change embodiment sampling freq, just change the batch_size
    def val_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.valid_dataset1,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.valid_dataloader_params,
        )
        new_dataloader2 = DataLoader(
            dataset=self.valid_dataset2,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.valid_dataloader_params,
        )
        return [new_dataloader1, new_dataloader2]

    # def val_dataloader(self):
    #     new_dataloader1 = DataLoader(dataset=self.valid_dataset1, **self.valid_dataloader_params)
    #     new_dataloader2 = DataLoader(dataset=self.valid_dataset2, **self.valid_dataloader_params)
    #     return [new_dataloader1, new_dataloader2]


class DataModuleWrapper(LightningDataModule):
    """
    Wrapper around a LightningDataModule that allows for the data loader to be refreshed
    constantly.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_params,
        valid_dataloader_params,
        collate_max_length=128,
        model_name="google/paligemma-3b-mix-224",
        sampling_mode: Literal["first", "random"] = "random",
        annotation_key=None,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = build_tokenized_collate(
            max_length=collate_max_length,
            model_name=model_name,
            sampling_mode=sampling_mode,
            annotation_key=annotation_key,
        )

    def train_dataloader(self):
        new_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            **self.train_dataloader_params,
        )
        return new_dataloader

    def val_dataloader_1(self):
        new_dataloader = DataLoader(
            dataset=self.valid_dataset,
            collate_fn=self.collate_fn,
            **self.valid_dataloader_params,
        )
        return new_dataloader


def _extract_list_keys(batch):
    """Pop all list-valued keys from *batch* samples and return them separately.

    This lets ``default_collate`` handle tensors / numbers while variable-length
    annotation lists (``key_type == "annotation_keys"``) are preserved as
    ``list[list[str]]``.
    """
    list_keys = {k for k in batch[0] if isinstance(batch[0][k], list)}
    return {k: [sample.pop(k) for sample in batch] for k in list_keys}


def _extract_keys(batch, keys):
    return {k: [sample.pop(k) for sample in batch] for k in keys}


def annotation_collate(batch):
    """Collate that preserves variable-length list-valued keys (e.g. annotation_keys)."""
    extracted = _extract_list_keys(batch)
    collated = default_collate(batch)
    collated.update(extracted)
    return collated


def build_tokenized_collate(
    max_length=128,
    model_name="google/paligemma-3b-mix-224",
    sampling_mode: Literal["first", "random"] = "random",
    annotation_key="annotations",
    default_prompt="",
    proprio_keys: list[str] | None = None,
    discrete_state_input: bool = False,
    state_num_bins: int = 256,
    proprio_mode: Literal["original", "specific", "language"] = "original",
):
    """Return a collate_fn closure that tokenizes the annotations field.

    If ``discrete_state_input=True``, the per-sample proprio listed in
    ``proprio_keys`` is concatenated, clipped to ``[-1, 1]``, discretized into
    ``state_num_bins`` bins, and spliced into the prompt before tokenization
    (mirrors openpi's pi05 PaligemmaTokenizer convention). State is assumed to
    already be normalized to ``[-1, 1]`` upstream; values outside that range
    are clipped.

    The text template depends on ``proprio_mode``:
      - ``"original"`` (default, openpi-compatible):
          ``"Task: {prompt}, State: {b0} {b1} ...;\\nAction: "``
      - ``"specific"`` (adds embodiment + control-mode descriptors):
          ``"Task: {prompt}, Embodiment: {emb}, Control mode: {mode}, State: {b0} {b1} ...;\\nAction: "``
        where the control-mode descriptor is
        ``"cam frame xyzypr gripper per arm"`` for robot embodiments and
        ``"cam frame xyzypr per arm"`` for Aria (no gripper).
      - ``"language"`` (no proprio in prompt; matches the pre-pi0.5 behaviour):
          the raw ``prompt`` is tokenized as-is, with no ``State:`` block and
          no ``Action:`` anchor. Useful for ablations or pi0-style models that
          take state through ``state_proj`` instead of the language stream.
          Effective regardless of ``discrete_state_input``.
    """
    if proprio_mode not in ("original", "specific", "language"):
        raise ValueError(
            "proprio_mode must be 'original', 'specific', or 'language', "
            f"got {proprio_mode!r}"
        )
    from egomimic.rldb.embodiment.embodiment import get_embodiment

    tok = AutoTokenizer.from_pretrained(model_name)
    state_bin_edges = np.linspace(-1.0, 1.0, state_num_bins + 1)[:-1]
    # Default to the canonical concat key produced by the embodiment transform_list
    # (ConcatKeys with delete_old_keys=True removes the per-arm zarr keys).
    if proprio_keys is None:
        proprio_keys = ["observations.state.ee_pose"]
    else:
        proprio_keys = list(proprio_keys)

    def _embodiment_name(sample):
        eid = sample.get("embodiment")
        if eid is None:
            return None
        if isinstance(eid, torch.Tensor):
            eid = int(eid.item())
        elif isinstance(eid, np.ndarray):
            eid = int(eid.item())
        else:
            eid = int(eid)
        name = get_embodiment(eid)
        if name is None:
            return None
        return name.lower().replace("_", " ")

    def _control_mode_for(emb_name):
        if emb_name is not None and "aria" in emb_name:
            return "cam frame xyzypr per arm"
        return "cam frame xyzypr gripper per arm"

    def _discretize_sample_state(sample):
        if not proprio_keys:
            return None
        parts = []
        for k in proprio_keys:
            if k not in sample:
                continue
            v = sample[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            else:
                v = np.asarray(v)
            v = np.asarray(v, dtype=np.float32)
            # Use the most recent timestep if proprio carries a time axis.
            while v.ndim > 1:
                v = v[-1]
            parts.append(v.reshape(-1))
        if not parts:
            return None
        state = np.concatenate(parts, axis=-1)
        state = np.clip(state, -1.0, 1.0)
        bins = np.digitize(state, bins=state_bin_edges) - 1
        return " ".join(map(str, bins.tolist()))

    def _collate(batch):
        if annotation_key is None:
            annotation = {}
            prompts = [default_prompt] * len(batch)
        else:
            if annotation_key not in batch[0]:
                raise KeyError(f"Annotation key {annotation_key} not found in batch")
            annotation = _extract_keys(batch, [annotation_key])
            prompts = []
            for sample in annotation[annotation_key]:
                if len(sample) == 0:
                    sampled_prompt = default_prompt
                elif sampling_mode == "random":
                    sampled_prompt = sample[random.randint(0, len(sample) - 1)]
                elif sampling_mode == "first":
                    sampled_prompt = sample[0]
                prompts.append(sampled_prompt)

        if discrete_state_input and proprio_mode != "language":
            spliced = []
            for i, prompt in enumerate(prompts):
                state_str = _discretize_sample_state(batch[i])
                if state_str is None:
                    spliced.append(prompt)
                    continue
                if proprio_mode == "original":
                    spliced.append(f"Task: {prompt}, State: {state_str};\nAction: ")
                else:  # "specific"
                    emb_name = _embodiment_name(batch[i])
                    control_mode = _control_mode_for(emb_name)
                    emb_part = f", Embodiment: {emb_name}" if emb_name else ""
                    spliced.append(
                        f"Task: {prompt}{emb_part}, Control mode: {control_mode}, "
                        f"State: {state_str};\nAction: "
                    )
            prompts = spliced

        list_keys = _extract_list_keys(batch)

        enc = tok(
            prompts,
            padding="max_length" if max_length is not None else "longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        collated = default_collate(batch)
        collated["sampled_prompt"] = prompts
        collated.update(list_keys)
        attention_mask = enc["attention_mask"].bool()
        token_loss_mask = attention_mask.clone()
        token_loss_mask[:, -1] = False

        collated["tokenized_prompt"] = enc["input_ids"].requires_grad_(False)
        collated["tokenized_mask"] = attention_mask.requires_grad_(False)
        collated["token_loss_mask"] = token_loss_mask.requires_grad_(False)
        collated["token_ar_mask"] = attention_mask.clone().requires_grad_(False)
        return collated

    return _collate
