import json
import os
from abc import abstractmethod

from openai import OpenAI


def micro_seconds_to_frames(micro_seconds: int, fps: int) -> int:
    return micro_seconds / 1000000 * fps


class ScaleToZarrAnnotationConverter:
    def __init__(self, scale_annotation_dir: str):
        self.scale_annotation_dir = scale_annotation_dir

    def convert(self, tid: str) -> dict:
        scale_annotation_path = os.path.join(self.scale_annotation_dir, f"{tid}.json")
        with open(scale_annotation_path, "r") as f:
            annotation_dict = json.load(f)
        return self.scale_to_str_format(annotation_dict)

    @abstractmethod
    def scale_to_str_format(self, annotation_dict: dict) -> dict:
        pass


class LLMConverter(ScaleToZarrAnnotationConverter):
    def __init__(self, scale_annotation_dir: str, prompt_filepath: str):
        super().__init__(scale_annotation_dir)
        self.client = OpenAI()
        self.model = "gpt-5.4"
        self.fps = 30
        with open(prompt_filepath, "r") as f:
            self.prompt_template = f.read()


class PickPlaceLLMConverter(LLMConverter):
    def scale_to_str_format(self, annotation_dict: dict) -> dict:
        annotations = annotation_dict["annotations"]
        zarr_annotations_list = []
        for annotation in annotations:
            arm = annotation["label"].split(" ")[0].lower()
            clips = annotation["clips"]
            for clip in clips:
                timestamp = micro_seconds_to_frames(clip["timestamp"], self.fps)
                duration = micro_seconds_to_frames(clip["duration"], self.fps)
                attributes = clip["attributes"]
                text = clip["text"]
                attr_dict = {}
                for attribute in attributes:
                    attr_dict[attribute["name"]] = attribute["values"][0]
                if attr_dict["Mistake"] == "Yes":
                    continue
                if "Action" not in attr_dict or attr_dict["Action"] == "Adjust":
                    continue
                prompt_dict = attr_dict.copy()
                prompt_dict.pop("Mistake")
                prompt_dict["description"] = text
                prompt_dict["arm"] = arm

                zarr_annotation_str = self.scale_annotation_to_str(prompt_dict)
                start_idx = timestamp
                end_idx = timestamp + duration
                zarr_annotations_list.append((zarr_annotation_str, start_idx, end_idx))
        return zarr_annotations_list

    def scale_annotation_to_str(self, scale_annotation_dict: dict) -> str:
        model_prompt = self.prompt_template + "\n" + json.dumps(scale_annotation_dict)
        response = self.client.responses.create(model=self.model, input=model_prompt)
        return response.output_text


class HardCodedConverter(ScaleToZarrAnnotationConverter):
    def scale_to_str_format(self, annotation: dict) -> dict:
        pass
