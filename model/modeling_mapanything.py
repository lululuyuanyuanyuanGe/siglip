#/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from .mapanything.models.mapanything.model import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput


class MapAnythingWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.map_anything_model = MapAnything.from_pretrained(config.map_anything_model_name_or_path)

    def forward(self, pixel_values, intrinsics):
        # Prepare the input format that MapAnything's internal methods expect.
        views = [{"img": pixel_values, "data_norm_type": ["dinov2_vitl14_reg"], "intrinsics": intrinsics}]

        # Call the internal encoder of the MapAnything model.
        all_encoder_features = self.map_anything_model._encode_n_views(views)

        # Prepare the input for the multi-view transformer.
        info_sharing_input = MultiViewTransformerInput(features=all_encoder_features)

        # Call the multi-view transformer module directly.
        final_features, _ = self.map_anything_model.info_sharing(info_sharing_input)

        # Extract the features we want.
        geometric_features = final_features.features[0]

        return geometric_features
