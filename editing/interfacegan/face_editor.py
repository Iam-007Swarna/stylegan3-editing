from typing import Optional, Tuple

import numpy as np
import torch

from configs.paths_config import interfacegan_aligned_edit_paths, interfacegan_unaligned_edit_paths
from models.stylegan3.model import GeneratorType
from models.stylegan3.networks_stylegan3 import Generator
from utils.common import tensor2im, generate_random_transform


class FaceEditor:

    def __init__(self, stylegan_generator: Generator, generator_type=GeneratorType.ALIGNED):
        self.generator = stylegan_generator
        if generator_type == GeneratorType.ALIGNED:
            paths = interfacegan_aligned_edit_paths
        else:
            paths = interfacegan_unaligned_edit_paths

        self.interfacegan_directions = {
            '5_0_Clock_Shadow': torch.from_numpy(np.load(paths['5_0_Clock_Shadow'])).cuda(),
            'Age': torch.from_numpy(np.load(paths['Age'])).cuda(),
            'Arched_Eyebrows': torch.from_numpy(np.load(paths['Arched_Eyebrows'])).cuda(),
            'Attractive': torch.from_numpy(np.load(paths['Attractive'])).cuda(),
            'Bags_Under_Eyes': torch.from_numpy(np.load(paths['Bags_Under_Eyes'])).cuda(),
            'Bald': torch.from_numpy(np.load(paths['Bald'])).cuda(),
            'Bangs': torch.from_numpy(np.load(paths['Bangs'])).cuda(),
            'Big_Lips': torch.from_numpy(np.load(paths['Big_Lips'])).cuda(),
            'Big_Nose': torch.from_numpy(np.load(paths['Big_Nose'])).cuda(),
            'Black_Hair': torch.from_numpy(np.load(paths['Black_Hair'])).cuda(),
            'Blond_Hair': torch.from_numpy(np.load(paths['Blond_Hair'])).cuda(),
            'Blurry': torch.from_numpy(np.load(paths['Blurry'])).cuda(),
            'Brown_Hair': torch.from_numpy(np.load(paths['Brown_Hair'])).cuda(),
            'Bushy_Eyebrows': torch.from_numpy(np.load(paths['Bushy_Eyebrows'])).cuda(),
            'Chubby': torch.from_numpy(np.load(paths['Chubby'])).cuda(),
            'Double_Chin': torch.from_numpy(np.load(paths['Double_Chin'])).cuda(),
            'Eyeglasses': torch.from_numpy(np.load(paths['Eyeglasses'])).cuda(),
            'Goatee': torch.from_numpy(np.load(paths['Goatee'])).cuda(),
            'Gray_Hair': torch.from_numpy(np.load(paths['Gray_Hair'])).cuda(),
            'Heavy_Makeup': torch.from_numpy(np.load(paths['Heavy_Makeup'])).cuda(),
            'High_Cheekbones': torch.from_numpy(np.load(paths['High_Cheekbones'])).cuda(),
            'Male': torch.from_numpy(np.load(paths['Male'])).cuda(),
            'Mouth_Slightly_Open': torch.from_numpy(np.load(paths['Mouth_Slightly_Open'])).cuda(),
            'Mustache': torch.from_numpy(np.load(paths['Mustache'])).cuda(),
            'Narrow_Eyes': torch.from_numpy(np.load(paths['Narrow_Eyes'])).cuda(),
            'No_Beard': torch.from_numpy(np.load(paths['No_Beard'])).cuda(),
            'Oval_Face': torch.from_numpy(np.load(paths['Oval_Face'])).cuda(),
            'Pale_Skin': torch.from_numpy(np.load(paths['Pale_Skin'])).cuda(),
            'Pointy_Nose': torch.from_numpy(np.load(paths['Pointy_Nose'])).cuda(),
            'Pose': torch.from_numpy(np.load(paths['Pose'])).cuda(),
            'Receding_Hairline': torch.from_numpy(np.load(paths['Receding_Hairline'])).cuda(),
            'Rosy_Cheeks': torch.from_numpy(np.load(paths['Rosy_Cheeks'])).cuda(),
            'Sideburns': torch.from_numpy(np.load(paths['Sideburns'])).cuda(),
            'Smiling': torch.from_numpy(np.load(paths['Smiling'])).cuda(),
            'Straight_Hair': torch.from_numpy(np.load(paths['Straight_Hair'])).cuda(),
            'Wavy_Hair': torch.from_numpy(np.load(paths['Wavy_Hair'])).cuda(),
            'Wearing_Earrings': torch.from_numpy(np.load(paths['Wearing_Earrings'])).cuda(),
            'Wearing_Hat': torch.from_numpy(np.load(paths['Wearing_Hat'])).cuda(),
            'Wearing_Lipstick': torch.from_numpy(np.load(paths['Wearing_Lipstick'])).cuda(),
            'Wearing_Necklace': torch.from_numpy(np.load(paths['Wearing_Necklace'])).cuda(),
            'Wearing_Necktie': torch.from_numpy(np.load(paths['Wearing_Necktie'])).cuda(),
            'Young': torch.from_numpy(np.load(paths['Young'])).cuda()
        }

    def edit(self, latents: torch.tensor, direction: str, factor: int = 1, factor_range: Optional[Tuple[int, int]] = None,
             user_transforms: Optional[np.ndarray] = None, apply_user_transformations: Optional[bool] = False):
        edit_latents = []
        edit_images = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + f * direction
                edit_image, user_transforms = self._latents_to_image(edit_latent,
                                                                     apply_user_transformations,
                                                                     user_transforms)
                edit_latents.append(edit_latent)
                edit_images.append(edit_image)
        else:
            edit_latents = latents + factor * direction
            edit_images, _ = self._latents_to_image(edit_latents, apply_user_transformations)
        return edit_images, edit_latents

    def _latents_to_image(self, all_latents: torch.tensor, apply_user_transformations: bool = False,
                          user_transforms: Optional[torch.tensor] = None):
        with torch.no_grad():
            if apply_user_transformations:
                if user_transforms is None:
                    # if no transform provided, generate a random transformation
                    user_transforms = generate_random_transform(translate=0.3, rotate=25)
                # apply the user-specified transformation
                if type(user_transforms) == np.ndarray:
                    user_transforms = torch.from_numpy(user_transforms)
                self.generator.synthesis.input.transform = user_transforms.cuda().float()
            # generate the images
            images = self.generator.synthesis(all_latents, noise_mode='const')
            images = [tensor2im(image) for image in images]
        return images, user_transforms
