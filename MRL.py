from typing import List

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from models.vit import NestedPatchProj, ViTFixedPosNestedDualPatchNorm

'''
Loss function for Matryoshka Representation Learning 
'''

class Matryoshka_CE_Loss(nn.Module):
	def __init__(self, relative_importance: List[float]=None, **kwargs):
		super(Matryoshka_CE_Loss, self).__init__()
		self.criterion = nn.CrossEntropyLoss(**kwargs)
		# relative importance shape: [G]
		self.relative_importance = relative_importance

	def forward(self, output, target):
		# output shape: [G granularities, N batch size, C number of classes]
		# target shape: [N batch size]

		# Calculate losses for each output and stack them. This is still O(N)
		losses = torch.stack([self.criterion(output_i, target) for output_i in output])
		
		# Set relative_importance to 1 if not specified
		if self.relative_importance is None:
			rel_importance = torch.ones_like(losses)
		else:
			rel_importance = torch.tensor(self.relative_importance, device=losses.device, dtype=losses.dtype)
		
		# Apply relative importance weights
		weighted_losses = rel_importance * losses
		return weighted_losses.sum()


class MRL_Linear_Layer(nn.Module):
	def __init__(self, nesting_list: List, num_classes=1000, **kwargs):
		super(MRL_Linear_Layer, self).__init__()
		self.nesting_list = nesting_list
		self.num_classes = num_classes # Number of classes for classification
		
		for i, num_feat in enumerate(self.nesting_list):
			setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))	

	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			nesting_logits += (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

		return nesting_logits


class MRL_Entire_Transformer(nn.Module):
	def __init__(self, nesting_list: List, vit_class, vit_kwargs):
		"""
		Matryoshka Representation Learning for entire ViT models.
		
		Args:
			nesting_list: List of embedding dimensions to use (e.g., [64, 128, 256, 512])
			vit_class: The ViT class to instantiate (e.g., ViTFixedPosDualPatchNorm)
			vit_kwargs: Dictionary of kwargs to pass to the ViT constructor (must include 'dim' key)
		"""
		super(MRL_Entire_Transformer, self).__init__()
		self.nesting_list = nesting_list
		self.vit_class = vit_class
		self.vit_kwargs = vit_kwargs
		
		# Create separate models for each dimension
		for i, dim in enumerate(self.nesting_list):
			kwargs_copy = vit_kwargs.copy()
			kwargs_copy['dim'] = dim
			setattr(self, f"nesting_vit_{i}", vit_class(**kwargs_copy))
	
	def forward(self, x):
		nesting_logits = ()
		
		# Run each model independently
		for i, dim in enumerate(self.nesting_list):
			vit_model = getattr(self, f"nesting_vit_{i}")
			logits = vit_model(x)
			nesting_logits += (logits,)
		
		return nesting_logits
	
	def get_vit_model(self, dim_index):
		"""
		Extract a single ViT model for standalone inference.
		
		Args:
			dim_index: Index of the model to extract (0 to len(nesting_list)-1)
		
		Returns:
			The ViT model at the specified index
		
		Example:
			# Extract the largest model (last in the list)
			largest_model = mrl_model.get_vit_model(-1)
			# Extract the smallest model
			smallest_model = mrl_model.get_vit_model(0)
			# Use for inference
			output = largest_model(input_image)
		"""
		if dim_index < 0:
			dim_index = len(self.nesting_list) + dim_index
		
		if dim_index < 0 or dim_index >= len(self.nesting_list):
			raise IndexError(f"dim_index {dim_index} out of range for nesting_list of length {len(self.nesting_list)}")
		
		return getattr(self, f"nesting_vit_{dim_index}")
	
	def save_vit_model(self, dim_index, save_path):
		"""
		Save a single ViT model to disk for standalone use.
		
		Args:
			dim_index: Index of the model to save (0 to len(nesting_list)-1)
			save_path: Path where to save the model
		
		Example:
			# Save the largest model
			mrl_model.save_vit_model(-1, 'vit_dim512.pth')
		"""
		vit_model = self.get_vit_model(dim_index)
		torch.save({
			'model_state_dict': vit_model.state_dict(),
			'dim': self.nesting_list[dim_index],
			'vit_class': self.vit_class.__name__,
			'vit_kwargs': self.vit_kwargs
		}, save_path)
		print(f"Saved ViT model (dim={self.nesting_list[dim_index]}) to {save_path}")


class MRL_Entire_Transformer_NestedProj(nn.Module):
	def __init__(self, nesting_list: List, image_size, patch_size, num_classes, depth, heads, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
		super(MRL_Entire_Transformer_NestedProj, self).__init__()
		self.nesting_list = nesting_list
		self.image_size = image_size
		self.patch_size = patch_size
		self.num_classes = num_classes
		self.depth = depth
		self.heads = heads
		self.pool = pool
		self.channels = channels
		self.dropout = dropout
		self.emb_dropout = emb_dropout

		if isinstance(image_size, tuple):
			image_height, image_width = image_size
		else:
			image_height = image_width = image_size

		if isinstance(patch_size, tuple):
			patch_height, patch_width = patch_size
		else:
			patch_height = patch_width = patch_size

		patch_dim = channels * patch_height * patch_width
		max_dim = max(nesting_list)
		self.shared_nested_proj = NestedPatchProj(patch_dim, max_dim)

		for i, dim in enumerate(self.nesting_list):
			vit_model = ViTFixedPosNestedDualPatchNorm(
				image_size=image_size,
				patch_size=patch_size,
				num_classes=num_classes,
				dim=dim,
				depth=depth,
				heads=heads,
				pool=pool,
				channels=channels,
				dropout=dropout,
				emb_dropout=emb_dropout,
				nested_proj=self.shared_nested_proj,
				max_dim=max_dim,
			)
			setattr(self, f"nesting_vit_{i}", vit_model)
	
	def forward(self, x):
		nesting_logits = ()
		for i, dim in enumerate(self.nesting_list):
			vit_model = getattr(self, f"nesting_vit_{i}")
			logits = vit_model(x)
			nesting_logits += (logits,)
		return nesting_logits
	
	def get_vit_model(self, dim_index):
		if dim_index < 0:
			dim_index = len(self.nesting_list) + dim_index
		if dim_index < 0 or dim_index >= len(self.nesting_list):
			raise IndexError(f"dim_index {dim_index} out of range for nesting_list of length {len(self.nesting_list)}")
		return getattr(self, f"nesting_vit_{dim_index}")
	
	def save_vit_model(self, dim_index, save_path):
		vit_model = self.get_vit_model(dim_index)
		vit_kwargs = {
			'image_size': self.image_size,
			'patch_size': self.patch_size,
			'num_classes': self.num_classes,
			'depth': self.depth,
			'heads': self.heads,
			'pool': self.pool,
			'channels': self.channels,
			'dropout': self.dropout,
			'emb_dropout': self.emb_dropout,
		}
		torch.save({
			'model_state_dict': vit_model.state_dict(),
			'dim': self.nesting_list[dim_index],
			'vit_class': 'ViTFixedPosNestedDualPatchNorm',
			'vit_kwargs': vit_kwargs,
		}, save_path)
		print(f"Saved nested-projection ViT model (dim={self.nesting_list[dim_index]}) to {save_path}")
