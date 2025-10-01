"""
Learning Module

Contains training functionality for triplet loss models.
"""

from .separate_viewpoint import train_model, create_embedding_model, TripletGenerator

__all__ = ['train_model', 'create_embedding_model', 'TripletGenerator']