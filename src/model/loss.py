"""
Skip-gram Loss Module.

This module implements the skip-gram loss with negative sampling for
training GraphSAGE embeddings. The objective is based on Word2Vec:
- Positive pairs (target, context) should have high dot-product similarity
- Negative pairs (target, negative) should have low dot-product similarity

Loss Function:
    L = -log(σ(e_target · e_context)) - Σ log(σ(-e_target · e_negative))

Where:
    - σ is the sigmoid function
    - · is the dot product
    - The first term maximizes similarity for positive pairs
    - The second term minimizes similarity for negative pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SkipGramLoss(nn.Module):
    """
    Skip-gram loss with negative sampling.

    This loss function trains embeddings so that:
    1. Co-occurring nodes (from random walks) have similar embeddings
    2. Random non-co-occurring nodes have dissimilar embeddings

    The loss uses log-sigmoid for numerical stability and supports
    any number of negative samples per positive pair.

    IMPORTANT: When using L2-normalized embeddings, dot products are bounded
    to [-1, 1], which limits sigmoid outputs to [0.27, 0.73]. The scale_factor
    parameter addresses this by scaling scores before sigmoid, allowing the
    model to achieve higher positive similarity and lower negative similarity.

    Example:
        >>> import torch
        >>> from src.model import SkipGramLoss
        >>>
        >>> # scale_factor=5.0 maps dot product of 1.0 to sigmoid(5.0)=0.993
        >>> loss_fn = SkipGramLoss(scale_factor=5.0)
        >>>
        >>> # Embeddings for all nodes
        >>> embeddings = torch.randn(100, 64)
        >>> embeddings = F.normalize(embeddings, dim=1)
        >>>
        >>> # Batch of positive pairs
        >>> targets = torch.tensor([0, 1, 2, 3])
        >>> contexts = torch.tensor([5, 6, 7, 8])
        >>>
        >>> # Negative samples for each positive
        >>> negatives = torch.tensor([[10, 11], [12, 13], [14, 15], [16, 17]])
        >>>
        >>> loss = loss_fn(embeddings, targets, contexts, negatives)
    """

    def __init__(self, scale_factor: float = 5.0):
        """
        Initialize skip-gram loss.

        Args:
            scale_factor: Multiplier for dot product scores before sigmoid.
                         For L2-normalized embeddings, this is critical:
                         - scale=1.0: sigmoid bounded to [0.27, 0.73] (can't learn!)
                         - scale=5.0: sigmoid can reach [0.007, 0.993] (recommended)
                         - scale=10.0: sigmoid can reach [0.00005, 0.99995] (may be unstable)
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute skip-gram loss.

        Args:
            embeddings: All node embeddings [num_nodes, dim]
            targets: Target node indices [batch_size]
            contexts: Context node indices [batch_size]
            negatives: Negative node indices [batch_size, num_negatives]

        Returns:
            loss: Scalar loss value
        """
        # Get embeddings for each role
        # target_emb: [batch_size, dim]
        target_emb = embeddings[targets]

        # context_emb: [batch_size, dim]
        context_emb = embeddings[contexts]

        # negative_emb: [batch_size, num_negatives, dim]
        negative_emb = embeddings[negatives]

        # === Positive scores ===
        # Dot product between target and context
        # [batch_size, dim] * [batch_size, dim] -> sum -> [batch_size]
        pos_scores = (target_emb * context_emb).sum(dim=1)

        # === Negative scores ===
        # Dot product between target and each negative
        # target_emb: [batch_size, dim] -> [batch_size, dim, 1]
        # negative_emb: [batch_size, num_negatives, dim]
        # bmm result: [batch_size, num_negatives, 1] -> squeeze -> [batch_size, num_negatives]
        neg_scores = torch.bmm(
            negative_emb,
            target_emb.unsqueeze(2)
        ).squeeze(2)

        # === Apply scale factor ===
        # Critical for L2-normalized embeddings!
        # Without scaling: dot products in [-1, 1] -> sigmoid in [0.27, 0.73]
        # With scale=5.0: scaled scores in [-5, 5] -> sigmoid can reach [0.007, 0.993]
        pos_scores = pos_scores * self.scale_factor
        neg_scores = neg_scores * self.scale_factor

        # === Skip-gram loss ===
        # Positive term: maximize log(sigmoid(pos_score))
        # Negative term: maximize log(sigmoid(-neg_score))
        #
        # Using log_sigmoid for numerical stability
        # log_sigmoid(x) = -softplus(-x) = -log(1 + exp(-x))
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()

        total_loss = pos_loss + neg_loss

        return total_loss

    def forward_with_details(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss with detailed breakdown.

        Useful for debugging and monitoring training.

        Args:
            embeddings: All node embeddings [num_nodes, dim]
            targets: Target node indices [batch_size]
            contexts: Context node indices [batch_size]
            negatives: Negative node indices [batch_size, num_negatives]

        Returns:
            Tuple of (total_loss, details_dict)
        """
        target_emb = embeddings[targets]
        context_emb = embeddings[contexts]
        negative_emb = embeddings[negatives]

        # Raw dot product scores (before scaling)
        pos_scores_raw = (target_emb * context_emb).sum(dim=1)
        neg_scores_raw = torch.bmm(
            negative_emb,
            target_emb.unsqueeze(2)
        ).squeeze(2)

        # Apply scale factor for loss computation
        pos_scores = pos_scores_raw * self.scale_factor
        neg_scores = neg_scores_raw * self.scale_factor

        # Losses (using scaled scores)
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        total_loss = pos_loss + neg_loss

        # Probabilities for monitoring (using scaled scores)
        pos_prob = torch.sigmoid(pos_scores).mean()
        neg_prob = torch.sigmoid(neg_scores).mean()

        details = {
            'total_loss': total_loss.item(),
            'pos_loss': pos_loss.item(),
            'neg_loss': neg_loss.item(),
            'pos_score_mean': pos_scores_raw.mean().item(),  # Raw dot product for interpretability
            'neg_score_mean': neg_scores_raw.mean().item(),  # Raw dot product for interpretability
            'pos_prob': pos_prob.item(),  # Should be high (~1) - now achievable!
            'neg_prob': neg_prob.item(),  # Should be low (~0) - now achievable!
        }

        return total_loss, details


class MarginRankingLoss(nn.Module):
    """
    Margin-based ranking loss for structural embeddings.

    Alternative to skip-gram loss. Uses a margin to separate
    positive and negative scores.

    Loss = max(0, margin - pos_score + neg_score)

    This can be more stable in some cases but typically
    skip-gram loss works better.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize margin ranking loss.

        Args:
            margin: Minimum desired margin between pos and neg scores
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.

        Args:
            embeddings: All node embeddings [num_nodes, dim]
            targets: Target node indices [batch_size]
            contexts: Context node indices [batch_size]
            negatives: Negative node indices [batch_size, num_negatives]

        Returns:
            loss: Scalar loss value
        """
        target_emb = embeddings[targets]
        context_emb = embeddings[contexts]
        negative_emb = embeddings[negatives]

        # Positive scores
        pos_scores = (target_emb * context_emb).sum(dim=1, keepdim=True)

        # Negative scores
        neg_scores = torch.bmm(
            negative_emb,
            target_emb.unsqueeze(2)
        ).squeeze(2)

        # Margin loss: we want pos_score > neg_score + margin
        # Loss = max(0, margin - pos_score + neg_score)
        loss = F.relu(self.margin - pos_scores + neg_scores)

        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss.

    This is a softmax-based contrastive loss that treats the
    positive pair against all negatives as a classification problem.

    Loss = -log(exp(pos_score / τ) / (exp(pos_score / τ) + Σ exp(neg_score / τ)))

    Where τ (tau) is a temperature parameter.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature for softmax (lower = sharper)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            embeddings: All node embeddings [num_nodes, dim]
            targets: Target node indices [batch_size]
            contexts: Context node indices [batch_size]
            negatives: Negative node indices [batch_size, num_negatives]

        Returns:
            loss: Scalar loss value
        """
        target_emb = embeddings[targets]
        context_emb = embeddings[contexts]
        negative_emb = embeddings[negatives]

        # Positive scores: [batch_size, 1]
        pos_scores = (target_emb * context_emb).sum(dim=1, keepdim=True) / self.temperature

        # Negative scores: [batch_size, num_negatives]
        neg_scores = torch.bmm(
            negative_emb,
            target_emb.unsqueeze(2)
        ).squeeze(2) / self.temperature

        # Concatenate: [batch_size, 1 + num_negatives]
        # Position 0 is the positive
        logits = torch.cat([pos_scores, neg_scores], dim=1)

        # Labels: always 0 (the positive is always first)
        labels = torch.zeros(targets.size(0), dtype=torch.long, device=targets.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


def create_loss(loss_type: str = 'skipgram', **kwargs) -> nn.Module:
    """
    Create loss function by name.

    Args:
        loss_type: Type of loss ('skipgram', 'margin', 'infonce')
        **kwargs: Additional arguments for specific loss types
            - scale_factor (float): For skipgram loss, scales dot products before sigmoid.
                                   Default 5.0, critical for L2-normalized embeddings.
            - margin (float): For margin loss, default 1.0
            - temperature (float): For infonce loss, default 0.07

    Returns:
        Loss module
    """
    if loss_type == 'skipgram':
        return SkipGramLoss(scale_factor=kwargs.get('scale_factor', 5.0))
    elif loss_type == 'margin':
        return MarginRankingLoss(margin=kwargs.get('margin', 1.0))
    elif loss_type == 'infonce':
        return InfoNCELoss(temperature=kwargs.get('temperature', 0.07))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
