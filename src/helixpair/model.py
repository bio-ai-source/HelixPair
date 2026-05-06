from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from helixpair.constants import DEFAULT_OFFSETS, ORIENTATION_TO_INDEX


class TFHierarchyEmbedding(nn.Module):
    def __init__(
        self,
        num_families: int,
        num_subfamilies: int,
        num_paralogs: int,
        num_tfs: int,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.family = nn.Embedding(num_families, embedding_dim)
        self.subfamily = nn.Embedding(num_subfamilies, embedding_dim)
        self.paralog = nn.Embedding(num_paralogs, embedding_dim)
        self.identity = nn.Embedding(num_tfs, embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(
        self,
        family_id: torch.Tensor,
        subfamily_id: torch.Tensor,
        paralog_id: torch.Tensor,
        tf_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.proj(
            torch.cat(
                [
                    self.family(family_id),
                    self.subfamily(subfamily_id),
                    self.paralog(paralog_id),
                    self.identity(tf_id),
                ],
                dim=-1,
            )
        )


class FlatTFEmbedding(nn.Module):
    def __init__(self, num_tfs: int, embedding_dim: int = 32):
        super().__init__()
        self.identity = nn.Embedding(num_tfs, embedding_dim)

    def forward(
        self,
        family_id: torch.Tensor,
        subfamily_id: torch.Tensor,
        paralog_id: torch.Tensor,
        tf_id: torch.Tensor,
    ) -> torch.Tensor:
        del family_id, subfamily_id, paralog_id
        return self.identity(tf_id)


class MonomerEnergyHead(nn.Module):
    def __init__(self, sequence_channels: int = 5, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(sequence_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        offset_tensors: torch.Tensor,
        tf_embedding: torch.Tensor,
        temperature: float = 1.0,
        use_anchor_refinement: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_offsets, channels, width = offset_tensors.shape
        flat = offset_tensors.reshape(batch * num_offsets, channels, width)
        encoded = self.encoder(flat)
        pooled = self.pool(encoded).squeeze(-1).reshape(batch, num_offsets, -1)
        expanded_embedding = tf_embedding.unsqueeze(1).expand(-1, num_offsets, -1)
        energies = self.head(torch.cat([pooled, expanded_embedding], dim=-1)).squeeze(-1)
        if use_anchor_refinement and num_offsets > 1:
            weights = torch.softmax(-energies / max(temperature, 1e-6), dim=-1)
            marginal = -(torch.logsumexp(-energies / max(temperature, 1e-6), dim=-1) * max(temperature, 1e-6))
        else:
            center_index = num_offsets // 2
            weights = torch.zeros_like(energies)
            weights[:, center_index] = 1.0
            marginal = energies[:, center_index]
        return marginal, weights


class GeometryResidualHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 32,
        geometry_dim: int = 19,
        rank: int = 8,
        num_gap_bins: int = 25,
        helical_order: int = 2,
    ):
        super().__init__()
        self.left_proj = nn.Linear(embedding_dim, rank)
        self.right_proj = nn.Linear(embedding_dim, rank)
        self.geometry_proj = nn.Sequential(
            nn.Linear(geometry_dim, rank * 2),
            nn.GELU(),
            nn.Linear(rank * 2, rank),
        )
        distribution_dim = embedding_dim * 2 + rank * 2
        self.distribution_head = nn.Sequential(
            nn.Linear(distribution_dim, distribution_dim),
            nn.GELU(),
            nn.Linear(distribution_dim, distribution_dim // 2),
            nn.GELU(),
        )
        self.gap_classifier = nn.Linear(distribution_dim // 2, num_gap_bins)
        self.orientation_classifier = nn.Linear(distribution_dim // 2, len(ORIENTATION_TO_INDEX))
        self.helical_start = 7
        self.helical_dim = max(0, min(int(geometry_dim) - self.helical_start, 2 * int(helical_order)))
        if self.helical_dim > 0:
            self.helical_head = nn.Sequential(
                nn.Linear(self.helical_dim + embedding_dim, rank),
                nn.GELU(),
                nn.Linear(rank, 1),
            )
            self.helical_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.helical_head = None
            self.helical_scale = None

    def forward(
        self,
        left_embedding: torch.Tensor,
        right_embedding: torch.Tensor,
        geometry_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_rank = self.left_proj(left_embedding)
        right_rank = self.right_proj(right_embedding)
        geometry_rank = self.geometry_proj(geometry_features)
        residual = (left_rank * right_rank * geometry_rank).sum(dim=-1)
        if self.helical_head is not None and geometry_features.shape[-1] >= self.helical_start + self.helical_dim:
            helical_features = geometry_features[:, self.helical_start : self.helical_start + self.helical_dim]
            helical_strength = helical_features.abs().mean(dim=-1)
            helical_residual = self.helical_head(torch.cat([helical_features, left_embedding * right_embedding], dim=-1)).squeeze(-1)
            residual = residual + self.helical_scale * helical_strength * helical_residual
        distribution_hidden = self.distribution_head(torch.cat([left_embedding, right_embedding, left_rank, geometry_rank], dim=-1))
        return residual, self.gap_classifier(distribution_hidden), self.orientation_classifier(distribution_hidden)


class BridgeResidualHead(nn.Module):
    def __init__(self, input_channels: int = 13, embedding_dim: int = 32, hidden_dim: int = 64, kernel_size: int = 7):
        super().__init__()
        kernel_size = max(int(kernel_size), 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_sizes = [kernel_size, max(kernel_size - 2, 3), max(kernel_size - 4, 3)]
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.bridge_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.composite_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, interface_tensor: torch.Tensor, pair_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(interface_tensor).squeeze(-1)
        residual = self.bridge_scale * self.residual_head(torch.cat([encoded, pair_embedding], dim=-1)).squeeze(-1)
        composite_logit = self.composite_head(encoded).squeeze(-1)
        return residual, composite_logit


class MonotonicChemicalPotentialHead(nn.Module):
    def __init__(self, availability_dim: int, state_dim: int, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.availability_weight = nn.Parameter(torch.zeros(availability_dim, hidden_dim))
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, availability: torch.Tensor, state_context: torch.Tensor, tf_embedding: torch.Tensor) -> torch.Tensor:
        positive_weight = torch.nn.functional.softplus(self.availability_weight)
        hidden = availability @ positive_weight + self.state_proj(torch.cat([state_context, tf_embedding], dim=-1)) + self.bias
        return self.out(torch.nn.functional.gelu(hidden)).squeeze(-1)


class StateGateHead(nn.Module):
    def __init__(self, state_dim: int, embedding_dim: int = 32, hidden_dim: int = 64, dropout_p: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + embedding_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout_p)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, state_context: torch.Tensor, left_embedding: torch.Tensor, right_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_features = torch.cat(
            [left_embedding, right_embedding, left_embedding * right_embedding, left_embedding - right_embedding],
            dim=-1,
        )
        values = self.network(torch.cat([state_context, pair_features], dim=-1))
        gate = torch.sigmoid(values[:, 0]) * 2.0
        correction = 0.1 * torch.tanh(values[:, 1])
        return gate, correction


class PartitionUsageHead(nn.Module):
    def __init__(self, state_dim: int, enable_signed_gain_diagnostic: bool = False):
        super().__init__()
        self.enable_signed_gain_diagnostic = bool(enable_signed_gain_diagnostic)
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_z_weight = nn.Parameter(torch.tensor(0.75, dtype=torch.float32))
        self.cooperative_weight = nn.Parameter(torch.tensor(1.25, dtype=torch.float32))
        self.residual_weight = nn.Parameter(torch.tensor(0.75, dtype=torch.float32))
        self.signed_gain_weight = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.state_bias = nn.Linear(state_dim, 1)

    def forward(
        self,
        left_effective_energy: torch.Tensor,
        right_effective_energy: torch.Tensor,
        state_residual: torch.Tensor,
        state_context: torch.Tensor,
        compatibility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compatibility = compatibility if compatibility is not None else torch.ones_like(state_residual)
        log_z_mono = torch.log1p(torch.exp(-left_effective_energy) + torch.exp(-right_effective_energy))
        log_z_pair = torch.log(torch.exp(log_z_mono) + compatibility * torch.exp(-(left_effective_energy + right_effective_energy + state_residual)))
        cooperative_gain = log_z_pair - log_z_mono
        residual_drive = compatibility * torch.tanh(-state_residual)
        signed_pair_gain = cooperative_gain * torch.tanh(-state_residual)
        logits = (
            self.bias
            + torch.nn.functional.softplus(self.log_z_weight) * log_z_mono
            + torch.nn.functional.softplus(self.cooperative_weight) * cooperative_gain
            + torch.nn.functional.softplus(self.residual_weight) * residual_drive
            + self.state_bias(state_context).squeeze(-1)
        )
        if self.enable_signed_gain_diagnostic:
            logits = logits + torch.nn.functional.softplus(self.signed_gain_weight) * signed_pair_gain
        usage_probability = torch.sigmoid(logits)
        return usage_probability, log_z_mono, cooperative_gain, signed_pair_gain


@dataclass
class HelixPairOutputs:
    left_intrinsic_energy: torch.Tensor
    right_intrinsic_energy: torch.Tensor
    intrinsic_monomer_energy: torch.Tensor
    left_offset_weights: torch.Tensor
    right_offset_weights: torch.Tensor
    geometry_residual: torch.Tensor
    bridge_residual: torch.Tensor
    biochemical_residual: torch.Tensor
    gap_logits: torch.Tensor
    orientation_logits: torch.Tensor
    composite_logit: torch.Tensor
    pair_embedding: torch.Tensor
    left_chemical_potential: torch.Tensor | None = None
    right_chemical_potential: torch.Tensor | None = None
    state_gate: torch.Tensor | None = None
    state_correction: torch.Tensor | None = None
    state_residual: torch.Tensor | None = None
    usage_probability: torch.Tensor | None = None
    monomer_free_energy: torch.Tensor | None = None
    cooperative_gain: torch.Tensor | None = None
    signed_pair_gain: torch.Tensor | None = None
    availability_only_probability: torch.Tensor | None = None


class HelixPairModel(nn.Module):
    def __init__(
        self,
        num_families: int,
        num_subfamilies: int,
        num_paralogs: int,
        num_tfs: int,
        geometry_dim: int,
        availability_dim: int,
        state_dim: int,
        sequence_channels: int = 5,
        embedding_dim: int = 32,
        rank: int = 8,
        interface_channels: int = 13,
        num_gap_bins: int = 25,
        helical_order: int = 2,
        use_hierarchy_embedding: bool = True,
        use_anchor_refinement: bool = True,
        use_geometry_head: bool = True,
        use_bridge_head: bool = True,
        use_state_gate: bool = True,
        use_partition: bool = True,
        use_availability: bool = True,
        disable_helical_basis: bool = False,
        disable_uncertainty_ensemble: bool = False,
        monomer_hidden_dim: int = 64,
        bridge_hidden_dim: int = 64,
        bridge_kernel_size: int = 7,
        state_gate_dropout_p: float = 0.1,
        enable_signed_gain_diagnostic: bool = False,
    ):
        super().__init__()
        self.offsets = tuple(DEFAULT_OFFSETS)
        self.use_anchor_refinement = use_anchor_refinement
        self.use_geometry_head = use_geometry_head
        self.use_bridge_head = use_bridge_head
        self.use_state_gate = use_state_gate
        self.use_partition = use_partition
        self.use_availability = use_availability
        self.disable_helical_basis = disable_helical_basis
        self.disable_uncertainty_ensemble = disable_uncertainty_ensemble
        self.enable_signed_gain_diagnostic = bool(enable_signed_gain_diagnostic)
        self.helical_order = int(helical_order)
        self.embedding = (
            TFHierarchyEmbedding(num_families, num_subfamilies, num_paralogs, num_tfs, embedding_dim=embedding_dim)
            if use_hierarchy_embedding
            else FlatTFEmbedding(num_tfs=num_tfs, embedding_dim=embedding_dim)
        )
        self.monomer_head = MonomerEnergyHead(
            sequence_channels=sequence_channels,
            embedding_dim=embedding_dim,
            hidden_dim=monomer_hidden_dim,
        )
        self.geometry_head = GeometryResidualHead(
            embedding_dim=embedding_dim,
            geometry_dim=geometry_dim,
            rank=rank,
            num_gap_bins=num_gap_bins,
            helical_order=helical_order,
        )
        self.bridge_head = BridgeResidualHead(
            input_channels=interface_channels,
            embedding_dim=embedding_dim,
            hidden_dim=bridge_hidden_dim,
            kernel_size=bridge_kernel_size,
        )
        self.chemical_potential_head = MonotonicChemicalPotentialHead(
            availability_dim=availability_dim,
            state_dim=state_dim,
            embedding_dim=embedding_dim,
        )
        self.state_gate_head = StateGateHead(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            dropout_p=0.0 if disable_uncertainty_ensemble else float(state_gate_dropout_p),
        )
        self.partition_head = PartitionUsageHead(
            state_dim=state_dim,
            enable_signed_gain_diagnostic=self.enable_signed_gain_diagnostic,
        )

    def _intrinsic_energy(
        self,
        anchor_offsets: torch.Tensor,
        tf_ids: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.embedding(**tf_ids)
        energy, weights = self.monomer_head(anchor_offsets, embedding, use_anchor_refinement=self.use_anchor_refinement)
        return energy, weights, embedding

    def forward(
        self,
        left_anchor_offsets: torch.Tensor,
        right_anchor_offsets: torch.Tensor,
        left_ids: dict[str, torch.Tensor],
        right_ids: dict[str, torch.Tensor],
        geometry_features: torch.Tensor,
        interface_tensor: torch.Tensor,
        availability: torch.Tensor | None = None,
        state_context: torch.Tensor | None = None,
        compatibility: torch.Tensor | None = None,
    ) -> HelixPairOutputs:
        left_intrinsic, left_weights, left_embedding = self._intrinsic_energy(left_anchor_offsets, left_ids)
        right_intrinsic, right_weights, right_embedding = self._intrinsic_energy(right_anchor_offsets, right_ids)
        intrinsic_monomer_energy = left_intrinsic + right_intrinsic
        geometry_input = geometry_features.clone()
        if self.disable_helical_basis:
            helical_start = 7
            helical_end = helical_start + (2 * self.helical_order)
            if geometry_input.shape[-1] >= helical_end:
                geometry_input[:, helical_start:helical_end] = 0.0
        if self.use_geometry_head:
            geometry_residual, gap_logits, orientation_logits = self.geometry_head(left_embedding, right_embedding, geometry_input)
        else:
            geometry_residual = torch.zeros_like(left_intrinsic)
            gap_logits = torch.zeros(
                (left_intrinsic.shape[0], self.geometry_head.gap_classifier.out_features),
                dtype=left_intrinsic.dtype,
                device=left_intrinsic.device,
            )
            orientation_logits = torch.zeros(
                (left_intrinsic.shape[0], self.geometry_head.orientation_classifier.out_features),
                dtype=left_intrinsic.dtype,
                device=left_intrinsic.device,
            )
        if self.use_bridge_head:
            bridge_residual, composite_logit = self.bridge_head(interface_tensor, left_embedding * right_embedding)
        else:
            bridge_residual = torch.zeros_like(geometry_residual)
            composite_logit = torch.zeros_like(geometry_residual)
        biochemical_residual = geometry_residual + bridge_residual
        outputs = HelixPairOutputs(
            left_intrinsic_energy=left_intrinsic,
            right_intrinsic_energy=right_intrinsic,
            intrinsic_monomer_energy=intrinsic_monomer_energy,
            left_offset_weights=left_weights,
            right_offset_weights=right_weights,
            geometry_residual=geometry_residual,
            bridge_residual=bridge_residual,
            biochemical_residual=biochemical_residual,
            gap_logits=gap_logits,
            orientation_logits=orientation_logits,
            composite_logit=composite_logit,
            pair_embedding=left_embedding * right_embedding,
        )
        if availability is None or state_context is None:
            return outputs

        if self.use_availability:
            left_chemical_potential = self.chemical_potential_head(availability, state_context, left_embedding)
            right_chemical_potential = self.chemical_potential_head(availability, state_context, right_embedding)
        else:
            left_chemical_potential = torch.zeros_like(left_intrinsic)
            right_chemical_potential = torch.zeros_like(right_intrinsic)
        left_effective_energy = left_intrinsic - left_chemical_potential
        right_effective_energy = right_intrinsic - right_chemical_potential
        availability_only_probability, monomer_free_energy, _, _ = self.partition_head(
            left_effective_energy,
            right_effective_energy,
            torch.zeros_like(biochemical_residual),
            state_context,
            compatibility=compatibility,
        )
        if self.use_state_gate:
            gate, correction = self.state_gate_head(state_context, left_embedding, right_embedding)
        else:
            gate = torch.ones_like(biochemical_residual)
            correction = torch.zeros_like(biochemical_residual)
        state_residual = gate * biochemical_residual + correction
        if self.use_partition:
            usage_probability, monomer_free_energy, cooperative_gain, signed_pair_gain = self.partition_head(
                left_effective_energy,
                right_effective_energy,
                state_residual,
                state_context,
                compatibility=compatibility,
            )
        else:
            # Without the partition head, fall back to a bounded correction on top of the
            # availability-only occupancy rather than a stronger direct energy shortcut.
            base_probability = availability_only_probability.clamp(1.0e-5, 1.0 - 1.0e-5)
            base_logit = torch.logit(base_probability)
            compatibility_term = compatibility if compatibility is not None else torch.ones_like(state_residual)
            cooperative_gain = 0.5 * compatibility_term * torch.tanh(-state_residual)
            signed_pair_gain = cooperative_gain
            usage_probability = torch.sigmoid(base_logit + cooperative_gain)
        outputs.left_chemical_potential = left_chemical_potential
        outputs.right_chemical_potential = right_chemical_potential
        outputs.state_gate = gate
        outputs.state_correction = correction
        outputs.state_residual = state_residual
        outputs.usage_probability = usage_probability
        outputs.monomer_free_energy = monomer_free_energy
        outputs.cooperative_gain = cooperative_gain
        outputs.signed_pair_gain = signed_pair_gain
        outputs.availability_only_probability = availability_only_probability
        return outputs
