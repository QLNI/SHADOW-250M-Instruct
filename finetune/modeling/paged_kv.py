
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


_POPCOUNT = tuple(int(i).bit_count() for i in range(256))


@dataclass(frozen=True)
class ArchiveAccounting:
    logical_tokens: int
    page_count: int
    payload_bytes: int
    allocated_payload_bytes: int
    position_index_bytes: int
    bytes_copied_on_append: int

    @property
    def allocated_total_bytes(self) -> int:
        return self.allocated_payload_bytes + self.position_index_bytes


class PagedKVArchive:

    format = "paged_random_walsh_ctv_1bit_v1"
    state_version = 1

    def __init__(
        self,
        batch: int,
        heads: int,
        packed_width: int,
        *,
        page_size: int = 256,
        device=None,
    ):
        if min(batch, heads, packed_width, page_size) < 1:
            raise ValueError("batch, heads, packed_width, and page_size must be positive")
        self.batch = int(batch)
        self.heads = int(heads)
        self.packed_width = int(packed_width)
        self.page_size = int(page_size)
        self.device = torch.device(device or "cpu")
        self.k_pages: list[torch.Tensor] = []
        self.v_pages: list[torch.Tensor] = []
        self.position_pages: list[torch.Tensor] = []
        self.length = 0
        self.bytes_copied_on_append = 0

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.heads, self.length, self.packed_width

    def __len__(self) -> int:
        return self.length

    def _allocate_page(self):
        shape = (self.batch, self.heads, self.page_size, self.packed_width)
        self.k_pages.append(torch.empty(shape, dtype=torch.uint8, device=self.device))
        self.v_pages.append(torch.empty(shape, dtype=torch.uint8, device=self.device))
        self.position_pages.append(
            torch.empty(self.page_size, dtype=torch.int64, device=self.device)
        )

    def append(
        self,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        positions: torch.Tensor | Iterable[int] | None = None,
    ) -> None:
        expected_prefix = (self.batch, self.heads)
        if packed_k.dtype != torch.uint8 or packed_v.dtype != torch.uint8:
            raise TypeError("paged one-bit K/V payloads must be torch.uint8")
        if packed_k.shape != packed_v.shape:
            raise ValueError("packed K and V shapes differ")
        if (
            packed_k.ndim != 4
            or tuple(packed_k.shape[:2]) != expected_prefix
            or packed_k.shape[-1] != self.packed_width
        ):
            raise ValueError(
                "expected packed K/V shape "
                f"({self.batch},{self.heads},T,{self.packed_width}), "
                f"got {tuple(packed_k.shape)}"
            )
        tokens = int(packed_k.shape[2])
        if tokens == 0:
            return
        if packed_k.device != self.device or packed_v.device != self.device:
            raise ValueError(
                f"archive is on {self.device}, payload is on "
                f"{packed_k.device}/{packed_v.device}"
            )
        if positions is None:
            pos = torch.arange(
                self.length, self.length + tokens, dtype=torch.int64, device=self.device
            )
        else:
            pos = torch.as_tensor(positions, dtype=torch.int64, device=self.device)
            if pos.ndim != 1 or pos.numel() != tokens:
                raise ValueError(f"positions must contain exactly {tokens} entries")

        source = 0
        while source < tokens:
            page_index = self.length // self.page_size
            page_offset = self.length % self.page_size
            if page_index == len(self.k_pages):
                self._allocate_page()
            take = min(tokens - source, self.page_size - page_offset)
            dst = slice(page_offset, page_offset + take)
            src = slice(source, source + take)
            self.k_pages[page_index][:, :, dst].copy_(packed_k[:, :, src])
            self.v_pages[page_index][:, :, dst].copy_(packed_v[:, :, src])
            self.position_pages[page_index][dst].copy_(pos[src])
            copied = 2 * self.batch * self.heads * take * self.packed_width
            self.bytes_copied_on_append += copied
            self.length += take
            source += take

    def _validate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if indices.ndim != 3 or tuple(indices.shape[:2]) != (self.batch, self.heads):
            raise ValueError(
                f"indices must have shape ({self.batch},{self.heads},K), "
                f"got {tuple(indices.shape)}"
            )
        if indices.numel() and (
            int(indices.min().item()) < 0 or int(indices.max().item()) >= self.length
        ):
            raise IndexError(f"archive index outside [0,{self.length})")
        return indices

    def gather(self, indices: torch.Tensor, kind: str) -> torch.Tensor:
        indices = self._validate_indices(indices)
        if kind not in {"k", "v"}:
            raise ValueError("kind must be 'k' or 'v'")
        pages = self.k_pages if kind == "k" else self.v_pages
        out = torch.empty(
            (*indices.shape, self.packed_width),
            dtype=torch.uint8,
            device=self.device,
        )
        page_ids = torch.div(indices, self.page_size, rounding_mode="floor")
        offsets = indices.remainder(self.page_size)
        for page_id in torch.unique(page_ids).tolist():
            mask = page_ids == int(page_id)
            b, h, slot = mask.nonzero(as_tuple=True)
            out[b, h, slot] = pages[int(page_id)][b, h, offsets[b, h, slot]]
        return out

    def gather_positions(self, indices: torch.Tensor) -> torch.Tensor:
        indices = self._validate_indices(indices)
        out = torch.empty(indices.shape, dtype=torch.int64, device=self.device)
        page_ids = torch.div(indices, self.page_size, rounding_mode="floor")
        offsets = indices.remainder(self.page_size)
        for page_id in torch.unique(page_ids).tolist():
            mask = page_ids == int(page_id)
            b, h, slot = mask.nonzero(as_tuple=True)
            out[b, h, slot] = self.position_pages[int(page_id)][offsets[b, h, slot]]
        return out

    @staticmethod
    def _popcount_bytes(value: torch.Tensor) -> torch.Tensor:
        lut = torch.tensor(_POPCOUNT, dtype=torch.int16, device=value.device)
        return lut[value.long()].sum(-1, dtype=torch.int32)

    def exact_hamming_topk(
        self, query: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.length == 0:
            raise ValueError("cannot search an empty archive")
        if (
            query.dtype != torch.uint8
            or query.ndim != 3
            or tuple(query.shape) != (self.batch, self.heads, self.packed_width)
        ):
            raise ValueError(
                "query must be uint8 with shape "
                f"({self.batch},{self.heads},{self.packed_width})"
            )
        take = min(max(1, int(k)), self.length)
        best_indices = torch.empty(
            (self.batch, self.heads, 0), dtype=torch.long, device=self.device
        )
        best_distances = torch.empty(
            (self.batch, self.heads, 0), dtype=torch.int32, device=self.device
        )
        absolute = 0
        tie_base = self.length + 1
        for page_id, page in enumerate(self.k_pages):
            used = min(self.page_size, self.length - absolute)
            if used <= 0:
                break
            distance = self._popcount_bytes(page[:, :, :used] ^ query[:, :, None])
            indices = torch.arange(
                absolute, absolute + used, dtype=torch.long, device=self.device
            ).view(1, 1, used).expand(self.batch, self.heads, used)
            candidate_indices = torch.cat((best_indices, indices), dim=-1)
            candidate_distances = torch.cat((best_distances, distance), dim=-1)
            rank_key = candidate_distances.to(torch.int64) * tie_base + candidate_indices
            selected = torch.topk(
                rank_key, min(take, rank_key.shape[-1]), largest=False, sorted=True
            ).indices
            best_indices = torch.gather(candidate_indices, -1, selected)
            best_distances = torch.gather(candidate_distances, -1, selected)
            absolute += used
        return best_indices, best_distances

    def materialize(self, kind: str) -> torch.Tensor:
        if kind not in {"k", "v"}:
            raise ValueError("kind must be 'k' or 'v'")
        pages = self.k_pages if kind == "k" else self.v_pages
        if not pages:
            return torch.empty(self.shape, dtype=torch.uint8, device=self.device)
        parts = []
        remaining = self.length
        for page in pages:
            used = min(self.page_size, remaining)
            parts.append(page[:, :, :used])
            remaining -= used
        return torch.cat(parts, dim=2)

    def accounting(self) -> ArchiveAccounting:
        payload = 2 * self.batch * self.heads * self.length * self.packed_width
        allocated = (
            2
            * self.batch
            * self.heads
            * len(self.k_pages)
            * self.page_size
            * self.packed_width
        )
        positions = len(self.position_pages) * self.page_size * 8
        return ArchiveAccounting(
            logical_tokens=self.length,
            page_count=len(self.k_pages),
            payload_bytes=payload,
            allocated_payload_bytes=allocated,
            position_index_bytes=positions,
            bytes_copied_on_append=self.bytes_copied_on_append,
        )

    def state(self) -> dict:
        return {
            "format": self.format,
            "version": self.state_version,
            "batch": self.batch,
            "heads": self.heads,
            "packed_width": self.packed_width,
            "page_size": self.page_size,
            "length": self.length,
            "bytes_copied_on_append": self.bytes_copied_on_append,
            "k_pages": [page.clone() for page in self.k_pages],
            "v_pages": [page.clone() for page in self.v_pages],
            "position_pages": [page.clone() for page in self.position_pages],
        }

    @classmethod
    def from_state(cls, state: dict, device=None) -> "PagedKVArchive":
        if state.get("format") != cls.format or int(state.get("version", -1)) != 1:
            raise ValueError("unknown paged KV archive state")
        inferred = state["k_pages"][0].device if state["k_pages"] else "cpu"
        target = torch.device(device or inferred)
        archive = cls(
            state["batch"],
            state["heads"],
            state["packed_width"],
            page_size=state["page_size"],
            device=target,
        )
        archive.k_pages = [page.to(target) for page in state["k_pages"]]
        archive.v_pages = [page.to(target) for page in state["v_pages"]]
        archive.position_pages = [page.to(target) for page in state["position_pages"]]
        archive.length = int(state["length"])
        archive.bytes_copied_on_append = int(state["bytes_copied_on_append"])
        return archive


class PagedKVView:

    def __init__(self, archive: PagedKVArchive, kind: str):
        self.archive = archive
        self.kind = kind

    @property
    def shape(self):
        return self.archive.shape

    @property
    def dtype(self):
        return torch.uint8

    @property
    def device(self):
        return self.archive.device

    def materialize(self):
        return self.archive.materialize(self.kind)


class ExactChunkCountView:

    def __init__(self, archive: PagedKVArchive, chunk_size: int):
        self.archive = archive
        self.chunk_size = int(chunk_size)

    @property
    def shape(self):
        complete = self.archive.length // self.chunk_size
        return (
            self.archive.batch,
            self.archive.heads,
            complete,
            self.archive.packed_width,
        )
