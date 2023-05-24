# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Optional, Sequence, Tuple, Type, Union

import numpy as np
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _collate_fn_t

from opacus.lightning import DPDataLoader, DPLightningDataModule

logger = logging.getLogger(__name__)


def wrap_collate_with_empty(
    *,
    collate_fn: Optional[_collate_fn_t],
    keys: Sequence[str],
    shapes: Sequence[Tuple],
    dtypes: Sequence[Union[np.dtype, Type]],
):
    """
    Wraps given collate function to handle empty batches.

    Args:
        collate_fn: collate function to wrap
        sample_empty_shapes: expected shape for a batch of size 0. Input is a sequence -
            one for each tensor in the dataset

    Returns:
        New collate function, which is equivalent to input ``collate_fn`` for non-empty
        batches and outputs empty tensors with shapes from ``sample_empty_shapes`` if
        the input batch is of size 0
    """

    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return None
            # return dict((key, torch.as_tensor(np.zeros(shape, dtype=dtype))) for key, shape, dtype in zip(keys, shapes, dtypes))

    return collate


def shape_safe(x: Any) -> Tuple:
    """
    Exception-safe getter for ``shape`` attribute

    Args:
        x: any object

    Returns:
        ``x.shape`` if attribute exists, empty tuple otherwise
    """
    if hasattr(x, "shape"): return x.shape
    elif isinstance(x, int): return tuple()
    else: return None


def dtype_safe(x: Any) -> Union[np.dtype, Type]:
    """
    Exception-safe getter for ``dtype`` attribute

    Args:
        x: any object

    Returns:
        ``x.dtype`` if attribute exists, type of x otherwise
    """
    return x.dtype if hasattr(x, "dtype") else type(x)


class MyDPDataLoader(DPDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: float,
        collate_fn: Optional[_collate_fn_t] = None,
        drop_last: bool = False,
        generator=None,
        distributed: bool = False,
        **kwargs,
    ):
        """

        Args:
            dataset: See :class:`torch.utils.data.DataLoader`
            sample_rate: probability with which each element of the dataset is included
                in the next batch.
            num_workers: See :class:`torch.utils.data.DataLoader`
            collate_fn: See :class:`torch.utils.data.DataLoader`
            pin_memory: See :class:`torch.utils.data.DataLoader`
            drop_last: See :class:`torch.utils.data.DataLoader`
            timeout: See :class:`torch.utils.data.DataLoader`
            worker_init_fn: See :class:`torch.utils.data.DataLoader`
            multiprocessing_context: See :class:`torch.utils.data.DataLoader`
            generator: Random number generator used to sample elements
            prefetch_factor: See :class:`torch.utils.data.DataLoader`
            persistent_workers: See :class:`torch.utils.data.DataLoader`
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
                Selects between ``DistributedUniformWithReplacementSampler`` and
                ``UniformWithReplacementSampler`` sampler implementations
        """

        self.sample_rate = sample_rate
        self.distributed = distributed

        if distributed:
            batch_sampler = DistributedUniformWithReplacementSampler(
                total_size=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        else:
            batch_sampler = UniformWithReplacementSampler(
                num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        # sample_empty_shapes = [(0, *shape_safe(x)) for x in dataset[0]]
        # dtypes = [dtype_safe(x) for x in dataset[0]]
        keys = dataset[0].keys()
        shapes = [(0, *shape_safe(x)) for x in dataset[0].values()]
        dtypes = [dtype_safe(x) for x in dataset[0].values()]
        if collate_fn is None:
            collate_fn = default_collate


        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        super(DPDataLoader, self).__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
                keys=keys,
                shapes=shapes,
                dtypes=dtypes,
                # sample_empty_shapes=sample_empty_shapes,
                # dtypes=dtypes,
            ),
            generator=generator,
            **kwargs,
        )


class MyDPLightningDataModule(DPLightningDataModule):
    def train_dataloader(self):
        dataloader = self.datamodule.train_dataloader()
        return MyDPDataLoader.from_data_loader(dataloader, distributed=False)
