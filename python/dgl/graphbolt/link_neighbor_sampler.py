"""Negative samplers"""


from collections.abc import Mapping

import torch

from .linked_data_format import LinkedDataFormat

__all__ = ["LinkNeighborSampler"]


class LinkNeighborSampler:
    def __init__(
        self,
        graph,
        negative_ratio,
        linked_data_format,
    ):
        """
        Initlization for a negative sampler.

        Parameters
        ----------
        negative_ratio : int
            The proportion of negative samples to positive samples. 0 means
            there is no negative edges in the given data.
        linked_data_format : LinkedDataFormat
            Determines the format of the output data:
                - Conditioned format: Outputs data as quadruples
                `[u, v, [negative heads], [negative tails]]`. Here, 'u' and 'v'
                are the source and destination nodes of positive edges,  while
                'negative heads' and 'negative tails' refer to the source and
                destination nodes of negative edges.
                - Independent format: Outputs data as triples `[u, v, label]`.
                In this case, 'u' and 'v' are the source and destination nodes
                of an edge, and 'label' indicates whether the edge is negative
                (0) or positive (1).
        """
        super().__init__()
        self.graph = graph
        self.negative_ratio = negative_ratio
        assert linked_data_format in [
            LinkedDataFormat.CONDITIONED,
            LinkedDataFormat.INDEPENDENT,
        ], f"Unsupported data format: {linked_data_format}."
        self.linked_data_format = linked_data_format

    def __call__(self, link_data):
        """

        Parameters
        ----------
        link_data : List[Tensor] or Dict[etype, List[Tensor]]
            Represents source-destination node pairs of positive edges, where
            positive means the edge must exist in the graph.

        Returns
        -------
        List[Tensor] or Dict[etype, List[Tensor]]
            A collection of edges or a dictionary that maps etypes to lists of
            edges which includes both positive and negative samples. The format
            of it is determined by the provided 'linked_data_format'.
        """

        def collect(data):
            u, v, neg_u, neg_v = data[:4]
            u = torch.cat([u, neg_u.view(-1)])
            v = torch.cat([v, neg_v.view(-1)])
            return (u, v)

        def dispatch(data):
            u, v = data
            pos_len = u.numel() // (self.negative_ratio + 1)
            u, neg_u = u[:pos_len], u[pos_len:]
            v, neg_v = v[:pos_len], v[pos_len:]
            return (
                u,
                v,
                neg_u.view(-1, self.negative_ratio),
                neg_v.view(-1, self.negative_ratio),
            )

        node_pairs = {}
        has_conditioned_negative = (
            self.linked_data_format == LinkedDataFormat.CONDITIONED
            and self.negative_ratio > 0
        )
        if has_conditioned_negative:
            if isinstance(link_data, Mapping):
                node_pairs = {
                    etype: collect(data) for etype, data in link_data.items()
                }
            else:
                node_pairs = collect(link_data)
        else:
            node_pairs = {etype: data[:2] for etype, data in link_data.items()}
            node_pairs = link_data[:2]

        compacted_pairs, sub_graph = self.graph.sample_neighbors_for_pairs(
            node_pairs
        )
        if has_conditioned_negative:
            if isinstance(link_data, Mapping):
                link_data = {
                    etype: collect(compacted_pairs[etype]) + data[4:]
                    for etype, data in link_data.items()
                }
            else:
                link_data = dispatch(compacted_pairs) + link_data[4:]
        else:
            if isinstance(link_data, Mapping):
                link_data = {
                    etype: compacted_pairs[etype] + data[2:]
                    for etype, data in link_data.items()
                }
            else:
                link_data = compacted_pairs + link_data[2:]
        return (link_data, self.graph.split_fused_homogeneous_graph(sub_graph))
