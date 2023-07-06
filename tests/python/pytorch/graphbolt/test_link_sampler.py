import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch


def rand_graph(N, density):
    # Graph metadata
    ntypes = {"n1": 0, "n2": 1, "n3": 2}
    etypes = {
        ("n1", "e1", "n2"): 0,
        ("n1", "e2", "n3"): 1,
        ("n3", "e3", "n1"): 2,
    }
    metadata = gb.GraphMetadata(ntypes, etypes)
    return gb_test_utils.rand_hetero_csc_graph(N, density, metadata)


def collect_nodes(g, dict):
    nodes = []
    for etype, node_pair in dict:
        u_type, _, v_type = etype
        u_type_id = g.metadata.node_type_to_id[u_type]
        v_type_id = g.metadata.node_type_to_id[v_type]
        u, v = node_pair
        nodes.append(u + g.node_type_offset[u_type_id])
        nodes.append(v + g.node_type_offset[v_type_id])
    return torch.unique(torch.cat(nodes))


def test_LinkSampler_Independent_Format():
    graph = rand_graph(100, 0.05)
    N1 = torch.randint(0, 33, (15,))
    N2 = torch.randint(0, 33, (5,))
    N3 = torch.randint(0, 33, (10,))
    unique_N1, compacted_N1 = torch.unique(N1, return_inverse=True)
    unique_N2, compacted_N2 = torch.unique(N2, return_inverse=True)
    unique_N3, compacted_N3 = torch.unique(N3, return_inverse=True)
    unique_N1 = unique_N1 + graph.node_type_offset[0]
    unique_N2 = unique_N2 + graph.node_type_offset[1]
    unique_N3 = unique_N3 + graph.node_type_offset[2]
    expected_unique_nodes = torch.sort(
        torch.cat([unique_N1, unique_N2, unique_N3])
    )[0]
    expected_compacted_pairs = {
        ("n1", "e1", "n2"): (
            compacted_N1[:5],
            compacted_N2[:5],
        ),
        ("n1", "e2", "n3"): (
            compacted_N1[5:10],
            compacted_N3[:5],
        ),
        ("n3", "e3", "n1"): (
            compacted_N3[5:10],
            compacted_N1[10:15],
        ),
    }

    node_pairs_with_label = {
        ("n1", "e1", "n2"): (
            N1[:5],
            N2[:5],
            torch.randint(0, 2, (5,)),
        ),
        ("n1", "e2", "n3"): (
            N1[5:10],
            N3[:5],
            torch.randint(0, 2, (5,)),
        ),
        ("n3", "e3", "n1"): (
            N3[5:10],
            N1[10:15],
            torch.randint(0, 2, (5,)),
        ),
    }
    link_sampler = gb.LinkNeighborSampler(
        graph,
        1,
        gb.LinkedDataFormat.INDEPENDENT,
    )
    compacted_node_pairs_with_label, sub_g = link_sampler(node_pairs_with_label)
    unique_seed_nodes = collect_nodes(sub_g)
    assert torch.equal(unique_seed_nodes, expected_unique_nodes)
    assert len(expected_compacted_pairs) == len(compacted_node_pairs_with_label)
    for etype, pair_with_label in compacted_node_pairs_with_label.items():
        u, v, label = pair_with_label
        expected_label = node_pairs_with_label[etype][2]
        expected_u, expected_v = expected_compacted_pairs[etype]
        assert torch.equal(u, expected_u)
        assert torch.equal(v, expected_v)
        assert torch.equal(label, expected_label)


def test_LinkSampler_Conditioned_Format():
    graph = rand_graph(100, 0.05)
    negative_ratio = 1
    N1 = torch.randint(0, 33, (15,))
    N2 = torch.randint(0, 33, (5,))
    N3 = torch.randint(0, 33, (10,))
    unique_N1, compacted_N1 = torch.unique(N1, return_inverse=True)
    unique_N2, compacted_N2 = torch.unique(N2, return_inverse=True)
    unique_N3, compacted_N3 = torch.unique(N3, return_inverse=True)
    unique_N1 = unique_N1 + graph.node_type_offset[0]
    unique_N2 = unique_N2 + graph.node_type_offset[1]
    unique_N3 = unique_N3 + graph.node_type_offset[2]
    expected_unique_nodes = torch.sort(
        torch.cat([unique_N1, unique_N2, unique_N3])
    )[0]
    expected_compacted_pairs = {
        ("n1", "e1", "n2"): (
            compacted_N1[:5],
            compacted_N2[:5],
            compacted_N1[:5].view(-1, 1),
            compacted_N2[3:8].view(-1, 1),
        ),
        ("n1", "e2", "n3"): (
            compacted_N1[5:10],
            compacted_N3[:5],
            compacted_N1[5:10].view(-1, 1),
            compacted_N3[4:9].view(-1, 1),
        ),
        ("n3", "e3", "n1"): (
            compacted_N3[5:10],
            compacted_N1[10:15],
            compacted_N3[5:10].view(-1, 1),
            compacted_N1[2:7].view(-1, 1),
        ),
    }

    node_pairs = {
        ("n1", "e1", "n2"): (
            N1[:5],
            N2[:5],
            N1[:5].view(-1, 1),
            N2[3:8].view(-1, 1),
        ),
        ("n1", "e2", "n3"): (
            N1[5:10],
            N3[:5],
            N1[5:10].view(-1, 1),
            N3[4:9].view(-1, 1),
        ),
        ("n3", "e3", "n1"): (
            N3[5:10],
            N1[10:15],
            N3[5:10].view(-1, 1),
            N1[2:7].view(-1, 1),
        ),
    }
    link_sampler = gb.LinkNeighborSampler(
        graph,
        negative_ratio,
        gb.LinkedDataFormat.CONDITIONED,
    )
    compacted_node_pairs, sub_g = link_sampler(node_pairs)
    unique_seed_nodes = collect_nodes(sub_g)
    assert torch.equal(unique_seed_nodes, expected_unique_nodes)
    assert len(expected_compacted_pairs) == len(compacted_node_pairs)
    for etype, pair in compacted_node_pairs.items():
        u, v, neg_u, neg_v = pair
        (
            expected_u,
            expected_v,
            expected_neg_u,
            expected_neg_v,
        ) = expected_compacted_pairs[etype]
        assert torch.equal(u, expected_u)
        assert torch.equal(v, expected_v)
        assert torch.equal(neg_u, expected_neg_u)
        assert torch.equal(neg_v, expected_neg_v)
