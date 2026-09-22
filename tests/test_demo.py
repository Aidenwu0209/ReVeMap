import json

import pytest

from pose_pipeline.artifacts import load_artifacts
from pose_pipeline.scene_graph import query_graph
from revemap.demo import create_demo


def test_demo_is_synthetic_unbound_and_queryable(tmp_path):
    out = tmp_path / 'demo'
    summary = create_demo(out)
    assert summary['synthetic'] and not summary['model_inference_executed']
    graph = json.loads((out / 'scene_graph.json').read_text())
    result = query_graph(graph, label='椅子')
    assert result['instance_ids'] == [3]
    assert result['provenance']['data_source'] == 'synthetic_geometric_boxes'
    assert 'hypotheses' in result['relation_scope']
    assert query_graph(graph, label='cup', relation='supported_by', reference_id=1)['instance_ids'] == [2]
    assert not load_artifacts(out)['provenance_bound']
    with pytest.raises(ValueError, match='bound input provenance'):
        load_artifacts(out, require_provenance=True)
    with pytest.raises(FileExistsError):
        create_demo(out)
