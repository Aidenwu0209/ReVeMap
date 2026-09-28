"""Exercise already-loaded BLAS pools, including restoration after failures."""
import sys
import types

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from pose_pipeline import rgbd_measured


@pytest.mark.parametrize("threads,fail", [(None, False), (4, False), (2, True)])
def test_graph_caps_loaded_pools_even_without_environment(monkeypatch, tmp_path, threads, fail):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)
    # No Open3D/geometry fixture is needed to verify the real BLAS controller.
    monkeypatch.setitem(sys.modules, "open3d", types.ModuleType("open3d"))
    np.ones((32, 32)) @ np.ones((32, 32))
    if sys.platform == 'darwin' and not threadpool_info():
        pytest.skip("Apple Accelerate is not a threadpoolctl-managed BLAS pool")
    expected = 2 if threads is None else threads

    def body(dense, output, limit):
        pools = [p for p in threadpool_info() if p['user_api'] == 'blas']
        assert pools
        assert all(p['num_threads'] <= expected for p in pools)
        assert limit == expected
        if fail:
            raise RuntimeError("graph failure")
        return {"ok": True}

    monkeypatch.setattr(rgbd_measured, "_run_measured_graph", body)
    with threadpool_limits(limits=8):
        before = [(p['filepath'], p['num_threads']) for p in threadpool_info()]
        kwargs = {} if threads is None else {"threads": threads}
        if fail:
            with pytest.raises(RuntimeError, match="graph failure"):
                rgbd_measured.run_measured_graph(tmp_path, tmp_path / 'out', **kwargs)
        else:
            assert rgbd_measured.run_measured_graph(tmp_path, tmp_path / 'out', **kwargs) == {"ok": True}
        assert [(p['filepath'], p['num_threads']) for p in threadpool_info()] == before


@pytest.mark.parametrize("threads", [0, -1, 2.5, True, "2"])
def test_invalid_graph_threads_fail_before_geometry_import(tmp_path, threads):
    with pytest.raises(ValueError, match="positive integer"):
        rgbd_measured.run_measured_graph(tmp_path, tmp_path / 'out', threads=threads)
    assert not (tmp_path / 'out').exists()
