import numpy as np
from pose_pipeline.sam3_fusion import PixelClaims
from pose_pipeline.sam3_refine import resolve_families, interior


def make_claims(extra=False):
    c=PixelClaims((12,12),35);mask=np.ones((12,12),bool)
    c.add(8,mask,.93);c.add(17,mask,.92)
    if extra:c.add(5,mask,.95)
    return c


def test_subtype_requires_real_compatible_anchors():
    rows,cols=np.indices((12,12));r=rows.ravel();u=cols.ravel()
    labels=np.full(144,17);confidence=np.ones(144)
    sem,*_=resolve_families(make_claims(),r,u,labels,confidence)
    assert np.all(sem==17)
    sem,*_=resolve_families(make_claims(),r,u,np.full(144,8),confidence)
    assert np.all(sem==8)  # never force desks into tables
    sem,*_=resolve_families(make_claims(),r,u,np.full(144,5),confidence)
    assert np.all(sem==33)  # incompatible SGF class cannot name this object


def test_other_family_conflict_stays_unknown():
    r,u=np.indices((12,12));sem,*_=resolve_families(make_claims(True),r.ravel(),u.ravel(),np.full(144,17),np.ones(144))
    assert not sem.any()


def test_single_view_interior_does_not_grow_boundaries():
    sem=np.zeros((7,7),int);sem[1:6,1:6]=5
    safe=interior(sem)
    assert safe.sum()==9 and not safe[1].any()
