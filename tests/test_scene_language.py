import copy
import numpy as np
import pytest

from pose_pipeline.scene_graph import build_graph, query_graph
from pose_pipeline.scene_query import execute_query, parse_question, validate_plan


def box(low, high):
    return [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]


def room(up=(0, 0, 1)):
    # Fixed oracle: table 1, monitor 2 above it, chair 3 near it, remote chair 4;
    # cabinet 5 contains cup 6 by AABB only.
    bounds = [([0,0,0],[1,1,1]), ([.2,.2,1.1],[.8,.8,1.5]),
              ([1.05,0,0],[1.5,.5,.7]), ([4,0,0],[4.5,.5,.7]),
              ([8,0,0],[10,2,2]), ([8.5,.5,.5],[9,1,1])]
    points = np.array([p for a,b in bounds for p in box(a,b)])
    ids = np.repeat(np.arange(1,7),8)
    semantic = np.repeat([1,2,3,3,4,5],8)
    return build_graph(points, semantic, ids, {'0':'unknown','1':'table','2':'monitor','3':'chair','4':'cabinet','5':'cup'}, world_up=up)


@pytest.mark.parametrize('question,ids,count', [
    ('找出所有椅子',[3,4],2), ('桌子附近有哪些椅子',[3],1),
    ('显示器下方有什么',[1],1), ('找出桌子附近且位于显示器下方的物体',[],0),
    ('有几把椅子',[3,4],2), ('当前地图中检测到了几把椅子',[3,4],2),
    ('柜子的包围盒内有哪些物体',[6],1), ('离桌子最近的椅子',[3],1),
    ('柜子的包围盒包含哪些物体',[6],1),
    ('实例#1附近有哪些椅子',[3],1),
    ('find all chairs',[3,4],2), ('chairs near the table',[3],1),
    ('how many chairs are near the table',[3],1), ('objects below the monitor',[1],1),
    ('chairs near the table and below the monitor',[],0),
    ('how many chairs are there',[3,4],2), ('cups inside the cabinet',[6],1),
])
def test_fixed_language_oracles(question,ids,count):
    result=execute_query(room(),question=question)
    assert result['instance_ids']==ids, result
    assert result['count']==count, result


def test_empty_count_unknown_reference_and_missing_gravity_are_distinct():
    assert execute_query(room(),question='有几张床')['count']==0
    result=execute_query(room(),question='床附近有几把椅子')
    assert result['count'] is None and result['reason']=='reference_not_found'
    result=execute_query(room(None),question='显示器下方有几个物体')
    assert result['count'] is None and result['reason']=='world_up_required'


def test_reference_ambiguity_requires_choice_and_returns_a_reusable_plan():
    graph=room()
    result=execute_query(graph,question='椅子附近有哪些物体')
    assert result['status']=='ambiguous' and result['instance_ids']==[]
    assert [n['instance_id'] for n in result['candidates']]==[3,4]
    plan=result['query'];plan['conditions'][0]['reference']={'instance_id':3}
    assert execute_query(graph,plan=plan)['instance_ids']==[1,2]


@pytest.mark.parametrize('question',['找出红色椅子','找出不是椅子的物体','桌子附近或者床下方的物体','为什么有椅子','chairs near the table or below the monitor'])
def test_unsupported_conditions_are_not_silently_dropped(question):
    result=execute_query(room(),question=question)
    assert result['status']=='unsupported' and not result['instance_ids']


def test_nearest_is_applied_after_filters_independently_of_condition_order():
    plan={'operation':'find','label':'chair','conditions':[
        {'relation':'nearest','reference':{'instance_id':1}},
        {'relation':'near','reference':{'instance_id':2}}]}
    graph=room();a=execute_query(graph,plan=plan)
    plan['conditions'].reverse()
    assert a['instance_ids']==execute_query(graph,plan=plan)['instance_ids']==[3]


def test_containment_is_directed_and_does_not_depend_on_gravity():
    graph=room(None)
    assert query_graph(graph,relation='inside',reference_id=5)['instance_ids']==[6]
    assert query_graph(graph,relation='contains',reference_id=6)['instance_ids']==[5]
    assert query_graph(graph,relation='contains',reference_id=5)['instance_ids']==[]


def test_invalid_plan_rejected_and_graph_unchanged():
    graph=room();before=copy.deepcopy(graph)
    execute_query(graph,question='有几把椅子')
    assert before==graph
    for plan in [{'operation':'eval'}, {'conditions':[{'relation':'near','reference':{'instance_id':True}}]},
                 {'conditions':[],'sql':'DELETE'}, {'conditions':[{'relation':'near','reference':{'label':'table','instance_id':1}}]}]:
        with pytest.raises(ValueError):validate_plan(plan)
