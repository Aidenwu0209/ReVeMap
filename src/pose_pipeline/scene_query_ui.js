/* Session-bound, bounded queries. Rendering only uses backend-selected IDs. */
(() => {
  const panel = document.createElement('section');
  panel.id = 'sceneQuery'; panel.className = 'card query-card';
  panel.innerHTML = `<div class="card-heading"><strong>场景查询</strong><button id="queryClose" type="button" aria-label="关闭查询">×</button></div>
    <form id="sceneQuestionForm"><label for="sceneQuestion">用一句话查找物体</label>
    <input id="sceneQuestion" type="text" maxlength="512" placeholder="桌子附近有哪些椅子" autocomplete="off">
    <label for="sceneUp">已确认的地图向上方向</label><select id="sceneUp"><option value="">尚未确定</option><option value="0,0,1">+Z</option><option value="0,0,-1">−Z</option><option value="0,1,0">+Y</option><option value="0,-1,0">−Y</option></select>
    <div class="query-actions"><button id="sceneAsk" type="submit">查询</button><button id="sceneClear" type="button">清除高亮</button><button id="sceneExport" type="button">导出场景图</button></div></form>
    <p class="query-hint">支持查找、计数、附近、上方、下方、包围盒包含与“且”条件。例：有几把椅子；实例#1附近且显示器下方的物体。</p>
    <div id="sceneAnswer" role="status" aria-live="polite">查询范围为当前地图中的预测对象。</div><div id="sceneCandidates"></div><div id="sceneEvidence"></div>`;
  const style = document.createElement('style');
  style.textContent = `.query-card{grid-column:1/-1}.query-card label{display:block;font-size:12px;color:#aaa;margin:12px 0 5px}.query-card input,.query-card select{width:100%;box-sizing:border-box;min-height:42px;font:inherit;color:#eee;background:#29292e;border:1px solid #555;border-radius:8px;padding:9px}.query-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.query-card button,.query-toggle{cursor:pointer;min-height:38px;color:#eee;background:#35353c;border:1px solid #666;border-radius:8px;padding:7px 10px}.query-card button:disabled{opacity:.5}.query-hint{font-size:12px;line-height:1.6;color:#a1a1aa}#sceneAnswer{font-size:14px;line-height:1.7;margin:12px 0}#sceneEvidence{font-size:12px;line-height:1.7;overflow-wrap:anywhere}#sceneEvidence a{color:#90c4ff;margin-right:10px}#sceneEvidence img{max-width:100%;max-height:180px;display:block;border-radius:6px;margin-top:8px}#sceneCandidates button{margin:4px}#queryClose{display:none}.query-toggle{display:none}.embed .query-toggle{display:block;position:fixed;right:82px;top:22px;z-index:15;background:#363638df;backdrop-filter:blur(15px)}.embed #sceneQuery{display:none;position:fixed;right:14px;top:78px;bottom:24px;width:min(360px,calc(100vw - 28px));overflow:auto;box-sizing:border-box;z-index:16;background:#1c1c1ef5;padding:18px;border:1px solid #555;border-radius:14px;box-shadow:0 8px 28px #0008}.embed #sceneQuery.query-open{display:block}.embed #queryClose{display:block;float:right}.query-card[hidden],.query-toggle[hidden]{display:none!important}`;
  document.head.appendChild(style);
  if (EMBED) document.body.appendChild(panel);
  else document.querySelector('.side').insertBefore(panel, document.getElementById('objectsCard'));
  const toggle = document.createElement('button'); toggle.className = 'query-toggle'; toggle.textContent = '查询';
  toggle.onclick = () => panel.classList.toggle('query-open'); document.body.appendChild(toggle);
  const el = id => document.getElementById(id);
  el('queryClose').onclick = () => panel.classList.remove('query-open');
  const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'button';
  retry.textContent = '使用已保存数据重新处理'; retry.hidden = true;
  document.querySelector('.progress-card').appendChild(retry);
  retry.onclick = async () => {
    if (!currentSessionId || retry.disabled) return;
    retry.disabled = true;
    try {
      const response = await fetch(source() + '/api/reprocess', {method:'POST',
        headers:{'Content-Type':'application/json','X-Scan-Token':token},
        body:JSON.stringify({vlm:el('vlm').value,schedule:el('schedule').value,refine:el('refine').checked})});
      const state = await response.json(); if (!response.ok) throw Error(state.error || '重新处理失败');
      resetSession(currentSessionId); render(state);
    } catch (error) { el('notice').textContent = error.message; el('notice').style.display = 'block'; }
    finally { retry.disabled = false; }
  };
  let identity = '', pending = false;
  function clear() { el('sceneCandidates').replaceChildren(); el('sceneEvidence').replaceChildren(); window.scanQueryHighlight?.(null); }
  function source() { return currentSessionId ? '/s/' + encodeURIComponent(currentSessionId) : ''; }
  function up() { return el('sceneUp').value ? el('sceneUp').value.split(',').map(Number) : null; }
  function apply(result, base) {
    clear(); el('sceneAnswer').textContent = result.answer || result.reason || result.status;
    if (result.status === 'ambiguous') {
      for (const candidate of result.candidates || []) {
        const button = document.createElement('button'); button.type = 'button';
        button.textContent = `#${candidate.instance_id} · ${candidate.name || candidate.label}`;
        button.onclick = () => {
          const plan = JSON.parse(JSON.stringify(result.query));
          for (const condition of plan.conditions) if (condition.reference.label === result.reference.label) condition.reference = {instance_id: candidate.instance_id};
          run({query: plan});
        }; el('sceneCandidates').appendChild(button);
      }
    }
    if (result.status !== 'matched') return;
    window.scanQueryHighlight?.(result.instance_ids);
    const info = document.createElement('p'); info.textContent = `命中 ${result.count} 个预测实例。关系来自地图几何，包围盒包含不表示真实装载或接触。`;
    el('sceneEvidence').appendChild(info);
    for (const node of (result.objects || []).slice(0, 50)) {
      const row = document.createElement('div');
      const name = document.createElement('strong'); name.textContent = `#${node.instance_id} ${node.name || node.label}`;
      row.appendChild(name);
      const link = document.createElement('a'); link.href = `${base}/object.ply?kind=instance&id=${node.instance_id}`; link.textContent = ' 导出物体'; link.download = `instance-${node.instance_id}.ply`; row.appendChild(link);
      if (node.name_category_conflict) { const note = document.createElement('span'); note.textContent = ' 名称与类别不一致'; row.appendChild(note); }
      if (!(node.evidence || []).length) { const note = document.createElement('span'); note.textContent = ' · 无可用观察裁图'; row.appendChild(note); }
      for (const [i, record] of (node.evidence || []).slice(0, 3).entries()) {
        const link = document.createElement('a');
        link.href = `${base}/api/evidence?instance_id=${node.instance_id}&index=${i}&context=${encodeURIComponent(result.context)}`;
        link.textContent = ` 观察帧 ${record.frame_id ?? i}`;
        link.onclick = event => { event.preventDefault(); const image = document.createElement('img'); image.alt = `实例 ${node.instance_id} 的原始观察`; image.src = link.href; image.onerror = () => { image.remove(); const note = document.createElement('span'); note.textContent = ' 证据文件已变化或不可用'; row.appendChild(note); }; row.appendChild(image); };
        row.appendChild(link);
      }
      el('sceneEvidence').appendChild(row);
    }
    for (const edge of (result.evidence || []).slice(0, 40)) {
      const row = document.createElement('div'); row.textContent = edge.relation ? `#${edge.source} — ${edge.relation} → #${edge.target}` : `#${edge.instance_id} 到 #${edge.reference_id}：${edge.centroid_distance_m.toFixed(3)} m`;
      el('sceneEvidence').appendChild(row);
    }
  }
  async function run(payload) {
    if (pending || lastState?.status !== 'completed') return;
    const sid = currentSessionId, context = lastState.scene_context, base = source();
    pending = true; el('sceneAsk').disabled = true; el('sceneAnswer').textContent = '正在查询…';
    try {
      const response = await fetch(base + '/api/query', {method:'POST', headers:{'Content-Type':'application/json','X-Scan-Token':token}, body:JSON.stringify({...payload,context,world_up:up()})});
      const data = await response.json(); if (!response.ok) throw Error(data.error || '查询失败');
      if (sid !== currentSessionId || context !== lastState?.scene_context || data.query_result.context !== context) throw Error('地图已切换，请重新查询。');
      apply(data.query_result, base);
    } catch (error) { clear(); el('sceneAnswer').textContent = error.message; }
    finally { pending = false; el('sceneAsk').disabled = false; }
  }
  el('sceneQuestionForm').onsubmit = event => { event.preventDefault(); run({question:el('sceneQuestion').value}); };
  el('sceneClear').onclick = () => { clear(); el('sceneAnswer').textContent = '已清除查询高亮。'; };
  el('sceneExport').onclick = async () => {
    try {
      const sid = currentSessionId, context = lastState.scene_context;
      const response = await fetch(source() + '/scene_graph.json' + (up() ? '?world_up=' + up().join(',') : ''));
      const graph = await response.json(); if (!response.ok) throw Error(graph.error);
      if (sid !== currentSessionId || graph.context !== context) throw Error('地图已更新，请重试。');
      const link = document.createElement('a');
      link.href = source() + '/scene_graph.json?context=' + encodeURIComponent(context) + (up() ? '&world_up=' + up().join(',') : '');
      link.download = sid + '-scene-graph.json'; link.click();
    } catch(error) { el('sceneAnswer').textContent = error.message; }
  };
  setInterval(() => {
    const available = lastState?.status === 'completed'; panel.hidden = !available; toggle.hidden = !available;
    retry.hidden = EMBED || !lastState?.can_reprocess || ['starting','recording','stopping','mapping','cancelling'].includes(lastState?.status);
    const next = JSON.stringify([currentSessionId,lastState?.scene_context,available]);
    if (next !== identity) { identity = next; clear(); el('sceneAnswer').textContent = '查询范围为当前地图中的预测对象。'; }
  }, 400);
})();
