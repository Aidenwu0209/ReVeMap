/* RGB TSDF surface layer. Point labels and object manipulation retain their own view. */
(() => {
  const controls = document.createElement('div');
  controls.className = 'surface-controls'; controls.hidden = true;
  controls.innerHTML = '<button type="button" id="surfacePoints"></button><button type="button" id="surfaceMesh"></button><a id="surfaceExport"></a><span id="surfaceStatus" role="status"></span>';
  document.querySelector('.viewer').appendChild(controls);
  const style = document.createElement('style');
  style.textContent = '.surface-controls{position:absolute;z-index:6;top:88px;left:24px;right:24px;display:flex;align-items:center;gap:6px;flex-wrap:wrap;pointer-events:none}.surface-controls[hidden],.point-controls[hidden]{display:none}.viewer .surface-controls button,.surface-controls a{pointer-events:auto;border:1px solid #555;border-radius:9px;background:#29292ee8;color:#eee;padding:7px 12px;min-height:36px;min-width:50px;font:inherit;font-size:13px;text-decoration:none;backdrop-filter:blur(16px)}.surface-controls button[aria-pressed="true"]{background:#316b59;border-color:#58cfa5}.surface-controls button:disabled{opacity:.5}.surface-controls span{font-size:12px;color:#c1c1c6;background:#151518c9;border-radius:6px;padding:4px 7px;max-width:320px}.surface-controls a[hidden]{display:none}';
  document.head.appendChild(style);
  const pointsButton = $('surfacePoints'), meshButton = $('surfaceMesh'), download = $('surfaceExport'), status = $('surfaceStatus');
  let active = false, identity = '', metadata = null, checkedAt = 0, checking = false;
  let loading = false, errorText = '', geometry = null, savedCamera = null, meshProgram = null;
  let vertexBuffer = null, indexBuffer = null, indexType, attributes = [];

  function dispose() {
    if (gl && vertexBuffer) gl.deleteBuffer(vertexBuffer);
    if (gl && indexBuffer) gl.deleteBuffer(indexBuffer);
    vertexBuffer = indexBuffer = null; geometry = null;
  }
  function points(redraw = true) {
    if (!active) return;
    active = false;
    if (gl) for (const location of attributes) gl.disableVertexAttribArray(location);
    restoreCamera(savedCamera); savedCamera = null;
    document.querySelector('.point-controls').hidden = false;
    updateObjectCaption(); updateControls();
    if (redraw) draw();
  }
  function reset() {
    points(false); dispose(); metadata = null; checkedAt = 0; identity = '';
    loading = false; checking = false; errorText = ''; controls.hidden = true;
  }
  function caption() {
    if (!active || !metadata) return false;
    $('cloudKind').textContent = T('原色表面','RGB surface');
    $('pointCount').textContent = metadata.viewer_triangles.toLocaleString() + T(' 个三角面',' triangles')
      + (metadata.viewer_triangles < metadata.triangles ? T(' · 显示已简化',' · simplified display') : '');
    return true;
  }
  function updateControls() {
    controls.hidden = lastState?.status !== 'completed';
    pointsButton.textContent = T('点云','Points'); meshButton.textContent = loading ? T('加载中…','Loading…') : T('表面','Surface');
    pointsButton.setAttribute('aria-pressed', String(!active)); meshButton.setAttribute('aria-pressed', String(active));
    meshButton.disabled = !metadata?.available || loading || !gl;
    download.textContent = T('导出表面','Export surface'); download.hidden = !active;
    status.textContent = errorText || (metadata && !metadata.available ? T('此记录尚未生成表面','No surface for this record') : '');
    status.hidden = !status.textContent;
    caption();
  }
  async function inspect() {
    const next = JSON.stringify([currentSessionId,lastState?.scene_context,lastState?.status]);
    if (next !== identity) { reset(); identity = next; }
    updateControls();
    if (checking || lastState?.status !== 'completed' || !currentSessionId
        || (metadata?.available && !errorText) || Date.now()-checkedAt < 10000) return;
    const requested = identity, base = sessionBase(), generation = sessionGeneration;
    checking = true; checkedAt = Date.now();
    try {
      const response = await fetch(base + '/api/surface'); const data = await response.json();
      if (!response.ok) throw Error(data.error || T('表面读取失败','Could not read surface'));
      if (requested !== identity || generation !== sessionGeneration) return;
      if (data.context !== lastState?.scene_context) return;
      metadata = data; errorText = ''; updateControls();
    } catch (error) { if (requested === identity) { errorText = error.message; points(); updateControls(); } }
    finally { if (requested === identity) checking = false; }
  }
  function program() {
    if (meshProgram) return meshProgram;
    const p = gl.createProgram();
    gl.attachShader(p,shader(gl.VERTEX_SHADER,`attribute vec3 position;attribute vec3 color;attribute vec3 normal;
      uniform vec3 center;uniform float scale;uniform vec2 angle;uniform vec2 pan;uniform float aspect;varying vec3 rgb;
      vec3 turn(vec3 p){float a=angle.x,b=angle.y;vec3 q=vec3(cos(a)*p.x+sin(a)*p.z,p.y,-sin(a)*p.x+cos(a)*p.z);return vec3(q.x,cos(b)*q.y-sin(b)*q.z,sin(b)*q.y+cos(b)*q.z);}
      void main(){vec3 p=turn((position-center)*scale);gl_Position=vec4((p.x+pan.x)/aspect,-p.y+pan.y,p.z*.08,1.);
      float light=.58+.42*abs(dot(normalize(turn(normal)),normalize(vec3(.3,-.4,1.))));rgb=color*light;}`));
    gl.attachShader(p,shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 rgb;void main(){gl_FragColor=vec4(rgb,1.);}'));
    gl.linkProgram(p); if (!gl.getProgramParameter(p,gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p));
    attributes = ['position','color','normal'].map(name=>gl.getAttribLocation(p,name));
    meshProgram = p; return p;
  }
  async function show() {
    if (!metadata?.available || loading || !gl) return;
    const requested = identity, generation = sessionGeneration, meta = metadata, base = sessionBase();
    loading = true; errorText = ''; updateControls();
    try {
      if (!geometry) {
        const suffix = '?revision='+encodeURIComponent(meta.revision)+'&context='+encodeURIComponent(meta.context);
        const response = await fetch(base+'/surface.bin'+suffix);
        if (!response.ok) throw Error(T('表面已更新或加载失败，请重试','Surface changed or could not load; retry'));
        const raw = await response.arrayBuffer();
        if (requested !== identity || generation !== sessionGeneration) return;
        const header = new DataView(raw);
        if (raw.byteLength < 16 || String.fromCharCode(...new Uint8Array(raw,0,8)) !== 'RVMESH01') throw Error('Invalid surface format');
        const n = header.getUint32(8,true), faces = header.getUint32(12,true);
        if (!n || !faces || n!==meta.viewer_vertices || faces!==meta.viewer_triangles || raw.byteLength!==16+n*36+faces*12) throw Error('Invalid surface size');
        const vertices = new Float32Array(raw,16,n*9), indices = new Uint32Array(raw,16+n*36,faces*3);
        if (vertices.some(value=>!Number.isFinite(value)) || indices.some(value=>value>=n)) throw Error('Invalid surface geometry');
        program(); dispose();
        vertexBuffer = gl.createBuffer(); indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER,vertexBuffer); gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);
        if (n<=65535) { indexType=gl.UNSIGNED_SHORT;gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(indices),gl.STATIC_DRAW); }
        else { if (!gl.getExtension('OES_element_index_uint')) throw Error(T('此设备不支持该表面，可导出后查看','This device cannot display this surface'));indexType=gl.UNSIGNED_INT;gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW); }
        geometry = {count:faces*3};
        download.href=base+'/surface.ply'+suffix;download.download=currentSessionId+'-surface.ply';
      }
      if (requested !== identity || generation !== sessionGeneration) return;
      if (!active) savedCamera=cameraState();
      active=true;bounds=meta.bounds;document.querySelector('.point-controls').hidden=true;
      if (objectIsolated) fit();else draw();
      caption();
    } catch (error) { if (requested===identity) { errorText=error.message;dispose();points();metadata=null; } }
    finally { if (requested===identity) { loading=false;updateControls(); } }
  }
  function renderSurface() {
    if (!active || !geometry || !gl) return false;
    const p=program();gl.useProgram(p);
    gl.uniform3fv(gl.getUniformLocation(p,'center'),center);gl.uniform1f(gl.getUniformLocation(p,'scale'),scale*zoom);
    gl.uniform2f(gl.getUniformLocation(p,'angle'),yaw,pitch);gl.uniform2fv(gl.getUniformLocation(p,'pan'),pan);
    gl.uniform1f(gl.getUniformLocation(p,'aspect'),canvas.width/canvas.height);
    gl.bindBuffer(gl.ARRAY_BUFFER,vertexBuffer);
    attributes.forEach((location,i)=>{gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,3,gl.FLOAT,false,36,i*12);});
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);gl.disable(gl.CULL_FACE);
    gl.drawElements(gl.TRIANGLES,geometry.count,indexType,0);return true;
  }
  pointsButton.onclick=()=>points();meshButton.onclick=show;
  window.surfaceViewer={draw:renderSurface,points,reset,caption};
  setInterval(inspect,600);inspect();
})();
