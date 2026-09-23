const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');

// Execute the shipped page, with only DOM/WebGL/network/clock replaced. The
// native configure/hand entry points and all camera/selection logic are real.
function viewer() {
  const messages = [], frames = new Map(), elements = new Map(), deleted = [];
  let frameId = 0, bufferId = 0, now = 100000;
  const noop = () => {};
  const gl = new Proxy({
    createBuffer: () => ++bufferId, deleteBuffer: b => deleted.push(b),
    getShaderParameter: () => true, getProgramParameter: () => true,
    getAttribLocation: (_, name) => ['position','color','normal'].indexOf(name),
    getExtension: () => true,
  }, {get: (target,key) => key in target ? target[key] : /^[A-Z_]+$/.test(key) ? 1 : noop});
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      style: {}, dataset: {}, value: '', hidden: false, textContent: '',
      clientWidth: 1000, clientHeight: 800, width: 1000, height: 800,
      classList: {add: noop, remove: noop, toggle: noop, contains: () => false},
      setAttribute: noop, removeAttribute: noop, addEventListener: noop,
      append: noop, appendChild: noop, replaceChildren: noop, querySelector: () => null,
      getContext: () => gl, setPointerCapture: noop,
    });
    return elements.get(id);
  }
  const document = {
    getElementById: element, querySelector: element, querySelectorAll: () => [],
    createElement: () => element(Symbol()), addEventListener: noop,
    body: element('body'), head: element('head'), documentElement: element('html'),
  };
  const window = {addEventListener: noop, devicePixelRatio: 1,
    webkit: {messageHandlers: {scanViewer: {postMessage: m => messages.push(JSON.parse(JSON.stringify(m)))}}}};
  const context = vm.createContext({window, document, URLSearchParams, Float32Array,
    Uint8Array, Uint16Array, Uint32Array, DataView, ArrayBuffer,
    location: {search: '?session=test&embed=1'}, console,
    ResizeObserver: class {observe() {}},
    Date: class extends Date {static now() {return now}},
    requestAnimationFrame: cb => {frames.set(++frameId,cb);return frameId},
    setTimeout: noop, setInterval: noop, clearTimeout: noop,
    fetch: () => new Promise(() => {}),
  });
  const run = code => vm.runInContext(code,context);
  const html = readFileSync(join(__dirname,'../../src/pose_pipeline/device_gui.html'),'utf8');
  for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) run(match[1]);
  function advance(ms = 500) {
    now += ms;
    for (let i=0; frames.size && i<80; i++) {
      const batch = [...frames.values()];frames.clear();batch.forEach(cb => cb());
    }
  }
  const snapshot = () => JSON.parse(run('JSON.stringify({airHolding,objectIsolated,pointN,pan,yaw,pitch,airNeedsOpen,airTransition:!!airTransition,center,scale,zoom})'));
  const settings = {language:'zh',selection:{kind:'instance',id:1,title:'chair'},
    isolated:false,isolationRevision:0,airEnabled:true,airSession:'session1',airMode:'move',reduceMotion:false};
  const configure = changes => {Object.assign(settings,changes);window.scanViewer.configure(settings)};
  let sequence = 0;
  const hand = (phase,extra={}) => window.scanViewer.hand({session:settings.airSession,
    sequence:++sequence,mode:settings.airMode,phase,x:.5,y:.5,...extra});
  async function load() {
    context.fetch = async url => url.includes('/api/objects') ? {ok:true,json:async()=>({
      revision:'r1',points:4,stride:8,format:'xyz_rgb_semantic_instance_float32_le',
      rgb_available:true,classes:[],instances:[{label_id:1,semantic_name:'chair'}],
    })} : {ok:true,arrayBuffer:async()=>new Float32Array([
      0,0,0,1,0,0,1,1, 1,1,1,1,0,0,1,1,
      4,4,4,0,1,0,2,2, 5,5,5,0,1,0,2,2,
    ]).buffer};
    await run('loadObjects()');configure({});advance();
  }
  return {window,context,run,messages,deleted,advance,snapshot,configure,hand,load,element};
}

test('loaded objects enable native Air Grab; lift, move, release and Put Back restore scene', async () => {
  const v=viewer();assert.equal(v.messages.length,0);await v.load();
  assert.deepEqual(v.messages[0],{action:'objectsReady'});
  const scene=v.snapshot();v.hand('ready');v.hand('began');
  assert.equal(v.snapshot().pointN,2);assert.equal(v.snapshot().airHolding,true);
  assert.equal(v.messages.at(-1).action,'airGrabbed');
  assert.equal(v.messages.at(-1).isolationRevision,'0');
  v.configure({isolated:true});v.hand('changed',{x:.7,y:.4});v.advance();
  assert.ok(v.snapshot().pan[0]>.4);assert.ok(v.snapshot().pan[1]>.1);
  assert.equal(v.deleted.length,1);v.hand('ended');
  assert.equal(v.snapshot().airHolding,false);assert.equal(v.snapshot().objectIsolated,true);
  v.configure({isolated:false,isolationRevision:1});v.advance();
  for (const k of ['pan','center','scale','zoom','yaw','pitch','pointN']) assert.deepEqual(v.snapshot()[k],scene[k],k);
  v.hand('began');assert.equal(v.snapshot().objectIsolated,false);
  v.hand('ready');v.hand('began');assert.equal(v.snapshot().airHolding,true);
});

test('session, sequence and mode gates reject stale packets; rotate is separate from move', async () => {
  const v=viewer();await v.load();v.hand('ready');
  v.hand('began',{session:'old'});v.hand('began',{mode:'rotate'});v.hand('began',{sequence:0});
  assert.equal(v.snapshot().airHolding,false);
  v.configure({airMode:'rotate',reduceMotion:true});v.hand('began');
  assert.equal(v.snapshot().airHolding,false);v.hand('ready');v.hand('began');
  const before=v.snapshot();v.hand('changed',{x:.7,y:.6});v.advance();
  assert.ok(v.snapshot().yaw<before.yaw-.9);assert.ok(v.snapshot().pitch>before.pitch+.4);
  assert.deepEqual(v.snapshot().pan,before.pan);assert.equal(v.snapshot().airTransition,false);
  v.hand('lost');v.hand('began');assert.equal(v.snapshot().airHolding,false);
  v.hand('ready');v.hand('began');assert.equal(v.snapshot().airHolding,true);
});

test('touch, query highlights and record changes stop an active grab', async () => {
  const v=viewer();await v.load();v.hand('ready');v.hand('began');
  v.element('view').onpointerdown({pointerType:'touch',pointerId:1,clientX:0,clientY:0,preventDefault(){}});
  assert.equal(v.snapshot().airHolding,false);assert.equal(v.messages.at(-1).action,'airTouch');
  v.hand('began');assert.equal(v.snapshot().airHolding,false);
  v.hand('ready');v.hand('began');v.window.scanQueryHighlight([2]);v.advance();
  assert.equal(v.snapshot().airHolding,false);assert.equal(v.snapshot().objectIsolated,false);
  v.configure({selection:{kind:'instance',id:1}});v.hand('ready');v.hand('began');
  v.run("resetSession('another')");v.advance();
  assert.equal(v.snapshot().pointN,0);assert.equal(v.snapshot().airTransition,false);
});

test('surface switches cancel holds and a delayed mesh cannot override a new grab', async () => {
  const v=viewer();await v.load();v.run("lastState={status:'completed',scene_context:'ctx'}");
  const meta={available:true,context:'ctx',revision:'r',viewer_vertices:3,viewer_triangles:1,triangles:1,bounds:{min:[0,0,0],max:[5,5,5]}};
  let resolveMesh;
  v.context.fetch=async url=>url.includes('/api/surface')?{ok:true,json:async()=>meta}:new Promise(resolve=>{resolveMesh=resolve});
  v.run(readFileSync(join(__dirname,'../../src/pose_pipeline/surface_viewer.js'),'utf8'));
  await new Promise(resolve=>setImmediate(resolve));
  const pending=v.element('surfaceMesh').onclick();
  v.hand('ready');v.hand('began');v.advance();assert.equal(v.snapshot().airHolding,true);
  const raw=new ArrayBuffer(16+3*36+12),header=new DataView(raw);
  new Uint8Array(raw,0,8).set(Buffer.from('RVMESH01'));header.setUint32(8,3,true);header.setUint32(12,1,true);
  new Uint32Array(raw,16+3*36,3).set([0,1,2]);
  resolveMesh({ok:true,arrayBuffer:async()=>raw});await pending;
  assert.equal(v.window.surfaceViewer.draw(),false);
  await v.element('surfaceMesh').onclick();assert.equal(v.snapshot().airHolding,false);
  assert.equal(v.window.surfaceViewer.draw(),true);
  v.hand('ready');v.hand('began');assert.equal(v.window.surfaceViewer.draw(),false);
  assert.equal(v.snapshot().airHolding,true);
});

test('unavailable or invalid object buffers never enable native Air Grab', async () => {
  const v=viewer();v.context.fetch=async()=>({status:404});
  await v.run('loadObjects()');assert.equal(v.messages.length,0);
  v.context.fetch=async()=>({ok:true,json:async()=>({stride:99})});
  await v.run('loadObjects()');assert.equal(v.messages.length,0);
});
