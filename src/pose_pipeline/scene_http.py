"""Session-scoped graph/query endpoints shared by desktop and embedded clients."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .contracts import sha256_file
from .live_io import read_json
from .scene_query import execute_query


def route(controller, path):
    if path.startswith('/s/'):
        parts = path.split('/')
        if len(parts) < 4:
            raise ValueError('Invalid session route')
        return controller.session_view(parts[2]), '/' + '/'.join(parts[3:])
    return controller, path


def graph_for(view, up=None):
    if view.state.get('status') != 'completed':
        raise ValueError('地图尚未完成，不能查询旧结果。')
    context = view.scene_context()
    graph = dict(view.scene_graph(up))
    graph['context'] = context
    graph['map_sha256'] = sha256_file(Path(view.state['result']['final_cloud']))
    return graph


def get(handler, controller, path):
    rest = '/' + '/'.join(path.split('/')[3:]) if path.startswith('/s/') else path
    if rest not in {'/api/graph', '/scene_graph.json', '/api/evidence', '/api/status'} and path not in {'/api/sessions', '/scene_query_ui.js'}:
        return False
    try:
        if path == '/scene_query_ui.js':
            handler.respond(200, Path(__file__).with_name('scene_query_ui.js').read_bytes(), 'text/javascript; charset=utf-8')
            return True
        if path == '/api/sessions':
            from .device_gui import _session_summary
            current = controller.session.name if controller.session else ''
            rows = []
            seen = set()
            for library in controller.library_roots():
                for root in sorted(library.glob('scan_*'), reverse=True):
                    if root.name in seen or not (root / 'session.json').is_file():
                        continue
                    controller.resolve_session(root.name)
                    seen.add(root.name)
                    rows.append(_session_summary(root, current))
            handler.respond(200, {'current': current, 'sessions': rows})
            return True
        view, rest = route(controller, path)
        if rest == '/api/status':
            state = view.snapshot()
            state['historical'] = view is not controller
            handler.respond(200, state)
            return True
        query = parse_qs(urlparse(handler.path).query)
        if rest == '/api/evidence':
            data = view.evidence_image(int(query['instance_id'][0]), int(query['index'][0]), query['context'][0])
            handler.respond(200, data, 'image/jpeg')
            return True
        up = [float(v) for v in query['world_up'][0].split(',')] if query.get('world_up') else None
        graph = graph_for(view, up)
        handler.respond(200, graph)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        handler.respond(409, {'error': str(error)})
    return True


def post(handler, controller, path):
    rest = '/' + '/'.join(path.split('/')[3:]) if path.startswith('/s/') else path
    if rest not in {'/api/query', '/api/reprocess', '/api/open'}:
        return False
    try:
        length = int(handler.headers.get('Content-Length', '0'))
        if not 0 < length <= 8192:
            raise ValueError('Invalid request size')
        options = json.loads(handler.rfile.read(length))
        if not isinstance(options, dict):
            raise ValueError('Expected a query object')
        if rest == '/api/open':
            controller.open_session(options['session'])
            handler.respond(200, controller.snapshot())
            return True
        if rest == '/api/reprocess':
            if path.startswith('/s/'):
                options['session'] = path.split('/')[2]
            controller.reprocess(options)
            handler.respond(200, controller.snapshot())
            return True
        if set(options) - {'question', 'query', 'context', 'world_up'}:
            raise ValueError('Unexpected query fields')
        view, _ = route(controller, path)
        with view.lock:
            context = view.scene_context()
            if options.get('context') != context:
                raise ValueError('地图已更新，请刷新后重新查询。')
            graph = graph_for(view, options.get('world_up'))
            result = execute_query(graph, question=options.get('question'), plan=options.get('query'))
            if view.scene_context() != context:
                raise ValueError('地图在查询期间发生变化，请重试。')
            result.update(context=context, map_sha256=graph['map_sha256'])
            handler.respond(200, {'query_result': result, 'state': view.snapshot()})
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        handler.respond(409, {'error': str(error)})
    return True
