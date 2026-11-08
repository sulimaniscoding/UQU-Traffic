"""OSM geometry plus explicitly hypothetical metering; routes checked for both gates."""
from pathlib import Path
import heapq, json, os, shutil, subprocess, sys, xml.etree.ElementTree as ET
import numpy as np
import sumo, sumolib

ROOT=Path(__file__).resolve().parent
GATE_NODES={'G1_IN_A':'8930377812','G1_IN_B':'8930377813','G1_OUT':'8930377811','G2_IN':'8930377799','G2_OUT':'8930377794'}
GATE_PARENT={'G1_IN_A':'133751581','G1_IN_B':'133751592','G1_OUT':'133751587','G2_IN':'157161607','G2_OUT':'157161613'}
ZONE_POINTS={'Z_NORTH':(39.9508,21.3311),'Z_CENTRAL':(39.9524,21.3298),'Z_WEST':(39.9478,21.3282),'Z_SOUTH':(39.952,21.326),'Z_EAST':(39.9568,21.3282)}

def exe(name):
    return shutil.which(name) or str(Path(sumo.SUMO_HOME)/'bin'/(name+('.exe' if sys.platform=='win32' else '')))

def shortest(net,start,end,forbidden=()):
    forbidden=set(forbidden)-{start,end}; best={start:0.}; heap=[(0.,start)]; prev={}
    while heap:
        dist,cur=heapq.heappop(heap)
        if dist!=best[cur]:continue
        if cur==end:
            route=[cur]
            while cur in prev:cur=prev[cur];route.append(cur)
            return route[::-1],dist
        for nxt in net.getEdge(cur).getOutgoing():
            nid=nxt.getID()
            if nid in forbidden or not nxt.allows('passenger'):continue
            nd=dist+nxt.getLength()/nxt.getSpeed()
            if nd<best.get(nid,float('inf')):best[nid]=nd;prev[nid]=cur;heapq.heappush(heap,(nd,nid))
    return None,float('inf')

def via(net,start,middle,end,allgates):
    a,_=shortest(net,start,middle,set(allgates)-{middle})
    b,_=shortest(net,middle,end,set(allgates)-{middle})
    if a is None or b is None:return None
    return a+b[1:]

def build(reference):
    out=ROOT/'network';out.mkdir(parents=True,exist_ok=True)
    if (reference/'source.osm').resolve()!=(out/'source.osm').resolve():shutil.copy2(reference/'source.osm',out/'source.osm')
    raw=ET.parse(out/'source.osm');r=raw.getroot()
    for node in r.findall('node'):
        if node.get('id') not in GATE_NODES.values():continue
        for tag in list(node.findall('tag')):
            if tag.get('k')=='highway':node.remove(tag)
        ET.SubElement(node,'tag',k='highway',v='traffic_signals')
    raw.write(out/'scenario_metering.osm',encoding='utf-8',xml_declaration=True)
    # Preserve gate nodes by explicitly introducing scenario-only signal points.
    command=[exe('netconvert'),'--osm-files',str(out/'scenario_metering.osm'),'-o',str(out/'campus.net.xml'),'--geometry.remove','--roundabouts.guess','--ramps.guess','--junctions.join','--keep-edges.by-vclass','passenger','--no-turnarounds','true']
    log=subprocess.run(command,capture_output=True,text=True);(out/'import_log.txt').write_text(log.stdout+log.stderr);log.check_returncode()
    net=sumolib.net.readNet(str(out/'campus.net.xml'));gates={};gateedges=[]
    for key,nodeid in GATE_NODES.items():
        node=net.getNode(nodeid)
        edges=[e for e in node.getIncoming() if e.allows('passenger') and e.getID().startswith(GATE_PARENT[key])]
        if not edges:raise ValueError('No gate input edge '+key)
        e=max(edges,key=lambda x:x.getLength());gateedges.append(e.getID())
        gates[key]={'node':nodeid,'edge':e.getID(),'lanes':[l.getID() for l in e.getLanes() if l.allows('passenger')],'position':node.getCoord(),'length':e.getLength(),'hypothetical_signal':True}
    # Choose named geographic zones, not claimed official building assignments.
    zone_candidates={}
    for name,(lon,lat) in ZONE_POINTS.items():
        x,y=net.convertLonLat2XY(lon,lat)
        eligible=[(e,d) for e,d in net.getNeighboringEdges(x,y,350) if e.allows('passenger') and e.getLength()>40 and e.getType() in ['highway.residential','highway.service','highway.unclassified']]
        zone_candidates[name]=[e.getID() for e,d in sorted(eligible,key=lambda q:q[1])[:12]]
    origins={'NORTH_WEST':{'in':'1374163589#1','out':'136139470#1'},'HIGHWAY_15':{'in':'1374163588#1','out':'132323973#0'}}
    # Stable OSM way segments may split during import: choose by prefix and proximity.
    def resolve(prefix,lon,lat):
        edges=[e for e in net.getEdges() if e.getID().startswith(prefix) and e.allows('passenger')]
        if not edges:raise ValueError('No origin prefix '+prefix)
        xy=net.convertLonLat2XY(lon,lat)
        return min(edges,key=lambda e:np.linalg.norm(np.array(e.getShape()[0])-xy)).getID()
    routes={};zones={}
    def possible(z,endpoint,direction):
        result={}
        for origin,ends in origins.items():
            for gate in [1,2]:
                start=ends['in'] if direction=='IN' else endpoint
                end=endpoint if direction=='IN' else ends['out']
                keys=['G1_IN_A','G1_IN_B'] if direction=='IN' and gate==1 else [f'G{gate}_{direction}']
                options=[(key,via(net,start,gates[key]['edge'],end,gateedges)) for key in keys]
                options=[(k,p) for k,p in options if p]
                if not options:return None
                key,route=min(options,key=lambda kp:sum(net.getEdge(e).getLength()/net.getEdge(e).getSpeed() for e in kp[1]))
                rid=f'{origin}_{z}_{direction}_G{gate}';result[rid]={'edges':route,'seconds':sum(net.getEdge(e).getLength()/net.getEdge(e).getSpeed() for e in route),'origin':origin,'zone':z,'direction':direction,'gate':gate,'gate_key':key}
        return result
    for z,candidates in zone_candidates.items():
        inc=outc=None
        for edge in candidates:
            if inc is None:
                a=possible(z,edge,'IN')
                if a is not None:inc=(edge,a)
            if outc is None:
                a=possible(z,edge,'OUT')
                if a is not None:outc=(edge,a)
        if inc is None or outc is None:raise ValueError(f'No connected zone {z}; candidates {candidates}')
        routes.update(inc[1]);routes.update(outc[1]);zones[z]={'in_edge':inc[0],'out_edge':outc[0],'reference_lonlat':ZONE_POINTS[z],'assignment':'synthetic zone, not official building mapping'}
    # Copy geographic building polygons and place useful labels at actual gate control points.
    polygon_source=reference/'campus_areas.add.xml'
    if not polygon_source.exists():polygon_source=reference/'campus.add.xml'
    add=ET.parse(polygon_source).getroot()
    for el in list(add.findall('poi')):add.remove(el)
    for key,g in gates.items():
        x,y=g['position'];ET.SubElement(add,'poi',id=key,name=key,x=str(x),y=str(y),color='0.85,0.2,0.12' if key.startswith('G1') else '0.1,0.55,0.75',width='12',height='12',layer='5')
    for z,info in zones.items():
        e=net.getEdge(info['in_edge']);x,y=e.getShape()[-1];ET.SubElement(add,'poi',id=z,name=z,x=str(x),y=str(y),color='0.18,0.6,0.3',width='12',height='12',layer='5')
    ET.ElementTree(add).write(out/'campus.add.xml',encoding='utf-8',xml_declaration=True)
    root=ET.Element('routes');ET.SubElement(root,'vType',id='car',vClass='passenger',accel='2.6',decel='4.5',sigma='0.5',length='5',minGap='2.5',maxSpeed='25',color='0.12,0.53,0.9')
    for rid,info in routes.items():ET.SubElement(root,'route',id=rid,edges=' '.join(info['edges']))
    ET.ElementTree(root).write(out/'routes_catalog.rou.xml',encoding='utf-8',xml_declaration=True)
    # Fixed illustrative gate processing programs. Not existing university traffic signals.
    data={'gates':gates,'zones':zones,'origins':origins,'routes':routes,'route_count':len(routes),'source':'OpenStreetMap contributors, ODbL','synthetic_changes':'5 hypothetical metering signal nodes, not field observations'}
    (out/'catalog.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='routes'},indent=2))
    return data

if __name__=='__main__':build(Path(sys.argv[1]))
