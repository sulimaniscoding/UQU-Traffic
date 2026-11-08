"""Academic demand -> geographic SUMO runs -> RF/XGBoost -> constrained routing.

Synthetic research experiment, not university field data or a calibrated digital twin.
"""
from pathlib import Path
import argparse, collections, contextlib, gzip, hashlib, heapq, json, os, shutil, sys, time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
from network_setup import ROOT,exe

SEED=20261027
DAYS=['Sunday','Monday','Tuesday','Wednesday','Thursday']
BEGIN=6*3600+30*60
END=21*3600
HORIZON=900
CONTROL_VERSION='capacity_guarded_v2'
FEATURES=['minute','day','gate','direction_out','queue','queue_lag5','queue_lag10',
          'other_queue','speed','vehicles','flow_last5','academic_next15','academic_next30',
          'network_halting','green_fraction','signal_green','incident']

def save(df,path):
    path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False,encoding='utf-8-sig')

def clock(t):return f'{int(t)//3600:02}:{int(t)%3600//60:02}'

def catalog():return json.loads((ROOT/'network/catalog.json').read_text())

def prepare(academic_source):
    """Preserve raw normalized tables, derive one row per physical student-day."""
    dst=ROOT/'data/academic';dst.mkdir(parents=True,exist_ok=True)
    for name in ['students','courses','sections','section_meetings','student_enrollments','student_timetables']:
        shutil.copy2(academic_source/(name+'.csv'),dst/(name+'.csv'))
    students=pd.read_csv(dst/'students.csv',dtype=str)
    assert len(students)==24500 and students.Student_ID.is_unique
    assert (students.Campus=='العابدية - طلاب').all() and (students.Gender=='Male').all()
    tt=pd.read_csv(dst/'student_timetables.csv',dtype=str)
    physical=tt[tt.Delivery_Mode=='In-person'].copy()
    for col in ['Start_Time','End_Time']:physical[col]=physical[col].str[:2].astype(int)*60+physical[col].str[3:5].astype(int)
    ordered=physical.sort_values(['Student_ID','Day','Start_Time','Section_ID'])
    sd=ordered.groupby(['Student_ID','Day'],as_index=False).agg(First_Class=('Start_Time','first'),Last_Class=('End_Time','max'),First_Building=('Building','first'),Last_Building=('Building','last'),Meeting_Count=('Course_Code','size'))
    zones=list(catalog()['zones']);buildings=sorted(physical.Building.unique())
    # Explicit synthetic mapping, not a claim about real building numbers or occupants.
    mapping={b:zones[i%len(zones)] for i,b in enumerate(buildings)}
    save(pd.DataFrame([{'Academic_Building':b,'Simulation_Zone':z,'Provenance':'Synthetic assignment, not official campus mapping'} for b,z in mapping.items()]),ROOT/'data/building_zone_mapping.csv')
    sd['First_Zone']=sd.First_Building.map(mapping);sd['Last_Zone']=sd.Last_Building.map(mapping)
    sd['Day_Index']=sd.Day.map(dict(zip(DAYS,range(5))))
    save(sd,ROOT/'data/student_days.csv')
    rows=[]
    for day in range(5):
        a=sd[sd.Day_Index==day]
        for minute in range(360,1260,15):
            for z in zones:
                rows.append(dict(Day=DAYS[day],Minute=minute,Time=clock(minute*60),Zone=z,
                    First_Class_Students=int(((a.First_Zone==z)&(a.First_Class>=minute)&(a.First_Class<minute+15)).sum()),
                    Last_Class_Students=int(((a.Last_Zone==z)&(a.Last_Class>=minute)&(a.Last_Class<minute+15)).sum())))
    save(pd.DataFrame(rows),ROOT/'data/academic_demand_15min.csv')
    print('Academic data ready:',len(students),'students,',len(sd),'physical student-days',flush=True)

def route_id(origin,zone,direction,gate):return f'{origin}_{zone}_{direction}_G{gate}'

def demand(run,day,driver_probability,preference,late_shift=0,output_dir=None,label=''):
    output_dir=output_dir or ROOT/'runs';output_dir.mkdir(parents=True,exist_ok=True)
    stem=f'demand_{run:03}'+label
    rng=np.random.default_rng(SEED+run);sd=pd.read_csv(ROOT/'data/student_days.csv');sd=sd[sd.Day_Index==day]
    selected=sd[rng.random(len(sd))<driver_probability].copy()
    origins=list(catalog()['origins']);rows=[]
    for r in selected.itertuples():
        origin=origins[int(rng.random()>.65)]
        # One paired entry/exit vehicle journey. Parking between legs is implicit, not simulated.
        in_t=r.First_Class*60-rng.normal(32*60+late_shift,10*60)
        out_t=r.Last_Class*60+rng.uniform(5*60,20*60)
        for direction,t,zone in [('IN',in_t,r.First_Zone),('OUT',out_t,r.Last_Zone)]:
            gate=1 if rng.random()<preference else 2
            rows.append(dict(Vehicle_ID=f'{r.Student_ID}_{direction}',Student_ID=r.Student_ID,Day=DAYS[day],Direction=direction,Origin=origin,Zone=zone,Depart_Second=round(max(BEGIN+5,t),2),Preferred_Gate=gate,Choice_U=float(rng.random()),Base_Route=route_id(origin,zone,direction,gate)))
    df=pd.DataFrame(rows).sort_values(['Depart_Second','Vehicle_ID']).reset_index(drop=True)
    assert df.Vehicle_ID.is_unique and len(df)%2==0
    save(df,output_dir/f'{stem}.csv')
    root=ET.Element('routes')
    # Exit legs are released dynamically, only after the paired entry has arrived.
    # This prevents an artificial exit from a student still waiting outside campus.
    for r in df[df.Direction=='IN'].itertuples():
        ET.SubElement(root,'vehicle',id=r.Vehicle_ID,type='car',route=r.Base_Route,depart=f'{r.Depart_Second:.2f}',departLane='best',departSpeed='max',color='0.12,0.52,0.9' if r.Direction=='IN' else '0.95,0.58,0.1')
    p=output_dir/f'{stem}.rou.xml';ET.ElementTree(root).write(p,encoding='utf-8',xml_declaration=True)
    digest=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return df,p,digest

def groups_and_monitors(cat):
    """Extend short exit approaches upstream; assign each road to at most one monitor."""
    import sumolib
    net=sumolib.net.readNet(str(ROOT/'network/campus.net.xml'))
    groups={'G1_IN':['G1_IN_A','G1_IN_B'],'G2_IN':['G2_IN'],'G1_OUT':['G1_OUT'],'G2_OUT':['G2_OUT']}
    distances={};gate_edges={v['edge'] for v in cat['gates'].values()}
    for route in cat['routes'].values():
        key=route['gate_key'];group=key.replace('_A','').replace('_B','')
        i=route['edges'].index(cat['gates'][key]['edge']);back=0.
        for eid in route['edges'][:i+1][::-1]:
            if eid in gate_edges and eid!=cat['gates'][key]['edge']:break
            distances[(group,eid)]=min(back,distances.get((group,eid),float('inf')))
            back+=net.getEdge(eid).getLength()
            if back>=400:break
    assignment={}
    for (group,eid),dist in distances.items():
        if eid not in assignment or dist<assignment[eid][1]:assignment[eid]=(group,dist)
    monitors={g:[eid for eid,(group,dist) in assignment.items() if group==g] for g in groups}
    return groups,monitors

def add_detectors(cat,path,run_name,output_dir=None):
    output_dir=output_dir or ROOT/'runs'
    root=ET.Element('additional');detectors=collections.defaultdict(list)
    for key,g in cat['gates'].items():
        group=key.replace('_A','').replace('_B','')
        for i,lane in enumerate(g['lanes']):
            det=f'D_{key}_{i}';detectors[group].append(det)
            ET.SubElement(root,'inductionLoop',id=det,lane=lane,pos='-1',period='60',file=str(output_dir/f'{run_name}.detectors.xml'))
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return detectors

def params(run):
    rng=np.random.default_rng(SEED+run*17)
    return dict(run=run,day=run%5,driver_probability=float(rng.uniform(.14,.44)),preference=float(rng.uniform(.52,.94)),g1green=float(rng.uniform(.28,.68)),g2green=float(rng.uniform(.48,.82)),outgreen=float(rng.uniform(.50,.85)),incident=run%5==0,late_shift=int(rng.uniform(-5,5)*60))

class LivePanel:
    def __init__(self,policy):
        import tkinter as tk
        self.tk=tk;self.root=tk.Tk();self.root.title('UQU Traffic Lab — '+policy);self.root.geometry('710x420');self.root.configure(bg='#102237');self.closed=False
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        tk.Label(self.root,text='UQU TRAFFIC LAB',font=('Segoe UI',22,'bold'),fg='white',bg='#102237').pack(pady=10)
        self.title=tk.Label(self.root,text='Starting geographic SUMO simulation',font=('Segoe UI',12),fg='#83d9e0',bg='#102237');self.title.pack()
        self.text=tk.Label(self.root,font=('Consolas',13),fg='white',bg='#102237',justify='left');self.text.pack(pady=15)
        self.note=tk.Label(self.root,text='Synthetic demand · OSM road geometry · Forecast horizon 15 minutes',font=('Segoe UI',10),fg='#becbd5',bg='#102237',wraplength=680);self.note.pack(pady=8)
        self.root.update()
    def close(self):self.closed=True;self.root.destroy()
    def update(self,t,states,pred,share,policy):
        if self.closed:return
        self.title.config(text=f'{clock(t)}  |  {policy.upper()}  |  Green: hypothetical gate processing')
        lines=['Gate / direction   Queue now   Forecast +15m    Gate 1 share']
        for group,row in states.items():lines.append(f"{group:17} {row['queue']:>5}      {pred.get(group,0):>8.1f}         {share[row['direction']]:.0%}")
        self.text.config(text='\n'.join(lines));self.root.update_idletasks();self.root.update()

def simulate(p,policy='baseline',model=None,gui=False,panel=False,capture=False,stop_at=END):
    # Four-row live predictions do not benefit from nested joblib threads.
    if model is not None:model.set_params(n_jobs=1)
    if gui:
        import traci as sim
    else:
        try:import libsumo as sim
        except ImportError:import traci as sim
    run=p['run'];name=f'{run:03}_{policy}';cat=catalog();groups,monitors=groups_and_monitors(cat)
    run_dir=ROOT/('live_runs' if gui else 'runs');run_dir.mkdir(parents=True,exist_ok=True)
    # Unique input files per policy avoid simultaneous route-file writes on Windows.
    trips,rou,digest=demand(run,p['day'],p['driver_probability'],p['preference'],p.get('late_shift',0),run_dir,'' if policy=='baseline' else '_'+policy)
    by_id=trips.set_index('Vehicle_ID').to_dict('index')
    detectors=add_detectors(cat,run_dir/f'{name}.add.xml',name,run_dir)
    tripfile=run_dir/f'{name}.tripinfo.xml'
    command=[exe('sumo-gui' if gui else 'sumo'),'-n',str(ROOT/'network/campus.net.xml'),'-r',str(ROOT/'network/routes_catalog.rou.xml')+','+str(rou),'-a',str(ROOT/'network/campus.add.xml')+','+str(run_dir/f'{name}.add.xml'),'--begin',str(BEGIN),'--seed',str(SEED+run),'--step-length','1','--time-to-teleport','-1','--collision.action','warn','--collision.mingap-factor','0','--collision-output',str(run_dir/f'{name}.collisions.xml'),'--no-step-log','true','--duration-log.disable','true','--tripinfo-output',str(tripfile),'--tripinfo-output.write-unfinished','true','--no-warnings','true','--xml-validation','never','--error-log',str(run_dir/f'{name}.warnings.log')]
    if gui:command+=['--start','--delay','0','--quit-on-end','false']
    sim.start(command)
    for key,g in cat['gates'].items():
        green=p['outgreen'] if key.endswith('OUT') else p['g1green'] if key.startswith('G1') else p['g2green']
        original=sim.trafficlight.getAllProgramLogics(g['node'])[0]
        size=len(original.phases[0].state);seconds=round(90*green)
        phases=[sim.trafficlight.Phase(seconds,'G'*size),sim.trafficlight.Phase(3,'y'*size),sim.trafficlight.Phase(90-seconds-3,'r'*size)]
        sim.trafficlight.setProgramLogic(g['node'],sim.trafficlight.Logic('synthetic_metering',0,0,phases))
    if gui:
        pos=cat['gates']['G1_IN_A']['position'];other=cat['gates']['G2_IN']['position']
        try:sim.gui.setOffset('View #0',(pos[0]+other[0])/2,(pos[1]+other[1])/2);sim.gui.setZoom('View #0',450)
        except Exception:pass
    dash=LivePanel(policy) if panel else None
    sd=pd.read_csv(ROOT/'data/student_days.csv');sd=sd[sd.Day_Index==p['day']]
    first=sd.First_Class.to_numpy();last=sd.Last_Class.to_numpy()
    history=collections.defaultdict(list);flows=collections.defaultdict(list);rows=[];actions=[];frames=[]
    share={'IN':p['preference'],'OUT':p['preference']};pred={g:0. for g in groups};states={}
    departed=arrived=teleports=collisions=redirected=0;actual_crossed=collections.Counter()
    exit_release=[];actual_departure={};entry_arrival={}
    monitored_free_speeds={e:sim.lane.getMaxSpeed(e+'_0') for edges in monitors.values() for e in edges}
    (ROOT/'network/queue_monitors.json').write_text(json.dumps(monitors,indent=2))
    reference_speeds={lane:sim.lane.getMaxSpeed(lane) for key,g in cat['gates'].items() if key.startswith('G1_IN') for lane in g['lanes']}
    tick=time.monotonic()
    try:
        for t in range(BEGIN+1,stop_at+1):
            blocked=bool(p['incident'] and 8*3600<=t<8*3600+20*60)
            if t in [BEGIN+1,8*3600,8*3600+20*60]:
                for lane,speed in reference_speeds.items():sim.lane.setMaxSpeed(lane,1.8 if blocked else speed)
            sim.simulationStep()
            new=sim.simulation.getDepartedIDList();departed+=len(new)
            for vid in new:actual_departure[vid]=t
            arrived_ids=sim.simulation.getArrivedIDList();arrived+=len(arrived_ids)
            for vid in arrived_ids:
                if vid.endswith('_IN'):
                    entry_arrival[vid]=t
                    out_id=vid[:-3]+'_OUT'
                    heapq.heappush(exit_release,(max(t,by_id[out_id]['Depart_Second']),out_id))
            while exit_release and exit_release[0][0]<=t:
                _,out_id=heapq.heappop(exit_release);rec=by_id[out_id]
                sim.vehicle.add(out_id,rec['Base_Route'],typeID='car',depart='now',departLane='best',departSpeed='max')
                sim.vehicle.setColor(out_id,(242,148,26,255))
            teleports+=sim.simulation.getStartingTeleportNumber();collisions+=sim.simulation.getCollidingVehiclesNumber()
            if policy!='baseline':
                for vid in new:
                    rec=by_id[vid]
                    # Common random numbers: with the unchanged initial share,
                    # every vehicle keeps exactly its baseline preferred gate.
                    # The extra random value ranks drivers within that preference.
                    threshold=(rec['Choice_U']*p['preference'] if rec['Preferred_Gate']==1
                               else p['preference']+rec['Choice_U']*(1-p['preference']))
                    g=1 if threshold<share[rec['Direction']] else 2
                    chosen=route_id(rec['Origin'],rec['Zone'],rec['Direction'],g)
                    if chosen!=rec['Base_Route']:
                        sim.vehicle.setRouteID(vid,chosen);redirected+=1
            if t%60:continue
            q={group:sum(sim.edge.getLastStepHaltingNumber(e) for e in edges) for group,edges in monitors.items()}
            network_halting=sum(sim.edge.getLastStepHaltingNumber(e) for e in sim.edge.getIDList() if not e.startswith(':'))
            minute=t/60
            for group,keys in groups.items():
                gate=int(group[1]);direction=group.split('_')[1];edges=monitors[group]
                hist=history[group];hist.append(q[group])
                flow=sum(sim.inductionloop.getLastIntervalVehicleNumber(d) for d in detectors[group]);flows[group].append(flow);actual_crossed[group]+=flow
                counts=[sim.edge.getLastStepVehicleNumber(e) for e in edges]
                speeds=[sim.edge.getLastStepMeanSpeed(e) for e in edges];total=sum(counts)
                speed=sum(c*s for c,s in zip(counts,speeds))/total if total else np.mean([monitored_free_speeds[e] for e in edges])
                a=first if direction=='IN' else last
                row=dict(run=run,policy=policy,second=t,time=clock(t),minute=minute,day=p['day'],gate=gate,direction=direction,direction_out=int(direction=='OUT'),queue=q[group],queue_lag5=hist[-6] if len(hist)>=6 else hist[0],queue_lag10=hist[-11] if len(hist)>=11 else hist[0],other_queue=q[f'G{3-gate}_{direction}'],speed=float(speed),vehicles=total,flow_last5=sum(flows[group][-5:]),academic_next15=int(((a>minute)&(a<=minute+15)).sum()),academic_next30=int(((a>minute)&(a<=minute+30)).sum()),network_halting=network_halting,green_fraction=p['outgreen'] if direction=='OUT' else p['g1green'] if gate==1 else p['g2green'],signal_green=float(np.mean(['G' in sim.trafficlight.getRedYellowGreenState(cat['gates'][key]['node']) for key in keys])),incident=int(blocked and group=='G1_IN'),flow_last1=flow)
                states[group]=row
            if model is not None:
                values=np.maximum(0,model.predict(pd.DataFrame(states.values())[FEATURES]))
                pred=dict(zip(states,values.astype(float)))
            else:pred={group:float(q[group]) for group in groups}
            if policy!='baseline' and t%300==0:
                for direction in ['IN','OUT']:
                    values=pred if policy=='predictive' else q
                    # Estimate queue-clearance delay plus the mean alternative-route detour.
                    costs={}
                    for gate in [1,2]:
                        keys=groups[f'G{gate}_{direction}'];lanes=sum(len(cat['gates'][k]['lanes']) for k in keys)
                        green=p['outgreen'] if direction=='OUT' else p['g1green'] if gate==1 else p['g2green']
                        ff=np.mean([r['seconds'] for r in cat['routes'].values() if r['direction']==direction and r['gate']==gate])
                        costs[gate]=ff+values[f'G{gate}_{direction}']*2.2/max(.1,lanes*green)
                    before=share[direction]
                    capacities={gate:sum(len(cat['gates'][k]['lanes']) for k in groups[f'G{gate}_{direction}'])*(p['outgreen'] if direction=='OUT' else p['g1green'] if gate==1 else p['g2green']) for gate in [1,2]}
                    capacity_share=capacities[1]/sum(capacities.values())
                    lower=max(.55,capacity_share-.10)
                    # Check receiving links rather than pushing traffic into a blocked gate.
                    def receiving_blocked(gate):
                        downstream=set()
                        for route in cat['routes'].values():
                            if route['gate']==gate and route['direction']==direction:
                                i=route['edges'].index(cat['gates'][route['gate_key']]['edge'])
                                downstream.update(route['edges'][i+1:i+3])
                        return any(sim.edge.getLastStepVehicleNumber(e)>=4 and sim.edge.getLastStepMeanSpeed(e)<1.0 and sim.edge.getLastStepOccupancy(e)>60 for e in downstream)
                    # The free-flow detour alone must not trigger redistribution on an empty road.
                    if costs[1]>costs[2]+25 and values[f'G1_{direction}']>=12 and not receiving_blocked(2):share[direction]=max(lower,before-.05)
                    elif costs[2]>costs[1]+25 and values[f'G2_{direction}']>=12 and not receiving_blocked(1):share[direction]=min(.90,before+.05)
                    actions.append(dict(second=t,time=clock(t),direction=direction,old_share=before,new_share=share[direction],gate1_cost_seconds=costs[1],gate2_cost_seconds=costs[2],prediction_gate1=pred[f'G1_{direction}'],prediction_gate2=pred[f'G2_{direction}']))
            for group,row in states.items():rows.append(dict(row,forecast_15min=pred[group],gate1_share=share[row['direction']]))
            if dash:dash.update(t,states,pred,share,policy)
            if capture and t%120==0:
                ids=sim.vehicle.getIDList()
                # All active vehicles, not a hand-picked visual sample.
                positions=[[vid,*[round(v,1) for v in sim.vehicle.getPosition(vid)],by_id[vid]['Direction']] for vid in ids]
                frames.append(dict(second=t,time=clock(t),departed=departed,arrived=arrived,pending=int((trips.Depart_Second<=t).sum())-departed,states=list({**x,'forecast_15min':pred[g],'gate1_share':share[x['direction']]} for g,x in states.items()),vehicles=positions))
            if gui and t%900==0:print(f'{clock(t)}: departed {departed}, arrived {arrived}, queues {q}',flush=True)
    finally:
        sim.close()
        if dash and not dash.closed:dash.close()
    data=pd.DataFrame(rows);save(data,run_dir/f'{name}.csv')
    if actions:save(pd.DataFrame(actions),(run_dir if gui else ROOT/'results')/f'actions_{name}.csv')
    if frames:
        replay_path=ROOT/'replay'/f'{name}.json.gz';temporary=replay_path.with_name(f'.{name}.{os.getpid()}.tmp')
        with gzip.open(temporary,'wt',encoding='utf-8') as f:json.dump(frames,f,separators=(',',':'))
        temporary.replace(replay_path)
    info=ET.parse(tripfile).getroot().findall('tripinfo')
    inserted=[x for x in info if float(x.get('depart','-1'))>=0]
    complete=[x for x in inserted if float(x.get('arrival','-1'))>=0]
    waits=sum(float(x.get('waitingTime','0')) for x in inserted)
    # Include time held for an unfinished entry, not just SUMO insertion delay.
    insertion_delay=sum(max(0,float(x.get('depart'))-by_id[x.get('id')]['Depart_Second']) for x in inserted)
    not_inserted=trips[~trips.Vehicle_ID.isin([x.get('id') for x in inserted])]
    # Demand not yet inserted contributes its waiting since its planned departure.
    pending_wait=float(np.maximum(0,stop_at-not_inserted.Depart_Second.to_numpy()).sum())
    impossible_exits=sum(vid.endswith('_OUT') and (vid[:-4]+'_IN' not in entry_arrival or start<entry_arrival[vid[:-4]+'_IN']) for vid,start in actual_departure.items())
    assert impossible_exits==0
    summary=dict(run=run,policy=policy,control_version=CONTROL_VERSION if policy!='baseline' else 'fixed_baseline',demand_sha256=digest,planned=len(trips),inserted=len(inserted),completed=len(complete),unfinished=len(inserted)-len(complete),not_inserted=len(not_inserted),mean_wait_inserted_seconds=waits/max(1,len(inserted)),total_queue_and_insertion_wait_seconds=waits+insertion_delay+pending_wait,mean_delay_per_planned_seconds=(waits+insertion_delay+pending_wait)/max(1,len(trips)),mean_network_halting=float(data.groupby('second').network_halting.first().mean()),max_monitored_queue=int(data.queue.max()),teleports=teleports,collisions=collisions,redirected=redirected,exit_before_entry_arrived=impossible_exits,wall_seconds=time.monotonic()-tick)
    for group,count in actual_crossed.items():summary['crossed_'+group]=count
    for direction in ['IN','OUT']:
        summary['completed_'+direction]=sum(x.get('id','').endswith('_'+direction) for x in complete)
    (run_dir/f'{name}.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'{name}: {len(trips)} legs, {len(complete)} completed, {len(not_inserted)} pending, {summary["wall_seconds"]:.1f}s',flush=True)
    return data,summary

def train():
    raw=pd.concat([pd.read_csv(ROOT/'runs'/f'{i:03}_baseline.csv') for i in range(30)],ignore_index=True)
    raw=raw.sort_values(['run','gate','direction','second'])
    raw['target_queue_15min']=raw.groupby(['run','gate','direction']).queue.shift(-15)
    raw['target_second']=raw.second+900;raw=raw.dropna(subset=['target_queue_15min'])
    raw['split']=np.where(raw.run<18,'train',np.where(raw.run<24,'validation','test'))
    save(raw,ROOT/'data/training_dataset.csv')
    a=raw[raw.split=='train'];v=raw[raw.split=='validation'];test=raw[raw.split=='test'];metrics=[];models={};search=[]
    for name in ['RandomForest','XGBoost']:
        candidates=[]
        for depth in [6,12]:
            model=RandomForestRegressor(n_estimators=160,max_depth=depth,min_samples_leaf=8,max_features=.85,random_state=SEED,n_jobs=2) if name=='RandomForest' else XGBRegressor(n_estimators=260,max_depth=4 if depth==6 else 7,learning_rate=.045,min_child_weight=8,subsample=.85,colsample_bytree=.85,reg_lambda=8,random_state=SEED,n_jobs=2,objective='reg:squarederror',tree_method='hist')
            print(f'Training {name}, depth {model.get_params()["max_depth"]}, {len(a)} training rows...',flush=True)
            model.fit(a[FEATURES],a.target_queue_15min)
            val=mean_absolute_error(v.target_queue_15min,np.maximum(0,model.predict(v[FEATURES])))
            search.append(dict(model=name,max_depth=model.get_params()['max_depth'],validation_MAE=val));candidates.append((val,model))
            print(f'  Validation MAE: {val:.3f}',flush=True)
        _,model=min(candidates,key=lambda t:t[0]);models[name]=model;joblib.dump(model,ROOT/'models'/f'{name}.joblib')
        if name=='XGBoost':model.save_model(ROOT/'models'/'XGBoost.json')
        save(pd.DataFrame({'feature':FEATURES,'importance':model.feature_importances_}).sort_values('importance',ascending=False),ROOT/'results'/f'{name}_importance.csv')
    predictions=test[['run','second','time','gate','direction','queue','target_queue_15min']].copy()
    for split,f in [('validation',v),('test',test)]:
        for name in ['Persistence',*models]:
            p=f.queue.to_numpy() if name=='Persistence' else np.maximum(0,models[name].predict(f[FEATURES]))
            metrics.append(dict(model=name,split=split,MAE=mean_absolute_error(f.target_queue_15min,p),RMSE=np.sqrt(mean_squared_error(f.target_queue_15min,p)),rows=len(f)))
            if split=='test':predictions[name]=p
    results=pd.DataFrame(metrics);save(results,ROOT/'results/model_metrics.csv');save(pd.DataFrame(search),ROOT/'results/hyperparameter_search.csv');save(predictions,ROOT/'results/test_predictions.csv')
    winner=results[(results.split=='validation')&(results.model!='Persistence')].sort_values('MAE').iloc[0].model
    detailed=[]
    for (gate,direction),f in predictions.groupby(['gate','direction']):
        for subset,mask in [('all',np.ones(len(f),dtype=bool)),('busy_target_ge_10',f.target_queue_15min>=10)]:
            b=f[mask]
            if b.empty:continue
            for name in ['Persistence',*models]:
                detailed.append(dict(gate=gate,direction=direction,subset=subset,model=name,rows=len(b),MAE=mean_absolute_error(b.target_queue_15min,b[name]),RMSE=np.sqrt(mean_squared_error(b.target_queue_15min,b[name]))))
    save(pd.DataFrame(detailed),ROOT/'results/model_metrics_by_stream.csv')
    # Diagnostic only. The full-feature winner was selected before this ablation.
    reduced=[x for x in FEATURES if not x.startswith('academic_')]
    from sklearn.base import clone
    ablated=clone(models[winner]);ablated.fit(a[reduced],a.target_queue_15min)
    save(pd.DataFrame([dict(model=winner,feature_set='full',test_MAE=float(results[(results.split=='test')&(results.model==winner)].iloc[0].MAE)),dict(model=winner,feature_set='without_academic_counts',test_MAE=mean_absolute_error(test.target_queue_15min,np.maximum(0,ablated.predict(test[reduced]))))]),ROOT/'results/academic_ablation.csv')
    (ROOT/'models/metadata.json').write_text(json.dumps(dict(seed=SEED,features=FEATURES,target='monitored stopped vehicles at t+900 seconds',selected_by_validation=winner,train_runs=list(range(18)),validation_runs=list(range(18,24)),test_runs=list(range(24,30)),begin=BEGIN,end=END,network_sha256=hashlib.sha256((ROOT/'network/campus.net.xml').read_bytes()).hexdigest()),indent=2))
    print(results.to_string(index=False),flush=True);return models[winner]

def demo_params(scenario):
    return dict(run={'normal':100,'peak':101,'incident':102}[scenario],day=0,driver_probability={'normal':.20,'peak':.38,'incident':.30}[scenario],preference=.90,g1green=.40,g2green=.70,outgreen=.65,incident=scenario=='incident',late_shift=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--train',action='store_true');ap.add_argument('--collect',action='store_true');ap.add_argument('--workers',type=int,default=1);ap.add_argument('--compare',action='store_true');ap.add_argument('--resume-comparison',action='store_true');ap.add_argument('--demo',choices=['normal','peak','incident']);ap.add_argument('--policy',choices=['baseline','reactive','predictive'],default='predictive');ap.add_argument('--gui',action='store_true');ap.add_argument('--panel',action='store_true');args=ap.parse_args()
    for p in ['data','models','results','runs','replay']: (ROOT/p).mkdir(exist_ok=True)
    if args.prepare:prepare(args.prepare)
    if args.smoke:
        p=demo_params('normal');p['run']=999
        simulate(p,stop_at=9*3600);return
    if args.collect:
        pending=[params(i) for i in range(30) if not (ROOT/'runs'/f'{i:03}_baseline.summary.json').exists()]
        if args.workers>1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(collect_one,pending))
        else:
            for p in pending:collect_one(p)
        save(pd.DataFrame([params(run) for run in range(30)]),ROOT/'data/scenario_manifest.csv')
    if args.train:train()
    if args.compare:
        tasks=[(scenario,policy) for scenario in ['normal','peak','incident'] for policy in ['baseline','reactive','predictive']]
        cached=[]
        if args.resume_comparison:
            pending=[]
            for scenario,policy in tasks:
                run=demo_params(scenario)['run'];name=f'{run:03}_{policy}'
                path=ROOT/'runs'/f'{name}.summary.json';replay=ROOT/'replay'/f'{name}.json.gz'
                compatible=path.exists() and (policy=='baseline' or json.loads(path.read_text()).get('control_version')==CONTROL_VERSION)
                if compatible and replay.exists():
                    try:
                        with gzip.open(replay,'rt',encoding='utf-8') as f:compatible=len(json.load(f))==435
                    except (EOFError,OSError,ValueError):compatible=False
                if compatible and replay.exists() and len(pd.read_csv(ROOT/'runs'/f'{name}.csv'))==3480:
                    summary=json.loads(path.read_text());summary['scenario']=scenario;cached.append(summary);print('Resuming completed case',name,flush=True)
                else:pending.append((scenario,policy))
            tasks=pending
        if args.workers>1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:summaries=list(pool.map(compare_one,tasks))
        else:summaries=[compare_one(task) for task in tasks]
        summaries+=cached
        save(pd.DataFrame(summaries).sort_values(['run','policy']),ROOT/'results/control_comparison.csv')
    if args.demo:
        meta=json.loads((ROOT/'models/metadata.json').read_text());model=joblib.load(ROOT/'models'/f"{meta['selected_by_validation']}.joblib")
        simulate(demo_params(args.demo),args.policy,model,args.gui,args.panel,capture=False)

def collect_one(p):
    _,summary=simulate(p)
    return summary

def compare_one(task):
    scenario,policy=task
    meta=json.loads((ROOT/'models/metadata.json').read_text());model=joblib.load(ROOT/'models'/f"{meta['selected_by_validation']}.joblib")
    _,summary=simulate(demo_params(scenario),policy,model,capture=True)
    summary['scenario']=scenario
    return summary

if __name__=='__main__':main()
