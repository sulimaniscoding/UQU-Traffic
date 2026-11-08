"""Independent data checks, honest experiment summaries and reproducible figures."""
from pathlib import Path
import gzip, hashlib, importlib.metadata, json, platform
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection,PolyCollection
import sumolib,joblib
from experiment import ROOT,FEATURES,SEED,BEGIN,END,catalog,save

CHECKS=[]
def check(name,errors,detail):
    n=int(errors);CHECKS.append(dict(Validation_Check=name,Result='PASS' if n==0 else 'FAIL',Error_Count=n,Details=detail))

def academic_checks():
    p=ROOT/'data/academic';frames={n:pd.read_csv(p/f'{n}.csv',dtype=str,keep_default_na=False) for n in ['students','courses','sections','section_meetings','student_enrollments','student_timetables']}
    s,c,se,m,e,t=[frames[n] for n in frames]
    check('exact_abidiyah_student_count',abs(len(s)-24500),'24,500 synthetic male students, not an official enrollment count')
    check('unique_student_ids',s.Student_ID.duplicated().sum(),'Synthetic identifiers only')
    check('campus_and_gender',((s.Campus!='العابدية - طلاب')|(s.Gender!='Male')).sum(),'No other campus or gender in this project subset')
    for name,f,key in [('course',c,'Course_Code'),('section',se,'Section_ID'),('meeting',m,'Meeting_ID')]:check('unique_'+name+'_ids',f[key].duplicated().sum(),key)
    for name,left,right in [('enrollment_student',e.Student_ID,s.Student_ID),('enrollment_section',e.Section_ID,se.Section_ID),('section_course',se.Course_Code,c.Course_Code),('meeting_section',m.Section_ID,se.Section_ID),('timetable_student',t.Student_ID,s.Student_ID),('timetable_section',t.Section_ID,se.Section_ID)]:check('reference_'+name,(~left.isin(right)).sum(),'All foreign keys must resolve')
    check('no_duplicate_enrollments',e.duplicated(['Student_ID','Section_ID']).sum(),'Theory and lab are distinct linked activities, not duplicate registrations')
    en=e.merge(se[['Section_ID','Course_Code','Capacity']],on='Section_ID',suffixes=('','_section'),validate='many_to_one')
    check('enrollment_course_matches_section',(en.Course_Code!=en.Course_Code_section).sum(),'Course code agrees across normalized files')
    occupancy=e.groupby('Section_ID').Student_ID.nunique();capacity=se.set_index('Section_ID').Capacity.astype(int)
    check('section_capacity',int((occupancy>capacity.reindex(occupancy.index)).sum()),'Actual students per section do not exceed capacity')
    check('every_student_has_timetable',(~s.Student_ID.isin(t.Student_ID)).sum(),'All 24,500 students have at least one weekly meeting')
    check('every_enrollment_has_meetings',(~e.Section_ID.isin(m.Section_ID)).sum(),'All registered sections have defined weekly meetings')
    rebuilt=e[['Student_ID','Section_ID','Course_Code']].merge(m,on='Section_ID',validate='many_to_many')
    check('timetable_row_count_matches_join',abs(len(rebuilt)-len(t)),'One row per student and section meeting')
    keys=['Student_ID','Section_ID','Course_Code','Day','Start_Time','End_Time','Campus','Building','Room','Delivery_Mode']
    counts=pd.concat([rebuilt[keys].assign(_n=1),t[keys].assign(_n=-1)]).groupby(keys,dropna=False)._n.sum()
    check('timetable_exactly_matches_enrollment_meetings',(counts!=0).sum(),'Full multiset comparison, not just counts or a sample')
    sort=t.sort_values(['Student_ID','Day','Start_Time','End_Time']);end=sort.End_Time.str[:2].astype(int)*60+sort.End_Time.str[3:5].astype(int)
    start=sort.Start_Time.str[:2].astype(int)*60+sort.Start_Time.str[3:5].astype(int)
    prev=end.groupby([sort.Student_ID,sort.Day]).cummax().shift();prev[~sort.duplicated(['Student_ID','Day'])]=np.nan
    check('no_student_time_overlap',(start<prev).sum(),'Per student, per weekday; adjacent meetings may touch but not overlap')
    check('weekday_and_time_bounds',((~sort.Day.isin(['Sunday','Monday','Tuesday','Wednesday','Thursday']))|(start<480)|(end>1200)|(end<=start)).sum(),'Sunday–Thursday, 08:00–20:00')
    for name,f,cols in [('students',s,['Student_ID','College','Major','Campus']),('timetables',t,keys)]:check('no_missing_'+name,int((f[cols]=='').sum().sum()),'Required fields populated')
    physical=m[m.Delivery_Mode=='In-person'].sort_values(['Campus','Building','Room','Day','Start_Time'])
    st=physical.Start_Time.str[:2].astype(int)*60+physical.Start_Time.str[3:5].astype(int);en=physical.End_Time.str[:2].astype(int)*60+physical.End_Time.str[3:5].astype(int)
    prev=en.groupby([physical.Campus,physical.Building,physical.Room,physical.Day]).cummax().shift();prev[~physical.duplicated(['Campus','Building','Room','Day'])]=np.nan
    check('no_room_booking_overlap',(st<prev).sum(),'Room identity includes campus, building and room code')
    return {n:len(f) for n,f in frames.items()}

def experiment_checks():
    cat=catalog();net=sumolib.net.readNet(str(ROOT/'network/campus.net.xml'));bad=0;other=0;allgates={v['edge'] for v in cat['gates'].values()}
    for rid,r in cat['routes'].items():
        for a,b in zip(r['edges'],r['edges'][1:]):bad+=net.getEdge(b) not in net.getEdge(a).getOutgoing()
        expected=cat['gates'][r['gate_key']]['edge'];other+=sum(e in allgates for e in r['edges'])!=1 or expected not in r['edges']
    check('route_connectivity',bad,'40 precomputed passenger routes, using actual connected OSM road edges')
    check('exactly_one_modeled_gate_per_route',other,'Each route crosses its selected modeled gate and no other modeled gate')
    raw=pd.concat([pd.read_csv(ROOT/'runs'/f'{i:03}_baseline.csv') for i in range(30)],ignore_index=True)
    check('baseline_measurement_count',abs(len(raw)-30*870*4),'30 runs × 870 minutes × 4 gate/direction streams')
    check('unique_observation_keys',raw.duplicated(['run','second','gate','direction']).sum(),'One observation per simulated minute and stream')
    check('complete_time_series',sum((f.second.sort_values().diff().dropna()!=60).sum() for _,f in raw.groupby(['run','gate','direction'])),'One-minute spacing without missing observations')
    flow_errors=0;choice_errors=0
    for _,f in raw.sort_values('second').groupby(['run','gate','direction']):
        flow_errors+=np.count_nonzero(f.flow_last5.to_numpy()!=f.flow_last1.rolling(5,min_periods=1).sum().to_numpy())
    from experiment import params
    for run in range(30):
        trips=pd.read_csv(ROOT/'runs'/f'demand_{run:03}.csv');pref=params(run)['preference']
        u=np.where(trips.Preferred_Gate==1,trips.Choice_U*pref,pref+trips.Choice_U*(1-pref))
        choice_errors+=np.count_nonzero(np.where(u<pref,1,2)!=trips.Preferred_Gate.to_numpy())
    check('five_minute_flow_is_rolling_sum',flow_errors,'Every flow_last5 equals five completed one-minute readings')
    check('unchanged_share_preserves_initial_gate',choice_errors,'Common-random-number control preserves every baseline gate until the target share changes')
    train=pd.read_csv(ROOT/'data/training_dataset.csv');truth=raw.set_index(['run','gate','direction','second']).queue
    index=pd.MultiIndex.from_frame(train[['run','gate','direction','target_second']].rename(columns={'target_second':'second'}))
    actual=truth.reindex(index).to_numpy()
    check('target_exactly_15_minutes_later',int(np.isnan(actual).sum()+np.count_nonzero(actual!=train.target_queue_15min.to_numpy())),'Labels are observed SUMO queue counts at t + 900 seconds, not an invented equation')
    check('feature_values_finite',int((~np.isfinite(train[FEATURES].to_numpy())).sum()),'Only measurements available at t and known academic schedules are inputs')
    memberships=train.groupby('run').split.nunique();check('no_cross_split_run_leakage',(memberships!=1).sum(),'Runs 0–17 training, 18–23 validation, 24–29 test')
    check('training_row_count',abs(len(train)-102600),'102,600 labeled observations; final 15 minutes per stream are excluded')
    for name in ['RandomForest','XGBoost']:
        model=joblib.load(ROOT/'models'/f'{name}.joblib');pred=pd.read_csv(ROOT/'results/test_predictions.csv')
        test=train[train.split=='test'].sort_values(['run','gate','direction','second'])
        saved=pred.sort_values(['run','gate','direction','second'])[name].to_numpy();calc=np.maximum(0,model.predict(test[FEATURES]))
        check('saved_predictions_'+name,np.count_nonzero(~np.isclose(saved,calc,atol=1e-5,rtol=1e-5)),'Saved test predictions reproduce from the delivered model')
    summaries=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*.summary.json')) if not p.name.startswith('999')]
    detector_errors=0;detector_audit=[]
    lanes_per_group={}
    for key,g in cat['gates'].items():
        group=key.replace('_A','').replace('_B','');lanes_per_group[group]=lanes_per_group.get(group,0)+len(g['lanes'])
    for summary in summaries:
        name=f"{summary['run']:03}_{summary['policy']}"
        counts={}
        for x in ET.parse(ROOT/'runs'/f'{name}.detectors.xml').getroot().findall('interval'):
            group='_'.join(x.get('id').split('_')[1:3]);key=(int(float(x.get('end'))),group)
            counts[key]=counts.get(key,0)+int(x.get('nVehContrib'))
        measurements=pd.read_csv(ROOT/'runs'/f'{name}.csv')
        for r in measurements.itertuples():
            group=f'G{r.gate}_{r.direction}';full=counts.get((int(r.second),group),-1);difference=r.flow_last1-full
            # Native last-interval retrieval can include an occupant entering at
            # the interval boundary. XML nVehContrib counts fully passed cars.
            # A point detector has at most one non-overlapping occupant per lane.
            invalid=full<0 or difference<0 or difference>lanes_per_group[group]
            detector_errors+=invalid
            if difference or invalid:detector_audit.append(dict(run=summary['run'],policy=summary['policy'],second=r.second,stream=group,native_last_interval_count=r.flow_last1,xml_fully_passed=full,difference=difference,boundary_occupant_limit=lanes_per_group[group],consistent=not invalid))
    save(pd.DataFrame(detector_audit),ROOT/'results/detector_interval_audit.csv')
    check('live_flow_consistent_with_XML_boundary_semantics',detector_errors,'Native count must be between XML full passages and full passages plus one boundary occupant per point-detector lane; differences remain documented')
    check('vehicle_accounting',sum(s['completed']+s['unfinished']+s['not_inserted']!=s['planned'] for s in summaries),'Completed, unfinished and not-inserted legs add up to full planned demand')
    check('no_detected_on_edge_collisions',sum(s['collisions'] for s in summaries),'No detected events under the configured on-edge checks; not an intersection or field safety certificate')
    check('no_vehicle_teleportation',sum(s['teleports'] for s in summaries),'Teleporting is disabled; stranded traffic remains visible in the metrics')
    check('exit_follows_completed_entry',sum(s['exit_before_entry_arrived'] for s in summaries),'Every released exit follows arrival of that student’s paired entry')
    comp=pd.read_csv(ROOT/'results/control_comparison.csv')
    comp['policy']=pd.Categorical(comp.policy,categories=['baseline','reactive','predictive'],ordered=True)
    comp=comp.sort_values(['run','policy']);save(comp,ROOT/'results/control_comparison.csv')
    check('comparison_cases',abs(len(comp)-9),'3 demand scenarios × 3 routing policies')
    check('final_control_version',(comp.loc[comp.policy!='baseline','control_version']!='capacity_guarded_v2').sum(),'No unguarded development controller mixed into the final comparison')
    check('paired_comparison_demand',sum(f.demand_sha256.nunique()!=1 or f.planned.nunique()!=1 for _,f in comp.groupby('scenario')),'SHA-256 verifies identical planned trips within each policy comparison')
    control_forecasts=[]
    for r in comp.itertuples():
        f=pd.read_csv(ROOT/'runs'/f'{r.run:03}_{r.policy}.csv').sort_values(['gate','direction','second'])
        f['future']=f.groupby(['gate','direction']).queue.shift(-15);f=f.dropna(subset=['future'])
        for label,prediction in [('selected_ML',f.forecast_15min),('Persistence',f.queue)]:
            control_forecasts.append(dict(scenario=r.scenario,policy=r.policy,forecast=label,MAE=float(np.abs(f.future-prediction).mean()),RMSE=float(np.sqrt(((f.future-prediction)**2).mean())),rows=len(f)))
    save(pd.DataFrame(control_forecasts),ROOT/'results/closed_loop_forecast_metrics.csv')
    replay_errors=0
    for r in comp.itertuples():
        try:
            with gzip.open(ROOT/'replay'/f'{r.run:03}_{r.policy}.json.gz','rt',encoding='utf-8') as f:frames=json.load(f)
            replay_errors+=len(frames)!=435 or frames[-1]['arrived']!=r.completed
        except (OSError,EOFError,ValueError,KeyError,IndexError):replay_errors+=1
    check('complete_replay_files_and_final_counts',replay_errors,'All nine gzip/JSON recordings have 435 frames and final completed counts agree with tripinfo summaries')
    ranges=[]
    for name,f in train.groupby('split'):ranges.append(dict(split=name,rows=len(f),runs=f.run.nunique(),nonzero_target_rows=int((f.target_queue_15min>0).sum()),busy_target_ge_10_rows=int((f.target_queue_15min>=10).sum()),maximum_target=int(f.target_queue_15min.max())))
    save(pd.DataFrame(ranges),ROOT/'results/split_profile.csv')
    return raw,train,comp

def figures(raw,train,comp):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metrics=pd.read_csv(ROOT/'results/model_metrics.csv');m=metrics[metrics.split=='test']
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for ax,col in zip(axes,['MAE','RMSE']):
        bars=ax.bar(m.model,m[col],color=['#9babbc','#1b76a6','#16a58e']);ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_ylim(0,m[col].max()*1.22);ax.set_ylabel(col+' (stopped vehicles)');ax.set_title('15-minute queue forecast: held-out simulation runs')
    fig.savefig(ROOT/'results/model_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for ax,scenario in zip(axes,['normal','peak','incident']):
        f=comp[comp.scenario==scenario];bars=ax.bar(f.policy,f.mean_delay_per_planned_seconds,color=['#9babbc','#1b76a6','#16a58e']);ax.bar_label(bars,fmt='%.1fs',padding=4);ax.set_ylim(0,max(1,f.mean_delay_per_planned_seconds.max())*1.25);ax.set_title(scenario.title()+' demand');ax.set_ylabel('Stopped + pre-insertion delay per planned leg (s)');ax.tick_params(axis='x',labelsize=9)
    fig.suptitle('Same demand, three policies. Synthetic, uncalibrated geographic SUMO experiment.',fontsize=12)
    fig.savefig(ROOT/'results/control_comparison.png',dpi=160);plt.close(fig)
    pred=pd.read_csv(ROOT/'results/test_predictions.csv');run=int(pred.run.iloc[0]);f=pred[(pred.run==run)&(pred.gate==1)&(pred.direction=='IN')]
    fig,ax=plt.subplots(figsize=(12,4.2),layout='constrained');x=(f.second+900)/3600
    ax.plot(x,f.target_queue_15min,color='#253e51',label='Observed queue at target time',linewidth=1.7)
    for name,color in [('Persistence','#a8b1bb'),('RandomForest','#1b76a6'),('XGBoost','#16a58e')]:ax.plot(x,f[name],color=color,label=name,linewidth=1.1,alpha=.9)
    ax.set(xlabel='Target time (hour of day)',ylabel='Stopped vehicles on monitored approach',title=f'Unseen run {run}: Gate 1 entry, forecasts made 15 minutes earlier');ax.legend(ncol=4,loc='upper right',fontsize=8)
    fig.savefig(ROOT/'results/forecast_example.png',dpi=160);plt.close(fig)
    from replay_viewer import load_geometry,load_frames
    roads,areas,cat=load_geometry();frames=load_frames('peak','baseline')
    frame=max(frames,key=lambda f:next(s['queue'] for s in f['states'] if s['gate']==1 and s['direction']=='IN'))
    fig,ax=plt.subplots(figsize=(12,8.2),layout='constrained',facecolor='#0d1726');ax.set_facecolor('#111f30')
    for shape,kind in areas:ax.add_collection(PolyCollection([shape],facecolors='#26374b',edgecolors='#34475c',linewidths=.5))
    ax.add_collection(LineCollection([s for s,n,e in roads],colors='#485d73',linewidths=[max(.8,n*1.1) for s,n,e in roads],zorder=2))
    for direction,color in [('IN','#48b9f3'),('OUT','#f4b45b')]:
        pos=np.array([[x,y] for vid,x,y,d in frame['vehicles'] if d==direction])
        if len(pos):ax.scatter(pos[:,0],pos[:,1],s=9,color=color,label='Entry trips' if direction=='IN' else 'Exit trips',zorder=5)
    for text,key in [('GATE 1','G1_IN_A'),('GATE 2','G2_IN')]:
        x,y=cat['gates'][key]['position'];ax.scatter([x],[y],s=190,facecolor='#54d6b5',edgecolor='#0d1726',linewidth=2,zorder=8);ax.annotate(text,(x,y),xytext=(12,15),textcoords='offset points',color='white',weight='bold',fontsize=12,zorder=9)
    for z,info in cat['zones'].items():
        e=next(s for s,n,eid in roads if eid==info['in_edge']);x,y=e[-1];ax.text(x,y,z.replace('Z_',''),color='#a9cbbb',fontsize=8,zorder=6)
    ax.set(xlim=(4250,6500),ylim=(1300,2920));ax.set_aspect('equal');ax.axis('off');ax.set_title(f"UQU Abidiyah road network · SUMO positions at {frame['time']}\nSynthetic peak scenario, fixed routing",color='white',fontsize=15,pad=15)
    ax.text(.01,.01,'Road geometry © OpenStreetMap contributors, ODbL. Hypothetical demand and gate processing.\nThis is a rendered simulation frame, not a campus aerial photograph.',transform=ax.transAxes,color='#a4b4c6',fontsize=9)
    ax.text(.97,.96,'N ↑',transform=ax.transAxes,color='white',ha='right',fontsize=16)
    handles,labels=ax.get_legend_handles_labels()
    if handles:ax.legend(handles,labels,facecolor='#152337',labelcolor='white',loc='lower right')
    fig.savefig(ROOT/'results/geographic_simulation_preview.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)

def report(counts,raw,train,comp):
    metrics=pd.read_csv(ROOT/'results/model_metrics.csv');winner=json.loads((ROOT/'models/metadata.json').read_text())['selected_by_validation'];ab=pd.read_csv(ROOT/'results/academic_ablation.csv')
    lines=['# نتائج النسخة الجغرافية من مشروع مرور العابدية','',
        'هذه نتائج تشغيل فعلية للبرنامج على طلب مروري صناعي داخل SUMO. ليست قياسات ميدانية لجامعة أم القرى، ولا تثبت بعدُ فاعلية النظام في الشارع الحقيقي.','',
        '## ما الذي تم تنفيذه؟','',
        f'- استخدام {counts["students"]:,} طالبًا افتراضيًا من مقر العابدية – طلاب، و{counts["student_timetables"]:,} لقاءً دراسيًا أسبوعيًا. العدد افتراض في البيانات وليس إحصاءً رسميًا.',
        '- ربط المباني الأكاديمية بخمس مناطق وصول افتراضية داخل شبكة طرق مستوردة من OpenStreetMap. الربط ليس خريطة رسمية للكليات.',
        '- إنشاء 40 مسارًا متصلًا يغطي بوابتين، اتجاهي الدخول والخروج، نقطتي منشأ/وجهة خارجية وخمس مناطق داخلية.',
        '- تشغيل 30 يومًا/سيناريو مستقل البذرة من 06:30 إلى 21:00، والقياس كل دقيقة لأربعة تيارات: دخول وخروج كل بوابة.',
        f'- حفظ {len(raw):,} قياسًا، واستخراج {len(train):,} مثالًا له هدف معلوم بعد 15 دقيقة.',
        '- تدريب Random Forest وXGBoost، وضبط عمقين لكل نموذج باستخدام مجموعة التحقق فقط.','',
        '## تقسيم البيانات والهدف','',
        'الهدف هو **عدد السيارات المتوقفة على مقاطع المراقبة قرب البوابة بعد 15 دقيقة**، وليس طول الطابور بالمتر. آخر 15 دقيقة من كل سلسلة لا تدخل التدريب لعدم وجود هدف لاحق داخل أفق التشغيل.',
        '18 تشغيلًا للتدريب، و6 للتحقق، و6 للاختبار. لا تتوزع دقائق التشغيل نفسه بين المجموعات. الجداول الأسبوعية الصناعية تتكرر بين السيناريوهات؛ الاختبار يقيس التعميم على سيناريوهات محاكاة جديدة، لا على جامعة أو فصل دراسي حقيقي جديد.','',
        '| النموذج | MAE على الاختبار | RMSE على الاختبار |','|---|---:|---:|']
    for r in metrics[metrics.split=='test'].itertuples():lines.append(f'| {r.model} | {r.MAE:.3f} سيارة | {r.RMSE:.3f} سيارة |')
    baseline=metrics[(metrics.model=='Persistence')&(metrics.split=='test')].iloc[0]
    chosen=metrics[(metrics.model==winner)&(metrics.split=='test')].iloc[0]
    finding=('خط الأساس Persistence حقق MAE أقل من النموذج المختار. لذلك لا تدعم النتائج ادعاء أن التعلم الآلي أفضل في متوسط خطأ التوقع. المتحكم التنبؤي المعروض تجربة بحثية بالنموذج الأفضل بين نموذجي التعلم فقط، وليس اختيارًا أثبت تفوقه على جميع البدائل.' if baseline.MAE<chosen.MAE else 'النموذج المختار حقق MAE أقل من Persistence في مجموعة الاختبار الحالية، مع بقاء حدود التعميم من المحاكاة إلى الواقع.')
    lines+=['',f'النموذج المختار **بين نموذجي التعلم الآلي** وفق MAE على مجموعة التحقق: **{winner}**. خط الأساس Persistence يفترض أن طابور بعد 15 دقيقة يساوي الطابور الحالي. النتائج التفصيلية لكل اتجاه وفترات الطابور الأكبر موجودة في `model_metrics_by_stream.csv`؛ مهمتها منع المتوسط العام المنخفض من إخفاء ضعف النموذج أثناء الزحام.','',finding,'',
        '## هل ساعدت بيانات الجداول؟','',
        f'اختبار تشخيصي على النموذج المختار: MAE مع جميع المدخلات = {ab.iloc[0].test_MAE:.3f}، وبدون أعداد بدايات/نهايات المحاضرات القادمة = {ab.iloc[1].test_MAE:.3f}. المقارنة تخص هذه المحاكاة ولا تثبت أن الأثر سيكون نفسه في الواقع. لم نستخدم هذه النتيجة لتغيير اختيار النموذج بعد الاطلاع على الاختبار.','',
        '## مقارنة التوجيه','',
        'نستخدم الطلب نفسه والبذرة نفسها لكل مقارنة. التوجيه التفاعلي يستخدم الطابور الحالي. التنبؤي يستخدم توقع الطابور. كلاهما يحسب تكلفة تقريبية تشمل زمن المسار الحر وعقوبة الطابور، ويحدث نسبة التوزيع كل خمس دقائق بحد أقصى خمس نقاط مئوية. الحد الأدنى لحصة البوابة الأولى مشتق من نسبة سعة مساراتها إلى مجموع سعة البوابتين، ولا يقل عن 55%، والسقف 90%. لا يبدأ التحويل بسبب قصر الطريق وحده دون وجود طابور أو توقع طابور، ويفحص المقاطع بعد البوابة قبل توجيه مزيد من السيارات إليها. لا يغير اتجاه السير أو يغلق بوابة أو يعيد توجيه سيارة سبق أن تجاوزت نقطة القرار.','',
        'استُخدمت الحالة العادية لتطوير قواعد التحكم بعد أن كشفت المحاولة الأولى ضرر التحويل الزائد. احتفظنا بملخصها في `diagnostics/controller_v1_normal_predictive.json`. لذلك تعد هذه المقارنات دراسة استكشافية، وليست اختبارًا خارجيًا مستقلًا للمتحكم أو إثباتًا إحصائيًا لفاعليته.','',
        '| الحالة | السياسة | الرحلات المكتملة | غير المدرجة بعد | التأخير لكل رحلة مخططة (ثانية) |','|---|---|---:|---:|---:|']
    for r in comp.itertuples():lines.append(f'| {r.scenario} | {r.policy} | {r.completed:,} | {r.not_inserted:,} | {r.mean_delay_per_planned_seconds:.1f} |')
    lines+=['','كل دخول أو خروج يعد رحلة مستقلة. مقياس التأخير يجمع التوقف على الطريق والتأخر عن موعد إدراج الرحلة، ويشمل الطلب الذي لم يدخل المحاكاة بعد حتى نهاية التشغيل. لا يشمل كامل زمن السير البطيء، ولا ينبغي تسميته متوسط زمن الرحلة. الخروج لا يُطلق قبل وصول دخول الطالب نفسه. الانتظار بين المحاضرات ضمنيًا خارج شبكة القيادة، ولا توجد محاكاة لسعة المواقف.','']
    for scenario,f in comp.groupby('scenario'):
        base=f[f.policy=='baseline'].iloc[0];pred=f[f.policy=='predictive'].iloc[0];react=f[f.policy=='reactive'].iloc[0]
        pct=(base.mean_delay_per_planned_seconds-pred.mean_delay_per_planned_seconds)/max(1e-9,base.mean_delay_per_planned_seconds)*100
        direction='انخفض' if pct>=0 else 'ارتفع'
        lines.append(f'- {scenario}: {direction} التأخير التنبؤي عن الوضع الثابت بمقدار {abs(pct):.1f}%. التنبؤي '+('أفضل' if pred.mean_delay_per_planned_seconds<react.mean_delay_per_planned_seconds else 'ليس أفضل')+' من التفاعلي وفق هذا المقياس في هذا التشغيل المحدد.')
    audit=pd.read_csv(ROOT/'results/detector_interval_audit.csv')
    lines+=['','## تدقيق الملفات','',
        f'نُفذ {len(CHECKS)} فحصًا برمجيًا، وعدد الفحوص غير المجتازة {sum(x["Result"]!="PASS" for x in CHECKS)}. تشمل العلاقات الأكاديمية والتعارضات والسعات، وهدف 15 دقيقة، وفصل التشغيلات، وحساب جميع الرحلات، وصحة التسجيلات التسعة.',
        f'سجل العدادات يوثق {len(audit)} قراءة اختلفت فيها واجهة SUMO الحية عن عدد العبور المكتمل في XML. لا نستبدل هذه القيم أو نخفيها؛ نتحقق من اتساقها مع اختلاف تعريف حد الفترة وعدد مسارات العدادات كما يوضح DATA_DICTIONARY.md. نجاح الفحوص التقنية لا يعني أن نموذج التوقع أدق أو أن المتحكم جاهز ميدانيًا.','',
        '## حدود التجربة المهمة','',
        '- لا توجد كاميرات أو حساسات فعلية في الجامعة في هذه النسخة. جميع القياسات المرورية من SUMO، والجداول صناعية.',
        '- الحركة الحالية سيارات طلاب افتراضيين من فئة passenger فقط. لا توجد حركة خلفية مستقلة لسكان الطرق المحيطة أو شاحنات أو حافلات أو مشاة، لذا لا تمثل النتائج كامل حركة المنطقة.',
        '- توزيع من يقود سيارة ونسبة تفضيل البوابة الأولى وهامش الوصول والمغادرة افتراضات معلنة في الكود وملفات الطلب. عدد الطلاب لا يتحول مباشرة إلى عدد سيارات.',
        '- إحداثيات الطرق من OSM، لكن أعداد المسارات والسرعات والاتجاهات تحتاج مراجعة ميدانية. أضفنا خمس نقاط إشارة/معالجة افتراضية عند مقاطع البوابتين لبناء سيناريو السعة. ليست إشارات قائمة رُصدت في الجامعة.',
        '- الحادث محاكاة لانخفاض سرعة مقاطع دخول البوابة الأولى من 08:00 إلى 08:20. وجوده يُعرف من السيناريو؛ لم ندرّب كاشف حوادث بصريًا ولا ندعي اكتشاف حادث من كاميرا.',
        '- مراقبة الطوابير تعتمد مقاطع متصلة تمتد للخلف نحو 400 متر، وقد يتجاوز آخر مقطع هذا الحد. كل مقطع ينتمي لمراقب واحد منعًا لاحتساب السيارة في بوابتين. الملف `queue_monitors.json` يحدد المقاطع بدقة.',
        '- النماذج دُرّبت على تشغيلات التوزيع الثابت. تغيير التوجيه يغيّر ديناميكية النظام، لذلك دقة التوقع أثناء التحكم تحتاج جمع حالات سياسات متنوعة وإعادة تقييم، ولا نعتمدها ضمانًا للسلامة.',
        '- المقارنة تشمل ثلاثة سيناريوهات محددة ببذرة واحدة لكل سيناريو، وليست اختبار دلالة إحصائية متعدد البذور. قد يتفوق المتحكم التفاعلي في بعض الحالات.',
        '- القرار في هذه النسخة هو اختيار البوابة قبل استمرار الرحلة، وليس التحكم الفعلي بالحواجز أو تركيب إشارات على دوارات الجامعة. يبقى تشغيل الأجهزة الحقيقية مرحلة مستقلة تتطلب موافقات وتصميم سلامة.',
        '- لا يدعي المشروع أن الازدحام خارج الجامعة تم حله. الشبكة تتضمن جزءًا من الطرق الخارجية لرصد أثر التحويل ضمن المنطقة الممثلة فقط.','',
        '## الخطوة العلمية التالية','',
        'مراجعة المداخل والمخارج وعدد المسارات مع الجامعة، ثم معايرة الطلب والسرعات والطوابير بقياسات ميدانية. بعدها نجمع عدة بذور لكل سياسة وندرب على حالات تحكم متنوعة، ونختبر حدود السعة قبل إضافة تحكم فعلي بالإشارات أو الحواجز.']
    (ROOT/'RESULTS_AR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=dict(project='UQU GeoTraffic v2',seed=SEED,data_status='Synthetic academic data and simulated traffic, uncalibrated OSM geography',academic_rows=counts,raw_traffic_rows=len(raw),training_rows=len(train),run_count=30,comparison_run_count=9,selected_model=winner,validation_passed=all(x['Result']=='PASS' for x in CHECKS),versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','joblib','matplotlib','eclipse-sumo','traci','sumolib','libsumo','pyproj']},python=platform.python_version())
    (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    counts=academic_checks();raw,train,comp=experiment_checks()
    save(pd.DataFrame(CHECKS),ROOT/'results/validation_report.csv')
    report(counts,raw,train,comp);figures(raw,train,comp)
    failures=[x for x in CHECKS if x['Result']!='PASS']
    print(json.dumps(dict(checks=len(CHECKS),failures=failures,academic_rows=counts,traffic_rows=len(raw),training_rows=len(train)),ensure_ascii=False,indent=2))
    if failures:raise SystemExit('Validation failed. Do not treat the package as validated.')

if __name__=='__main__':main()
