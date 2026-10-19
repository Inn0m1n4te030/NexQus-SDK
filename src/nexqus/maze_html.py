"""Offline-capable UI for the bounded maze lesson. No external assets."""
import json


def render_explorer(data, explanation_html=''):
    payload = json.dumps(dict(experiment=data, explanation_html=explanation_html), allow_nan=False)
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    return _PAGE.replace('__MAZE_DATA__', payload)


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NexQus · Quantum Maze Explorer</title>
<style>
:root{color-scheme:light;--ink:#172b46;--muted:#526780;--blue:#2859d6;--green:#087a57;--rose:#bb2860;--line:#dce5ee}
*{box-sizing:border-box}body{margin:0;background:#f2f6fc;color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1380px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:20px}
h1{font-size:clamp(26px,4vw,42px);letter-spacing:-1.3px;margin:3px 0 6px}h2{font-size:19px;margin:0 0 12px}p{margin:8px 0 14px}.eyebrow{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:2px;text-transform:uppercase}
.badge{background:#e2f4ec;color:#07583e;padding:8px 12px;border-radius:999px;font-size:13px;font-weight:700;white-space:nowrap}
.layout{display:grid;grid-template-columns:minmax(300px,.82fr) minmax(400px,1.3fr);gap:20px}.card{background:white;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 3px 12px #243c6710;margin-bottom:18px}
.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.spread{justify-content:space-between}.muted{color:var(--muted)}.small{font-size:13px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
label{font-size:13px;font-weight:650}select,input[type=number]{display:block;border:1px solid #a8b9cd;border-radius:8px;padding:8px;background:white;color:var(--ink);max-width:100%}
button{font:inherit;font-size:13px;font-weight:650;border:1px solid #adbed1;border-radius:9px;background:white;color:var(--ink);padding:9px 12px;cursor:pointer}button:hover:not(:disabled){background:#eaf0ff}button:focus-visible,select:focus-visible,input:focus-visible,svg [role=button]:focus-visible{outline:3px solid #e79816;outline-offset:3px}
button.primary{background:var(--blue);color:white;border-color:var(--blue)}button[aria-pressed=true]{background:#e4ecff;border-color:var(--blue);color:#143a97}button:disabled{opacity:.45;cursor:default}
#maze{display:block;max-width:100%;width:100%;max-height:440px;margin:16px auto;background:#edf2f8;border-radius:12px}#maze [role=button]{cursor:pointer}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}.metric{background:#f0f5fb;border-radius:10px;padding:12px}.metric strong{display:block;font-size:25px}.metric span{font-size:11px;color:var(--muted)}
#step-range{width:100%;accent-color:var(--blue);margin:16px 0}#stage-title{margin:12px 0 5px;font-size:21px}#stage-text{min-height:50px}.plot{width:100%;height:175px;display:block;margin-top:10px}#route-list{max-height:250px;overflow:auto;border:1px solid var(--line);border-radius:8px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}th{position:sticky;top:0;background:#f0f5fb}tr.selected{background:#e5edff}code{font-family:ui-monospace,monospace}#route-info{white-space:pre-wrap;background:#f5f8fc;border-radius:10px;padding:12px;font-size:12px;max-height:220px;overflow:auto}
.notice{padding:12px 14px;border-left:4px solid #d59316;border-radius:6px;background:#fff7e6;font-size:13px;margin:14px 0}.error{color:#9c1737}#measurement{padding:12px;background:#eaf1ff;border-radius:9px;margin:14px 0;min-height:46px}
details{margin-top:15px}summary{cursor:pointer;font-weight:650}.circuit-wrap{overflow:auto;max-height:350px;background:white;border:1px solid var(--line);border-radius:8px;margin:12px 0}.circuit-wrap svg{display:block;min-width:650px;max-width:100%;height:auto}.full{grid-column:1/-1}.hidden{display:none!important}
@media(max-width:860px){main{padding:16px}.layout{grid-template-columns:1fr}header{align-items:flex-start}.badge{white-space:normal}.card{padding:16px}.metrics{gap:6px}.metric{padding:9px}.metric strong{font-size:21px}}
</style></head><body><main>
<header><div><div class="eyebrow">NexQus / school lab</div><h1>Quantum Maze Explorer</h1><p class="muted">Make a maze. Inspect the guesses. Watch interference change the odds.</p></div><span class="badge">Local CPU simulation · no QPU</span></header>
<div id="offline-note" class="notice hidden">Offline lesson: navigation and seeded samples work here. Run the Python example to edit and solve new mazes.</div>
<div class="layout"><section>
<div class="card"><h2>1. Make your maze</h2>
<div class="controls"><label>Preset<select id="preset"></select></label><label>Junction budget<select id="junctions"><option value="1">1 choice / 2 qubits</option><option value="2">2 choices / 4 qubits</option><option value="3">3 choices / 6 qubits</option></select></label></div>
<div class="row" aria-label="Maze editing tool"><button id="wall-tool" aria-pressed="true">Wall / erase</button><button id="start-tool" aria-pressed="false">Entrance S</button><button id="goal-tool" aria-pressed="false">Exit G</button></div>
<svg id="maze" xmlns="http://www.w3.org/2000/svg" role="group" aria-label="Editable maze"></svg>
<div class="legend"><span><i class="dot" style="background:#253b55"></i>Wall</span><span><i class="dot" style="background:#dd8415"></i>Selected plan</span><span><i class="dot" style="background:#2859d6"></i>BFS shortest path (dashed)</span></div>
<div class="controls"><label>Grover iterations<select id="iterations"><option value="auto">Automatic</option></select></label><label>Measurement seed<input id="seed" type="number" min="0" max="4294967295" value="7"></label></div>
<div class="row"><button class="primary" id="run">Run NexQus</button><label><input type="checkbox" id="show-bfs"> Show BFS path</label></div>
<p id="status" role="status" aria-live="polite" class="small muted"></p>
<div id="coverage" class="notice"></div>
</div>
<div class="card"><h2>What do the bits mean?</h2><p class="small">Each pair <code>00 / 01 / 10 / 11</code> picks an available, unvisited neighbor at a junction. Options are ordered right, down, left, up. A corridor follows automatically.</p>
<p class="small muted">A plan can run out of choices, pick an unavailable option, or hit a dead end. After reaching G, unused pairs must be 00 so the same route is not counted repeatedly.</p>
<label for="route-select">Inspect a candidate plan</label><select id="route-select"></select><pre id="route-info"></pre></div>
</section><section>
<div class="card"><h2>2. Explore the quantum search</h2>
<div class="metrics"><div class="metric"><strong id="candidate-count">—</strong><span>Encoded plans</span></div><div class="metric"><strong id="valid-count">—</strong><span>Marked valid plans</span></div><div class="metric"><strong id="success">—</strong><span>Probability of a valid plan</span></div></div>
<div class="row"><button id="reset">Prepare / reset</button><button id="back">Back</button><button id="next" class="primary">Next stage</button><button id="play">Play</button><button id="measure">Measure this stage</button></div>
<input id="step-range" aria-label="Circuit stage" type="range" min="0" max="0" value="0">
<div id="stage-title"></div><p id="stage-text" class="small muted"></p>
<div class="row spread"><strong class="small">Probability of each encoded plan</strong><span class="small muted">Fixed scale: 0–100%</span></div>
<svg class="plot" id="probabilities" viewBox="0 0 700 175" role="img" aria-label="Candidate probability bars"></svg>
<div class="row spread"><strong class="small">Signed real amplitudes</strong><span class="small muted" id="amplitude-scale"></span></div>
<svg class="plot" id="amplitudes" viewBox="0 0 700 175" role="img" aria-label="Positive and negative real amplitudes"></svg>
<div class="legend"><span><i class="dot" style="background:#087a57"></i>Valid plan</span><span><i class="dot" style="background:#2859d6"></i>Other / positive amplitude</span><span><i class="dot" style="background:#bb2860"></i>Negative amplitude</span></div>
<p class="small muted">These circuits use real-valued gates. Negative amplitude means a relative phase of π; probability is amplitude squared. Each shot prepares the displayed stage again. The plots show its pre-measurement state, not a collapsed state.</p><div id="measurement" role="status" aria-live="polite">Measure to sample a plan from the displayed stage. A sample can fail.</div>
<div id="route-list"><table><thead><tr><th>Plan</th><th>Oracle says</th><th>Moves</th><th>Probability</th></tr></thead><tbody id="route-body"></tbody></table></div>
</div>
<div class="card"><h2>3. Count the work honestly</h2><table><tbody id="work"></tbody></table>
<p class="small muted">These counts use different units. Python already checks every encoded plan to build the truth-table oracle. Corridor following is classical work. The oracle is one logical diagonal operation, not a reversible maze-checking gate decomposition. This project demonstrates amplitude amplification, not a laptop speedup over BFS.</p>
<p class="small muted">BFS is complete on this grid and finds a shortest path. Grover searches only the encoded plans and does not minimize route length. More iterations can reduce success after the optimum.</p>
<details><summary>The actual NexQus circuit</summary><div id="circuit" class="circuit-wrap"></div></details>
<div class="row" style="margin-top:14px"><button id="lesson">Download circuit lesson</button><button id="download">Download experiment JSON</button></div>
</div></section></div></main>
<script type="application/json" id="maze-data">__MAZE_DATA__</script><script>
'use strict';
const initial=JSON.parse(document.getElementById('maze-data').textContent);
const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
let data=initial.experiment, rows=data.rows.slice(), stage=0, selected=0, tool='wall', timer=null, busy=false;
const offline=Boolean(data.offline), presets=data.presets;
function element(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs||{}))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;}
function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='Play';}
function status(message,error=false){$('status').textContent=message;$('status').className='small '+(error?'error':'muted');}
function settings(){return {rows,junctions:Number($('junctions').value),iterations:$('iterations').value==='auto'?null:Number($('iterations').value),seed:Number($('seed').value)};}
function controls(){
 const ready=Boolean(data)&&!busy;
 for(const id of ['reset','back','next','play','measure','lesson','download','route-select','step-range'])$(id).disabled=!ready;
 if(ready){$('back').disabled=stage===0;$('next').disabled=stage===data.frames.length-1;$('play').disabled=data.frames.length<2;}
 for(const id of ['preset','junctions','iterations','seed','wall-tool','start-tool','goal-tool','run'])$(id).disabled=offline||busy;
}
function invalidate(){stop();data=null;stage=0;selected=0;controls();$('route-select').replaceChildren();$('route-info').textContent='Run NexQus to rebuild candidate plans.';$('route-body').replaceChildren();$('probabilities').replaceChildren();$('amplitudes').replaceChildren();$('circuit').replaceChildren();for(const id of ['candidate-count','valid-count','success'])$(id).textContent='—';$('coverage').textContent='Maze or settings changed. Run NexQus to build a new oracle.';$('stage-title').textContent='Ready to prepare';$('stage-text').textContent='';$('measurement').textContent='No current result.';$('work').replaceChildren();status('Ready to run your maze.');drawMaze();}
function edit(r,c){if(offline||busy)return;const grid=rows.map(row=>row.split('')),old=grid[r][c];
 if(tool==='wall'){if(old==='S'||old==='G')return;grid[r][c]=old==='#'?'.':'#';}
 else{const symbol=tool==='start'?'S':'G';if(old===(symbol==='S'?'G':'S'))return;for(const line of grid)for(let i=0;i<line.length;i++)if(line[i]===symbol)line[i]='.';grid[r][c]=symbol;}
 rows=grid.map(row=>row.join(''));invalidate();}
function drawMaze(){const svg=$('maze'),height=rows.length,width=rows[0].length;svg.setAttribute('viewBox',`0 0 ${width*64} ${height*64}`);svg.replaceChildren();
 rows.forEach((row,r)=>[...row].forEach((value,c)=>{const cell=element('rect',{x:c*64+2,y:r*64+2,width:60,height:60,rx:8,fill:value==='#'?'#253b55':value==='S'?'#bce9d7':value==='G'?'#ffe1a9':'#ffffff',stroke:'#d6e1ec','stroke-width':1,role:'button',tabindex:offline?-1:0,'aria-label':`Row ${r+1}, column ${c+1}, ${value==='#'?'wall':value==='S'?'entrance':value==='G'?'exit':'open'}`});cell.addEventListener('click',()=>edit(r,c));cell.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();edit(r,c);}});svg.append(cell);}));
 function route(path,color,dashed){if(path.length<2)return;svg.append(element('polyline',{points:path.map(([r,c])=>`${c*64+32},${r*64+32}`).join(' '),fill:'none',stroke:color,'stroke-width':dashed?4:7,'stroke-linejoin':'round','stroke-linecap':'round','stroke-dasharray':dashed?'8 7':'none',opacity:.85,'pointer-events':'none'}));}
 if(data){route(data.candidates[selected].path,'#dd8415',false);if($('show-bfs').checked)route(data.classical.path,'#2859d6',true);}
 rows.forEach((row,r)=>[...row].forEach((value,c)=>{if(value==='S'||value==='G')svg.append(element('text',{x:c*64+32,y:r*64+39,'text-anchor':'middle','font-size':21,'font-family':'system-ui','font-weight':800,fill:'#172b46','pointer-events':'none'},value));}));}
function choose(index){selected=index;$('route-select').value=String(index);drawMaze();renderPlots();renderRows();const plan=data.candidates[index];$('route-info').textContent=`${plan.bits.match(/../g).join(' · ')}\n${plan.reason}\n${plan.moves} moves; ${plan.choices_used} junction choices\n`+plan.decisions.map((d,i)=>`Choice ${i+1}: at (${d.cell.join(', ')}), ${d.bits} selects option ${d.selected+1} of ${d.options.length}`).join('\n');}
function renderPlots(){if(!data)return;const f=data.frames[stage],count=data.candidates.length,left=35,width=650,slot=width/count,range=Math.max(.15,...f.real.map(Math.abs));
 for(const id of ['probabilities','amplitudes'])$(id).replaceChildren();
 $('probabilities').append(element('line',{x1:left,y1:155,x2:690,y2:155,stroke:'#9bacbf'}));$('probabilities').append(element('text',{x:2,y:16,'font-size':11,fill:'#526780'},'100%'));
 $('amplitudes').append(element('line',{x1:left,y1:86,x2:690,y2:86,stroke:'#9bacbf'}));$('amplitude-scale').textContent=`Axis: ±${range.toFixed(3)}`;
 data.candidates.forEach((plan,i)=>{const p=f.probabilities[i],a=f.real[i],x=left+i*slot;
 const prob=element('rect',{x,y:155-p*140,width:Math.max(1,slot-1),height:Math.max(0,p*140),fill:plan.valid?'#087a57':'#2859d6',stroke:i===selected?'#dd8415':'none','stroke-width':2});prob.append(element('title',{},`${plan.bits}: ${(100*p).toFixed(3)}%`));prob.addEventListener('click',()=>choose(i));$('probabilities').append(prob);
 const h=Math.abs(a)/range*74,amp=element('rect',{x,y:a>=0?86-h:86,width:Math.max(1,slot-1),height:h,fill:a<0?'#bb2860':'#2859d6',stroke:i===selected?'#dd8415':plan.valid?'#087a57':'none','stroke-width':1.5});amp.append(element('title',{},`${plan.bits}: amplitude ${a.toFixed(6)}`));amp.addEventListener('click',()=>choose(i));$('amplitudes').append(amp);
 });}
function renderRows(){if(!data)return;$('route-body').replaceChildren();const f=data.frames[stage];data.candidates.forEach((plan,i)=>{const tr=document.createElement('tr');tr.className=i===selected?'selected':'';const button=document.createElement('button');button.textContent=plan.bits;button.setAttribute('aria-label','Inspect plan '+plan.bits);button.onclick=()=>choose(i);const td=document.createElement('td');td.append(button);tr.append(td);for(const value of [plan.valid?'Valid':'Reject',plan.moves,(100*f.probabilities[i]).toFixed(2)+'%']){const td=document.createElement('td');td.textContent=String(value);tr.append(td);}$('route-body').append(tr);});}
function renderStage(){if(!data)return;const f=data.frames[stage];$('step-range').value=String(stage);$('success').textContent=(100*f.success_probability).toFixed(1)+'%';
 const titles={prepare:'Prepare: equal guesses',mark:`Mark: oracle call ${f.iteration}`,amplify:`Amplify: iteration ${f.iteration}`};$('stage-title').textContent=titles[f.phase];
 $('stage-text').textContent=f.phase==='prepare'?'Hadamard gates give every encoded plan equal probability. No path has been selected.':f.phase==='mark'?'The oracle flips the sign of valid-plan amplitudes. Probabilities do not change at this step.':"Diffusion reflects amplitudes around their mean. Interference changes the probability distribution; extra iterations can overshoot.";
 $('measurement').textContent='Measure to sample this stage. A sample can fail.';renderPlots();renderRows();controls();}
function apply(result){data=result;rows=data.rows.slice();stage=0;selected=data.candidates.findIndex(p=>p.valid);if(selected<0)selected=0;$('candidate-count').textContent=String(data.candidates.length);$('valid-count').textContent=String(data.valid_plans);$('step-range').max=String(data.frames.length-1);$('junctions').value=String(data.junctions);
 $('route-select').replaceChildren();data.candidates.forEach((plan,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=plan.bits+' · '+plan.reason;$('route-select').append(option);});
 $('coverage').textContent=data.valid_plans?`${data.valid_plans} encoded routes reach the exit. Automatic recommendation: ${data.suggested_iterations} Grover iterations. You are viewing ${data.iterations}.`:data.classical.path.length?'BFS found a path, but none fits this junction-choice budget. Increase the budget or add corridors. Zero marked plans does not mean the maze is impossible.':'No exit path exists: BFS and the encoded search both find none.';
 $('work').replaceChildren();for(const [label,value] of [['BFS nodes expanded',data.classical.expanded_nodes],['BFS shortest path (moves)',data.classical.path.length?data.classical.path.length-1:'No path'],['Classical oracle compiler: plans checked',data.resources.classical_route_checks],['Classical compiler: route-cell visits',data.resources.classical_route_cell_visits],['Quantum circuit: logical oracle calls',data.resources.logical_oracle_calls],['Quantum circuit: logical depth',data.resources.logical_depth]]){const row=document.createElement('tr');for(const valuePart of [label,value]){const cell=document.createElement('td');cell.textContent=String(valuePart);row.append(cell);}$('work').append(row);}
 $('circuit').innerHTML=data.circuit_svg;choose(selected);renderStage();status(`${data.qubits} qubits · ${data.candidates.length} encoded plans · actual NexQus statevectors`);}
async function request(endpoint,payload){const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok)throw new Error(result.error||'Request failed');return result;}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
for(const name of Object.keys(presets)){const option=document.createElement('option');option.textContent=name;option.value=name;$('preset').append(option);}
for(let i=0;i<=8;i++){const option=document.createElement('option');option.value=String(i);option.textContent=String(i);$('iterations').append(option);}
for(const value of ['wall','start','goal'])$(value+'-tool').onclick=()=>{tool=value;for(const candidate of ['wall','start','goal'])$(candidate+'-tool').setAttribute('aria-pressed',String(candidate===value));};
for(const id of ['junctions','iterations'])$(id).onchange=()=>{if(!busy)invalidate();};
const matching=Object.keys(presets).find(name=>JSON.stringify(presets[name])===JSON.stringify(rows));if(matching)$('preset').value=matching;
$('preset').onchange=()=>{if(busy||offline)return;rows=presets[$('preset').value].slice();invalidate();};
$('run').onclick=async()=>{if(busy||offline)return;stop();busy=true;controls();status('Running NexQus locally…');try{apply(await request('/api/solve',settings()));}catch(error){status(error.message,true);}finally{busy=false;controls();}};
$('show-bfs').onchange=drawMaze;$('route-select').onchange=()=>{if(data)choose(Number($('route-select').value));};
$('step-range').oninput=()=>{stop();stage=Number($('step-range').value);renderStage();};
$('reset').onclick=()=>{stop();stage=0;renderStage();};$('back').onclick=()=>{stop();stage=Math.max(0,stage-1);renderStage();};$('next').onclick=()=>{stop();stage=Math.min(data.frames.length-1,stage+1);renderStage();};
$('play').onclick=()=>{if(timer!==null){stop();return;}if(stage===data.frames.length-1)stage=0;$('play').textContent='Pause';renderStage();timer=setInterval(()=>{stage++;renderStage();if(stage===data.frames.length-1)stop();},750);};
$('measure').onclick=async()=>{if(!data||busy)return;stop();let sample;busy=true;controls();try{sample=offline?data.samples[stage]:await request('/api/measure',{...settings(),frame:stage});choose(sample.index);$('measurement').textContent=`Measured ${sample.bits}: ${sample.valid?'valid exit route':'not a valid plan'}. ${sample.reason}. Seed ${sample.seed}.`;if(!offline)$('seed').value=String((sample.seed+1)%4294967296);}catch(error){status(error.message,true);}finally{busy=false;controls();}};
$('lesson').onclick=async()=>{if(!data||busy)return;stop();busy=true;controls();try{const lesson=offline?initial.explanation_html:(await request('/api/explain',settings())).html;download('nexqus-maze-circuit.html',lesson,'text/html');}catch(error){status(error.message,true);}finally{busy=false;controls();}};
$('download').onclick=()=>download('nexqus-maze-experiment.json',JSON.stringify(data,null,2),'application/json');
$('seed').value=String(data.seed);$('iterations').value=data.iterations===data.suggested_iterations?'auto':String(data.iterations);
$('offline-note').classList.toggle('hidden',!offline);apply(data);
window.mazeLesson={get data(){return data;},get stage(){return stage;}};
</script></body></html>'''
