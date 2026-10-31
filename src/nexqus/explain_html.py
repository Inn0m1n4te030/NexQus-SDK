"""Offline HTML renderer; every calculation is supplied by the trace engine."""
import json


def render_report(data):
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False)
    # Do not allow names, titles, annotations or parameter names to end a script.
    for char,escaped in [('&',r'\u0026'),('<',r'\u003c'),('>',r'\u003e'),
                         ('\u2028',r'\u2028'),('\u2029',r'\u2029')]:
        payload=payload.replace(char,escaped)
    return _HTML.replace('__TRACE_DATA__',payload)


_HTML=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NexQus · Circuit explorer</title>
<style>
:root{color-scheme:light;--ink:#202332;--muted:#626779;--line:#dce0e9;--paper:#f4f5f9;--purple:#6150c8;--blue:#3176be;--orange:#c67632}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}button,input,select{font:inherit}button,select{cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #a69be7;outline-offset:3px}button{background:white;border:1px solid var(--line);border-radius:8px;padding:7px 12px;color:var(--ink)}button:hover{border-color:var(--purple)}button:disabled{opacity:.45;cursor:default}header{background:#242337;color:white;padding:18px max(24px,calc((100vw - 1240px)/2));display:flex;align-items:center;justify-content:space-between;gap:16px}.brand{letter-spacing:.13em;font-size:12px;font-weight:750}.badge{color:#d3cef4;border:1px solid #625b83;border-radius:20px;padding:4px 12px;font-size:12px}main{max-width:1288px;margin:auto;padding:28px 24px 40px}h1{font-size:32px;line-height:1.2;margin:0 0 9px;letter-spacing:-.03em}h2{font-size:16px;margin:0 0 5px}p{margin:5px 0 12px}.muted,.caption{color:var(--muted)}.caption{font-size:12px}.intro{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:22px}.stats{display:flex;gap:18px;text-align:right;white-space:nowrap}.stats strong{display:block;font-size:23px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;min-width:0}.circuit-card{margin-bottom:18px}.toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.toolbar label{margin-left:auto}.toolbar select{padding:6px;border:1px solid var(--line);border-radius:7px;background:white}#circuit{overflow-x:auto;margin:12px 0}#step{width:100%;accent-color:var(--purple)}.step-line{display:flex;align-items:center;justify-content:space-between;margin:8px 0}.step-line strong{font-size:16px}.layout{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(320px,1fr);gap:18px}.stack{display:grid;gap:18px;align-content:start}.two{display:grid;grid-template-columns:1fr 1fr;gap:15px}.metric{border-left:3px solid #b2a6e5;padding-left:12px}.metric strong{font-size:22px;display:block;font-variant-numeric:tabular-nums}.metric span{font-size:12px;color:var(--muted)}.explanation{background:#f5f2ff;border:1px solid #e5dff9;border-radius:8px;padding:12px;margin:12px 0}.blocks,.parameters{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.chip{font-size:12px;padding:5px 10px}.chip.active{border-color:var(--purple);background:#f0ecff;color:#4a3caf}.plot{width:100%;display:block}.qubits{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:13px}.qubit{border:1px solid var(--line);border-radius:9px;padding:10px}.qubit strong{font-size:14px}.qubit svg{width:100%;height:120px}.qubit .caption{font-variant-numeric:tabular-nums}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:7px 5px;text-align:right;border-bottom:1px solid #edf0f5}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:550}.scroll{overflow:auto;max-height:280px}details{margin-top:18px;border-top:1px solid var(--line);padding-top:13px}summary{cursor:pointer;font-weight:600}li{margin-bottom:5px}footer{margin-top:20px;color:var(--muted);font-size:12px}.legend{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}.error{background:#fff0eb;padding:20px;color:#862f15}#status{font-variant-numeric:tabular-nums}.covariance{max-width:360px;margin:auto}.phase-note{margin-top:8px}.noscript{padding:20px;background:#fff0db}button svg{pointer-events:none}@media(max-width:850px){.layout{grid-template-columns:1fr}.intro{display:block}.stats{margin-top:15px;text-align:left}.toolbar label{margin-left:0}.two{grid-template-columns:1fr 1fr}main{padding:20px 14px}.card{padding:16px}h1{font-size:27px}}@media print{header{background:white;color:black}.toolbar,#step,.blocks,.parameters,#download{display:none}.layout{display:block}.card{break-inside:avoid;margin-bottom:12px}.scroll{max-height:none}footer:after{content:' · Printed snapshot only; open HTML for all steps.'}}
</style></head><body>
<header><div class="brand">NEXQUS / INSIGHT</div><div class="badge">Exact · Local simulation</div></header>
<noscript><div class="noscript">Enable JavaScript to explore this offline report. No network connection is required.</div></noscript>
<main><div class="intro"><div><h1 id="title"></h1><div class="muted">Follow the state. Inspect the evidence. Understand the measurement.</div></div><div class="stats"><div><strong id="qubit-count"></strong>qubits</div><div><strong id="gate-count"></strong>operations</div><div><strong id="depth"></strong>logical depth</div></div></div>
<section class="card circuit-card" id="training-panel" hidden>
<h2>Training checkpoint playback</h2><label for="checkpoint">Recorded checkpoint</label>
<select id="checkpoint"><option value="final">Supplied final bindings</option></select>
<button id="training-play">Play checkpoints</button><p id="training-summary" class="caption" aria-live="polite"></p>
<svg id="training-loss" class="plot" viewBox="0 0 640 120" role="img" aria-label="Recorded training losses"></svg>
<p class="caption">Only supplied checkpoints are replayed, without interpolation. Loss is caller-supplied training metadata; predictions and sensitivities are recomputed for the recorded bindings. Circuit steps remain independently selectable. Layer interventions belong to the supplied final bindings only.</p>
</section>
<section class="card circuit-card" id="comparison-panel" hidden>
<h2>Compare two predictions at the selected circuit step</h2><label for="comparison">Right-hand input</label><select id="comparison"></select>
<p id="comparison-summary" class="caption" aria-live="polite"></p>
<div class="two"><div><h3>Selected checkpoint / circuit</h3><p id="compare-left"></p></div><div><h3>Comparison input (fixed bindings)</h3><p id="compare-right"></p></div></div>
<div class="scroll"><table><thead><tr><th>Binding / role</th><th>Left value</th><th>Right value</th><th>Left derivative</th><th>Right derivative</th></tr></thead><tbody id="comparison-parameters"></tbody></table></div>
<div class="scroll"><table><thead><tr><th>Basis</th><th>Left P</th><th>Right P</th><th>Difference</th></tr></thead><tbody id="comparison-probabilities"></tbody></table></div>
<p id="comparison-coverage" class="caption"></p><p class="caption">Comparisons use one circuit topology and observable. The same selected layer/gate intervention is applied to both inputs. Both feature and weight bindings may differ; inspect them before attributing a prediction difference to inputs alone. Derivatives describe final outputs, not intermediate steps. Full amplitudes, phases and reduced states are in the downloaded JSON.</p>
</section>
<section class="card circuit-card" aria-label="Circuit execution explorer">
 <div class="toolbar"><button id="reset" aria-label="Go to initial state">Reset</button><button id="back" aria-label="Previous step">← Back</button><button id="play" aria-label="Play execution">Play</button><button id="next" aria-label="Next step">Next →</button><label for="granularity">Navigate by</label><select id="granularity"><option value="gate">Gate</option><option value="block">Annotated block</option></select><button id="download">Download trace JSON</button></div>
 <div id="circuit" aria-label="Clickable circuit diagram"></div><input id="step" type="range" min="0" value="0" aria-label="Execution step">
 <div class="step-line"><strong id="step-title"></strong><span class="caption" id="status" aria-live="polite"></span></div>
 <div class="caption" id="block-label"></div><div class="blocks" id="blocks"></div>
 <div class="parameters" id="parameters" aria-label="Parameter highlighting"></div>
</section>
<section class="card circuit-card" aria-label="Circuit interventions">
<h2>What if I change this circuit?</h2>
<label for="variant">Calculated alternative</label> <select id="variant"><option value="original">Original circuit</option></select>
<p id="variant-summary" class="caption" aria-live="polite"></p>
<p class="caption">Each option is one independent edit of the original bound circuit, recomputed by NexQus before export. Choices do not accumulate. Parameter chips retain the original binding as context; the edited gate label shows its actual angle. Add angle presets or gate/block removals with the Python interventions option.</p>
</section>
<section class="card circuit-card" id="sensitivity-panel" hidden>
<h2>Original prediction: input and parameter sensitivity</h2>
<p id="sensitivity-scope" class="caption"></p>
<div class="scroll"><table><thead><tr><th>Parameter / role</th><th>Value</th><th>Signed derivative</th><th>Local magnitude</th></tr></thead><tbody id="sensitivity-rows"></tbody></table></div>
<p id="sensitivity-note" class="caption"></p>
</section>
<div class="layout"><div class="stack">
<section class="card"><h2>Before → after the selected gate</h2><p class="caption" id="transition-label"></p>
<div class="scroll"><table><thead><tr><th>Basis</th><th>P before</th><th>P after</th><th>ΔP</th><th>Amplitude before → after</th></tr></thead><tbody id="transition"></tbody></table></div>
<p class="caption" id="transition-coverage"></p>
<div class="scroll"><table><thead><tr><th>Qubit</th><th>Bloch (x, y, z) before → after</th><th>Entropy before → after (bits)</th></tr></thead><tbody id="qubit-transition"></tbody></table></div>
</section>
 <section class="card"><h2>What changed?</h2><div id="explanation" class="explanation"></div><div class="two"><div class="metric"><strong id="tv"></strong><span>Probability change · total variation</span></div><div class="metric"><strong id="fidelity"></strong><span>State fidelity to previous step</span></div></div><p class="caption" id="phase-change"></p></section>
 <section class="card"><div class="toolbar"><h2>Measurement probabilities</h2><label for="basis-order" class="caption">Show states</label><select id="basis-order"><option value="probability">Largest probability</option><option value="basis">Basis order</option></select></div><svg id="probability" class="plot" viewBox="0 0 640 280" role="img" aria-label="Basis measurement probabilities"></svg><div class="caption" id="coverage"></div></section>
 <section class="card"><h2>Amplitudes and relative phase</h2><div class="two"><svg id="phase" class="plot" viewBox="0 0 260 250" role="img" aria-label="Complex amplitude plane with relative phases"></svg><div><p class="caption">Radius is amplitude magnitude √P. Angle is relative phase. Colours match the basis states below.</p><p class="caption phase-note" id="phase-reference"></p><p class="caption">Zero-probability phases are undefined. A change in global phase alone has no observable effect.</p></div></div><div class="scroll"><table><thead><tr><th>Basis</th><th>Probability</th><th>Amplitude Re</th><th>Amplitude Im</th><th>Relative phase</th></tr></thead><tbody id="amplitudes"></tbody></table></div></section>
</div><div class="stack">
 <section class="card"><h2>Readout through the circuit</h2><p class="caption" id="observable"></p><div class="metric"><strong id="expectation"></strong><span>Exact expectation · not a sampled estimate</span></div><svg id="readout" class="plot" viewBox="0 0 420 110" role="img" aria-label="Observable expectation at every circuit step"></svg><p class="caption">This is the selected Pauli expectation or all-zero probability readout. A classifier or output transform would need its own documented mapping.</p></section>
 <section class="card"><h2>Inside each qubit</h2><p class="caption">Projected Bloch spheres describe reduced states. Entropy below is qubit-versus-rest entanglement entropy because the global state is pure.</p><div id="qubits" class="qubits"></div></section>
 <section class="card"><h2>Z-basis correlation</h2><p class="caption">Covariance ⟨ZᵢZⱼ⟩ − ⟨Zᵢ⟩⟨Zⱼ⟩. Correlation depends on measurement basis; it does not certify entanglement.</p><svg id="correlation" class="plot covariance" viewBox="0 0 340 340" role="img" aria-label="Z-basis covariance heatmap"></svg><div class="legend"><span>−1 · opposite</span><span>0</span><span>+1 · same</span></div></section>
</div></div>
<details><summary>How to read this report</summary><ul id="notes"></ul><p class="caption">Block boundaries and feature/weight roles are annotations supplied by the report author. Highlighting shows where a parameter occurs, not how important it is. Opt-in sensitivity reports calculate local derivatives of the final expectation, not global feature attribution or real-world causal effects. Basis views show at most 16 states; the downloadable trace contains every basis state. Playback follows recorded operations, not hardware timing or the critical-path schedule.</p></details><footer id="footer"></footer></main>
<script type="application/json" id="trace-data">__TRACE_DATA__</script>
<script>
'use strict';
(()=>{
const suite=JSON.parse(document.getElementById('trace-data').textContent); let original=suite,d=original; const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
let step=0,highlight=null,timer=null; const last=d.frames.length-1;
function text(id,value){$(id).textContent=value;}
function svg(tag,attributes={},value){let e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attributes))e.setAttribute(k,String(v));if(value!==undefined)e.textContent=value;return e;}
function add(root,tag,attrs,value){let e=svg(tag,attrs,value);root.append(e);return e;}
function fmt(x,digits=4){if(x===null)return '—';return Math.abs(x)<1e-12?'0':Number(x).toFixed(digits).replace(/\.?0+$/,'');}
function color(i){let h=((i*137.508+250)%360)/60,c=.506,x=c*(1-Math.abs(h%2-1)),m=.207;let rgb=h<1?[c,x,0]:h<2?[x,c,0]:h<3?[0,c,x]:h<4?[0,x,c]:h<5?[x,0,c]:[c,0,x];return 'rgb('+rgb.map(v=>Math.round((v+m)*255)).join(',')+')';}
function label(i){return i.toString(2).padStart(d.n_qubits,'0');}
function stop(){if(timer!==null)clearInterval(timer);timer=null;text('play','Play');$('play').setAttribute('aria-label','Play execution');}
function targets(){return $('granularity').value==='block'?[0,...d.blocks.map(b=>b.stop),last].filter((v,i,a)=>i===0||v!==a[i-1]):Array.from({length:last+1},(_,i)=>i);}
function move(direction){let a=targets();let next=direction>0?a.find(v=>v>step):a.filter(v=>v<step).pop();if(next===undefined){stop();return false;}step=next;render();return true;}
function circuit(){let root=$('circuit');root.replaceChildren();let start=step===0?0:Math.floor((step-1)/12)*12,end=Math.min(last,start+12),w=Math.max(360,(end-start)*104+85),h=d.n_qubits*42+42;
 let s=svg('svg',{viewBox:`0 0 ${w} ${h}`,width:w,height:h,role:'group','aria-label':`Circuit operations ${start+1} to ${end}`});root.append(s);
 for(let q=0;q<d.n_qubits;q++){let y=30+42*q;add(s,'text',{x:8,y:y+4,fill:'#626779','font-size':12},`q${q}`);add(s,'line',{x1:48,y1:y,x2:w-10,y2:y,stroke:'#ccd0df'});}
 for(let i=start;i<end;i++){let op=d.operations[i],x=72+(i-start)*104,active=i+1===step,hit=highlight!==null&&op.parameters.includes(highlight),ys=op.wires.map(q=>30+42*q);
 let g=add(s,'g',{role:'button',tabindex:0,'aria-label':`Step ${i+1}: ${op.label}, wires ${op.wires.join(', ')}`});g.style.cursor='pointer';g.onclick=()=>{stop();step=i+1;render();};g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();g.onclick();}};
 if(ys.length>1)add(g,'line',{x1:x,y1:Math.min(...ys),x2:x,y2:Math.max(...ys),stroke:active?'#6150c8':'#81879c','stroke-width':2});
 for(let k=0;k<ys.length;k++){let y=ys[k];add(g,'rect',{x:x-39,y:y-14,width:78,height:28,rx:6,fill:active?'#6150c8':hit?'#eee8ff':'#fff',stroke:active||hit?'#6150c8':'#aab0c1','stroke-width':hit?2:1});let name=op.label.length>11?op.label.slice(0,10)+'…':op.label;add(g,'text',{x,y:y+4,'text-anchor':'middle','font-size':10,fill:active?'white':'#202332'},name+(ys.length>1?`:${k}`:''));}
 add(g,'title',{},`${i+1}. ${op.label}; ordered wires ${op.wires.join(', ')}; ${op.action}`);
 }
}
function parameters(){let root=$('parameters');root.replaceChildren();if(!d.parameters.length){let p=document.createElement('span');p.className='caption';p.textContent='No symbolic parameters in this trace.';root.append(p);return;}
 for(let p of d.parameters){let b=document.createElement('button');b.className='chip'+(highlight===p.name?' active':'');b.textContent=`${p.role}: ${p.name} = ${fmt(p.value)}`;b.setAttribute('aria-pressed',String(highlight===p.name));b.onclick=()=>{highlight=highlight===p.name?null:p.name;render();};root.append(b);}}
function basis(frame){let indices=Array.from({length:frame.probabilities.length},(_,i)=>i);if($('basis-order').value==='probability')indices.sort((a,b)=>frame.probabilities[b]-frame.probabilities[a]||a-b);return indices.slice(0,16);}
function probabilities(f,indices){let s=$('probability');s.replaceChildren();let width=560/indices.length;add(s,'line',{x1:45,y1:185,x2:620,y2:185,stroke:'#ccd0df'});
 for(let p of [0,.5,1]){let y=185-155*p;add(s,'line',{x1:45,y1:y,x2:620,y2:y,stroke:'#edf0f5'});add(s,'text',{x:37,y:y+4,'text-anchor':'end',fill:'#626779','font-size':11},p);}
 indices.forEach((i,j)=>{let p=f.probabilities[i],x=50+j*width,b=add(s,'rect',{x,y:185-p*155,width:Math.max(2,width-5),height:p*155,fill:color(i),rx:3});add(b,'title',{},`${label(i)}: P=${p}`);add(s,'text',{x:x+(width-5)/2,y:201,'text-anchor':'end',transform:`rotate(-35 ${x+(width-5)/2} 201)`,'font-size':Math.min(11,80/d.n_qubits+3),fill:'#626779'},label(i));});
 let coverage=indices.reduce((sum,i)=>sum+f.probabilities[i],0);text('coverage',`Showing ${indices.length} of ${f.probabilities.length} basis states · ${(coverage*100).toFixed(2)}% of total probability displayed.`);
}
function phase(f,indices){let s=$('phase');s.replaceChildren();add(s,'circle',{cx:122,cy:120,r:88,fill:'#fafaff',stroke:'#dce0e9'});add(s,'line',{x1:20,y1:120,x2:226,y2:120,stroke:'#dce0e9'});add(s,'line',{x1:122,y1:20,x2:122,y2:220,stroke:'#dce0e9'});add(s,'text',{x:224,y:136,'font-size':11,fill:'#626779'},'Re');add(s,'text',{x:128,y:22,'font-size':11,fill:'#626779'},'Im');
 for(let i of indices){let p=f.probabilities[i],angle=f.phases[i];if(angle===null)continue;let r=Math.sqrt(p)*88,x=122+r*Math.cos(angle),y=120-r*Math.sin(angle);add(s,'line',{x1:122,y1:120,x2:x,y2:y,stroke:color(i),'stroke-width':2,opacity:.7});let c=add(s,'circle',{cx:x,cy:y,r:4,fill:color(i)});add(c,'title',{},`${label(i)}: magnitude ${Math.sqrt(p)}, relative phase ${angle} radians`);}
 text('phase-reference',`Phase reference at this step: |${f.phase_reference}⟩. The displayed complex plane removes this reference amplitude's phase; table Re/Im values retain the original simulator convention.`);
 let body=$('amplitudes');body.replaceChildren();for(let i of indices){let row=document.createElement('tr');let values=[label(i),fmt(f.probabilities[i],6),fmt(f.amplitudes[i][0],6),fmt(f.amplitudes[i][1],6),f.phases[i]===null?'undefined':fmt(f.phases[i],5)+' rad'];values.forEach((v,k)=>{let cell=document.createElement('td');cell.textContent=v;if(k===0)cell.style.color=color(i);row.append(cell);});body.append(row);}
}
function qubits(f){let root=$('qubits');root.replaceChildren();for(let q of f.qubits){let [x,y,z]=q.bloch,box=document.createElement('div');box.className='qubit';let name=document.createElement('strong');name.textContent=`q${q.wire}`;box.append(name);
 let s=svg('svg',{viewBox:'0 0 180 120',role:'img','aria-label':`q${q.wire} Bloch vector x=${fmt(x)}, y=${fmt(y)}, z=${fmt(z)}`});box.append(s);add(s,'circle',{cx:90,cy:58,r:37,fill:'#faf9ff',stroke:'#d7d0ef'});add(s,'ellipse',{cx:90,cy:58,rx:37,ry:37/Math.sqrt(3),fill:'none',stroke:'#d7d0ef'});
 let ax=37/Math.sqrt(2),ay=37/Math.sqrt(6),az=37*Math.sqrt(2/3);
 for(let [dx,dy,t] of [[ax,ay,'x'],[-ax,ay,'y'],[0,-az,'z']]){add(s,'line',{x1:90-dx,y1:58-dy,x2:90+dx,y2:58+dy,stroke:'#c9c2dc'});add(s,'text',{x:90+dx+(dx<0?-12:4),y:58+dy+(dy<0?-5:10),'font-size':10,fill:'#626779'},t);}
 let px=90+ax*x-ax*y,py=58+ay*x+ay*y-az*z;add(s,'line',{x1:90,y1:58,x2:px,y2:py,stroke:'#6150c8','stroke-width':3});add(s,'circle',{cx:px,cy:py,r:4,fill:'#6150c8'});
 let p=document.createElement('div');p.className='caption';p.textContent=`x ${fmt(x)} · y ${fmt(y)} · z ${fmt(z)}`;box.append(p);let metric=document.createElement('div');metric.className='caption';metric.textContent=`Purity ${fmt(q.purity)} · entropy ${fmt(q.entropy)} bits`;box.append(metric);root.append(box);}}
function covariance(f){let s=$('correlation');s.replaceChildren();let side=270/d.n_qubits;for(let i=0;i<d.n_qubits;i++){add(s,'text',{x:50+(i+.5)*side,y:28,'text-anchor':'middle','font-size':11,fill:'#626779'},`q${i}`);add(s,'text',{x:34,y:48+(i+.5)*side,'text-anchor':'end','font-size':11,fill:'#626779'},`q${i}`);
 for(let j=0;j<d.n_qubits;j++){let v=f.z_covariance[i][j],a=Math.min(1,Math.abs(v)),fill=i===j?'#eff0f5':v>=0?`rgba(49,118,190,${.07+.88*a})`:`rgba(198,118,50,${.07+.88*a})`;let r=add(s,'rect',{x:50+j*side,y:42+i*side,width:side-2,height:side-2,rx:3,fill});add(r,'title',{},i===j?`q${i}: diagonal variance omitted`:`q${i},q${j}: covariance ${v}`);if(d.n_qubits<=6)add(s,'text',{x:50+(j+.5)*side,y:46+(i+.5)*side,'text-anchor':'middle','font-size':12,fill:a>.65&&i!==j?'white':'#424858'},i===j?'—':fmt(v,2));}}
}
function readout(f){text('observable',d.observable==='P0'?'All-zero measurement probability P(00…0)':`Pauli observable ${d.observable} · q0 first`);text('expectation',fmt(f.expectation,6));let s=$('readout');s.replaceChildren();add(s,'line',{x1:12,y1:55,x2:408,y2:55,stroke:'#dce0e9'});let x=i=>12+396*i/Math.max(1,last),y=v=>55-v*42;let pts=d.frames.map((f,i)=>`${x(i)},${y(f.expectation)}`).join(' ');add(s,'polyline',{points:pts,fill:'none',stroke:'#6150c8','stroke-width':2});add(s,'circle',{cx:x(step),cy:y(f.expectation),r:5,fill:'#6150c8'});add(s,'text',{x:12,y:107,fill:'#626779','font-size':11},'Initial');add(s,'text',{x:408,y:107,'text-anchor':'end',fill:'#626779','font-size':11},'Final');}
function transition(f){
 const before=d.frames[Math.max(0,step-1)], body=$('transition');body.replaceChildren();
 text('transition-label',step?`Input to operation ${step} → output of operation ${step}. Amplitudes retain the simulator convention; a global phase has no observable effect.`:'Initial state: there is no preceding gate.');
 const indices=Array.from({length:f.probabilities.length},(_,i)=>i).sort((a,b)=>Math.abs(f.probabilities[b]-before.probabilities[b])-Math.abs(f.probabilities[a]-before.probabilities[a])||f.probabilities[b]-f.probabilities[a]||a-b).slice(0,16);
 const complex=a=>`${fmt(a[0],5)} ${a[1]<0?'−':'+'} ${fmt(Math.abs(a[1]),5)}i`;
 for(const i of indices){const row=document.createElement('tr');for(const value of [label(i),fmt(before.probabilities[i],6),fmt(f.probabilities[i],6),fmt(f.probabilities[i]-before.probabilities[i],6),complex(before.amplitudes[i])+' → '+complex(f.amplitudes[i])]){const c=document.createElement('td');c.textContent=value;row.append(c);}body.append(row);}
 text('transition-coverage',`Showing ${indices.length} of ${f.probabilities.length} states, ordered by absolute probability change. The other panels and JSON retain phase information and the full trace.`);
 const qbody=$('qubit-transition');qbody.replaceChildren();
 for(const q of f.qubits){const b=before.qubits[q.wire],row=document.createElement('tr');for(const value of ['q'+q.wire,b.bloch.map(x=>fmt(x)).join(', ')+' → '+q.bloch.map(x=>fmt(x)).join(', '),fmt(b.entropy)+' → '+fmt(q.entropy)]){const c=document.createElement('td');c.textContent=value;row.append(c);}qbody.append(row);}
}
function variantSummary(){
 const base=original.frames[last],current=d.frames[last];
 const tv=current.probabilities.reduce((sum,p,i)=>sum+Math.abs(p-base.probabilities[i]),0)/2;
 text('variant-summary',d===original?(original.alternatives?.length?'Original circuit selected. Choose an alternative to inspect its full recalculated trace.':'No alternatives were exported. Add interventions when calling circuit.explain().'):`Final output versus original: probability total variation ${fmt(tv,6)}; observable expectation ${fmt(base.expectation,6)} → ${fmt(current.expectation,6)}. A difference on this input is not a general gate-importance score.`);
}
function sensitivity(){
 const info=original.sensitivity,panel=$('sensitivity-panel');panel.hidden=!info;
 if(!info)return;
 text('sensitivity-scope',info.scope+' · output '+fmt(info.output,6)+'. These values remain for the original circuit when an alternative is selected.');
 text('sensitivity-note',info.method+'. '+info.interpretation);
 const root=$('sensitivity-rows');root.replaceChildren();
 const ranked=[...info.parameters].sort((a,b)=>Math.abs(b.derivative)-Math.abs(a.derivative)||a.name.localeCompare(b.name));
 const scale=Math.max(1e-15,...ranked.map(p=>Math.abs(p.derivative)));
 for(const p of ranked){const row=document.createElement('tr');for(const v of [p.name+' / '+p.role,fmt(p.value,6),fmt(p.derivative,6)]){const cell=document.createElement('td');cell.textContent=v;row.append(cell);}
 const cell=document.createElement('td'),bar=document.createElement('progress');bar.max=scale;bar.value=Math.abs(p.derivative);bar.setAttribute('aria-label',p.name+' absolute local derivative');cell.append(bar);row.append(cell);root.append(row);}
}
function comparison(){
 const cases=suite.comparisons||[];$('comparison-panel').hidden=!cases.length;if(!cases.length)return;
 const item=cases[Number($('comparison').value)||0],variant=$('variant').value,
 right=variant==='original'?item.trace:item.trace.alternatives[Number(variant)].trace,left=d,a=left.frames[step],b=right.frames[step];
 const tv=a.probabilities.reduce((v,p,i)=>v+Math.abs(p-b.probabilities[i]),0)/2;
 text('comparison-summary',item.name+' · step '+step+' · probability total variation '+fmt(tv,6)+' · final readout difference '+fmt(right.frames[last].expectation-left.frames[last].expectation,6));
 const describe=(trace,f)=>'Gate: '+(step?trace.operations[step-1].label:'Initial state')+'; step expectation '+fmt(f.expectation,6)+'; final prediction '+fmt(trace.frames[last].expectation,6);
 text('compare-left',describe(left,a));text('compare-right',describe(right,b));
 const parameters=$('comparison-parameters');parameters.replaceChildren();
 const derivatives=trace=>Object.fromEntries((trace.sensitivity?.parameters||[]).map(p=>[p.name,p.derivative]));const ga=derivatives(left),gb=derivatives(right);
 const rightParams=Object.fromEntries(right.parameters.map(p=>[p.name,p]));
 function row(root,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td);}root.append(tr);}
 for(const p of left.parameters)row(parameters,[p.name+' / '+p.role,fmt(p.value,6),fmt(rightParams[p.name].value,6),ga[p.name]===undefined?'not computed':fmt(ga[p.name],6),gb[p.name]===undefined?'not computed':fmt(gb[p.name],6)]);
 const indices=Array.from({length:a.probabilities.length},(_,i)=>i).sort((i,j)=>Math.max(a.probabilities[j],b.probabilities[j])-Math.max(a.probabilities[i],b.probabilities[i])||i-j).slice(0,16);
 const body=$('comparison-probabilities');body.replaceChildren();for(const i of indices)row(body,[label(i),fmt(a.probabilities[i],6),fmt(b.probabilities[i],6),fmt(b.probabilities[i]-a.probabilities[i],6)]);
 text('comparison-coverage',`Showing ${indices.length} of ${a.probabilities.length} states; displayed mass left ${fmt(indices.reduce((v,i)=>v+a.probabilities[i],0),6)}, right ${fmt(indices.reduce((v,i)=>v+b.probabilities[i],0),6)}.`);
}
function render(){sensitivity();comparison();let f=d.frames[step],op=step?d.operations[step-1]:null;text('step-title',step?`${step}. ${op.label}`:'Initial state');text('status',`Step ${step} / ${last}`);text('explanation',f.explanation);text('tv',fmt(f.probability_change,6));text('fidelity',fmt(f.fidelity_to_previous,6));text('phase-change',f.phase_change===null?'Relative-phase change: not defined across fewer than two shared nonzero basis amplitudes.':`Largest relative-phase change on shared nonzero amplitudes: ${fmt(f.phase_change,6)} radians.`);let block=op&&op.block!==null?d.blocks[op.block]:null;text('block-label',block?`Author annotation: ${block.name} · ${block.role}`:'No block annotation at this step.');$('step').value=step;$('back').disabled=step===0;$('next').disabled=step===last;$('play').disabled=last===0;circuit();parameters();let indices=basis(f);probabilities(f,indices);phase(f,indices);qubits(f);covariance(f);readout(f);transition(f);variantSummary();for(let b of $('blocks').children)b.classList.toggle('active',Number(b.dataset.stop)===step);}
function variants(){
 const select=$('variant');select.replaceChildren();const base=document.createElement('option');base.value='original';base.textContent='Original circuit';select.append(base);
 for(const [i,item] of (original.alternatives||[]).entries()){const option=document.createElement('option');option.value=String(i);option.textContent=item.label;select.append(option);}
 select.disabled=!(original.alternatives||[]).length;
}
variants();
$('variant').onchange=()=>{stop();d=$('variant').value==='original'?original:original.alternatives[Number($('variant').value)].trace;render();text('depth',d.depth);};
text('title',d.title);document.title=d.title+' · NexQus';text('qubit-count',d.n_qubits);text('gate-count',last);text('depth',d.depth);text('footer',`NexQus SDK ${d.sdk_version} · schema ${d.schema_version} · ${d.execution.backend} / ${d.execution.device} · double precision · offline report · no external assets`);$('step').max=last;
for(let note of d.notes){let li=document.createElement('li');li.textContent=note;$('notes').append(li);}
for(let b of d.blocks){let button=document.createElement('button');button.className='chip';button.textContent=b.name;button.dataset.stop=b.stop;button.onclick=()=>{stop();step=b.stop;render();};$('blocks').append(button);}
$('granularity').querySelector('option[value="block"]').disabled=d.blocks.length===0;
$('step').oninput=e=>{stopTraining();stop();step=Number(e.target.value);render();};$('back').onclick=()=>{stopTraining();stop();move(-1);};$('next').onclick=()=>{stopTraining();stop();move(1);};$('reset').onclick=()=>{stopTraining();stop();step=0;render();};$('basis-order').onchange=render;$('granularity').onchange=()=>stop();
$('play').onclick=()=>{stopTraining();if(timer!==null){stop();return;}if(step===last)step=0;text('play','Pause');$('play').setAttribute('aria-label','Pause execution');render();timer=setInterval(()=>{if(!move(1)||step===last)stop();},850);};
$('download').onclick=()=>{let url=URL.createObjectURL(new Blob([JSON.stringify(suite.comparisons||suite.training_history?suite:d,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='nexqus-explanation.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
let trainingTimer=null;
function stopTraining(){if(trainingTimer!==null)clearInterval(trainingTimer);trainingTimer=null;text('training-play','Play checkpoints');}
function selectCheckpoint(){
 stop();const index=$('checkpoint').value,record=index==='final'?null:suite.training_history[Number(index)];
 original=record?record.trace:suite;d=original;variants();highlight=null;
 text('training-summary',record?`Epoch ${record.epoch} · recorded loss ${record.loss===undefined?'not supplied':fmt(record.loss,8)} · prediction ${fmt(d.frames[last].expectation,6)}`:'Supplied final bindings selected.');
 render();text('depth',d.depth);
}
for(const [i,item] of (suite.comparisons||[]).entries()){const option=document.createElement('option');option.value=String(i);option.textContent=item.name;$('comparison').append(option);}
$('comparison').onchange=render;
const history=suite.training_history||[];$('training-panel').hidden=!history.length;
for(const [i,record] of history.entries()){const option=document.createElement('option');option.value=String(i);option.textContent='Epoch '+record.epoch;$('checkpoint').append(option);}
$('checkpoint').onchange=()=>{stopTraining();selectCheckpoint();};
$('training-play').onclick=()=>{if(trainingTimer!==null){stopTraining();return;}stop();let index=0;const advance=()=>{if(index>=history.length){stopTraining();return;}$('checkpoint').value=String(index++);selectCheckpoint();};advance();text('training-play','Pause checkpoints');trainingTimer=setInterval(advance,1000);};
const lossPlot=$('training-loss'),known=history.filter(r=>r.loss!==undefined);
if(known.length){const minEpoch=history[0].epoch,maxEpoch=history[history.length-1].epoch,maxLoss=Math.max(1e-15,...known.map(r=>r.loss));
 for(const r of known)add(lossPlot,'circle',{cx:20+600*(r.epoch-minEpoch)/Math.max(1,maxEpoch-minEpoch),cy:95-75*(r.loss/maxLoss),r:4,fill:'#6150c8'});
 add(lossPlot,'text',{x:20,y:115,fill:'#626779','font-size':11},'Recorded loss by epoch (missing losses omitted)');}
text('training-summary','Supplied final bindings selected.');
render();
})();
</script></body></html>'''
