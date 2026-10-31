/* Offline DOM regression tests. jsdom does not load external resources. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {JSDOM,VirtualConsole}=require('jsdom');
const directory=process.argv[2]||'.release-artifacts/explanations';
const errors=[];
function open(source){
 const virtualConsole=new VirtualConsole();virtualConsole.on('jsdomError',e=>errors.push(String(e)));
 let interval=null,blob=null,download=null;
 const dom=new JSDOM(source,{runScripts:'dangerously',virtualConsole,beforeParse(w){
  w.setInterval=fn=>{interval=fn;return 1;};w.clearInterval=()=>{interval=null;};
  w.URL.createObjectURL=value=>{blob=value;return 'blob:local-test';};w.URL.revokeObjectURL=()=>{};
  w.HTMLAnchorElement.prototype.click=function(){download=this.download;};
 }});
 const w=dom.window,d=w.document;
 return {dom,w,d,get:id=>d.getElementById(id),tick:()=>interval&&interval(),
  payload:()=>new Promise((resolve,reject)=>{let r=new w.FileReader();r.onload=()=>resolve(JSON.parse(r.result));r.onerror=reject;r.readAsText(blob);}),download:()=>download};
}
function setStep(ui,value){ui.get('step').value=String(value);ui.get('step').dispatchEvent(new ui.w.Event('input'));}
function checkSVG(ui){
 for(const e of ui.d.querySelectorAll('svg *'))for(const attr of e.attributes)
  assert(!/NaN|Infinity/.test(attr.value),`Invalid SVG ${attr.name}: ${attr.value}`);
 for(const s of ui.d.querySelectorAll('.qubit svg')){
  const point=s.querySelector('circle:last-child');
  const radius=Math.hypot(Number(point.getAttribute('cx'))-90,Number(point.getAttribute('cy'))-58);
  assert(radius<=37+1e-9,'Bloch projection left the unit sphere');
 }
}
function replaceData(source,data){return source.replace(/(<script type="application\/json" id="trace-data">)[\s\S]*?(<\/script>)/,
 (_,a,b)=>a+JSON.stringify(data).replace(/</g,'\\u003c').replace(/>/g,'\\u003e').replace(/&/g,'\\u0026')+b);}
(async()=>{
 const source=fs.readFileSync(path.join(directory,'bell.html'),'utf8'),bell=open(source);
 const predictionUI=open(fs.readFileSync(path.join(directory,'qnn.html'),'utf8'));
 assert.equal(predictionUI.get('sensitivity-panel').hidden,false);
 assert.equal(predictionUI.get('sensitivity-rows').children.length,5);
 assert.match(predictionUI.get('sensitivity-rows').textContent,/x \/ feature/);
 assert.match(predictionUI.get('sensitivity-rows').textContent,/w0 \/ weight/);
 assert.equal(predictionUI.get('variant').options.length,5);
 const originalSensitivity=predictionUI.get('sensitivity-rows').textContent;
 predictionUI.get('variant').value='1';predictionUI.get('variant').dispatchEvent(new predictionUI.w.Event('change'));
 assert.equal(predictionUI.get('sensitivity-rows').textContent,originalSensitivity);
 assert.match(predictionUI.get('variant-summary').textContent,/Final output versus original/);
 assert.equal(bell.get('sensitivity-panel').hidden,true);

 assert.equal(predictionUI.get('comparison-panel').hidden,false);
 assert.match(predictionUI.get('compare-right').textContent,/Gate:/);
 assert.equal(predictionUI.get('training-panel').hidden,false);
 assert.equal(predictionUI.get('checkpoint').options.length,6);
 setStep(predictionUI,3);
 assert.match(predictionUI.get('comparison-summary').textContent,/step 3/);
 assert.equal(predictionUI.get('comparison-parameters').children.length,5);
 predictionUI.get('checkpoint').value='0';predictionUI.get('checkpoint').dispatchEvent(new predictionUI.w.Event('change'));
 assert.match(predictionUI.get('training-summary').textContent,/Epoch 0/);
 assert.equal(predictionUI.get('step').value,'3');
 assert.equal(predictionUI.get('variant').options.length,1);
 assert.notEqual(predictionUI.get('sensitivity-rows').textContent,originalSensitivity);
 predictionUI.get('checkpoint').value='final';predictionUI.get('checkpoint').dispatchEvent(new predictionUI.w.Event('change'));
 assert.equal(predictionUI.get('sensitivity-rows').textContent,originalSensitivity);
 assert.equal(predictionUI.get('variant').options.length,5);
 predictionUI.get('training-play').click();predictionUI.tick();
 assert.match(predictionUI.get('training-summary').textContent,/Epoch 10/);
 predictionUI.get('training-play').click();
 predictionUI.get('download').click();const series=await predictionUI.payload();
 assert.equal(series.training_history.length,5);assert.equal(series.comparisons.length,1);
 checkSVG(predictionUI);predictionUI.dom.window.close();

 const lab=open(fs.readFileSync(path.join(directory,'laboratory.html'),'utf8'));
 setStep(lab,2);
 assert.match(lab.get('qubit-transition').textContent,/0 → 0.332/);
 assert.equal(lab.get('variant').options.length,5);
 lab.get('variant').value='0';lab.get('variant').dispatchEvent(new lab.w.Event('change'));
 assert.equal(lab.get('expectation').textContent,'1');
 assert.match(lab.get('variant-summary').textContent,/Final output versus original/);
 setStep(lab,1);assert.match(lab.get('step-title').textContent,/RX\(0\)/);
 lab.get('variant').value='3';lab.get('variant').dispatchEvent(new lab.w.Event('change'));
 setStep(lab,2);assert.match(lab.get('qubit-transition').textContent,/0 → 0/);
 assert.match(lab.get('transition').textContent,/→/);checkSVG(lab);
 lab.get('download').click();const edited=await lab.payload();assert.equal(edited.intervention.skip,true);
 lab.get('variant').value='original';lab.get('variant').dispatchEvent(new lab.w.Event('change'));
 assert.match(lab.get('explanation').textContent,/entangled/);
 lab.dom.window.close();
 assert.equal(bell.get('status').textContent,'Step 0 / 2');
 bell.get('next').click();assert.equal(bell.get('step-title').textContent,'1. H');
 bell.get('next').click();assert.equal(bell.get('step-title').textContent,'2. CX');
 assert.match(bell.get('explanation').textContent,/entangled/);
 assert.match(bell.get('qubits').textContent,/Purity 0.5 · entropy 1 bits/);
 assert.equal(bell.get('next').disabled,true);checkSVG(bell);
 bell.get('reset').click();assert.equal(bell.get('step').value,'0');
 bell.get('granularity').value='block';bell.get('next').click();assert.equal(bell.get('step').value,'1');
 bell.get('next').click();assert.equal(bell.get('step').value,'2');
 bell.get('back').click();assert.equal(bell.get('step').value,'1');
 bell.get('reset').click();let gate=bell.d.querySelector('#circuit g[role="button"]');
 gate.dispatchEvent(new bell.w.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));assert.equal(bell.get('step').value,'1');
 bell.get('reset').click();bell.get('play').click();bell.tick();assert.equal(bell.get('step').value,'1');
 bell.tick();assert.equal(bell.get('step').value,'2');assert.equal(bell.get('play').textContent,'Play');
 bell.get('download').click();const payload=await bell.payload();assert.equal(payload.frames.length,3);
 assert.equal(bell.download(),'nexqus-explanation.json');
 const svgNames=['probability','phase','correlation','readout'];
 const panels=svgNames.map((id,i)=>{let e=bell.get(id).cloneNode(true);e.setAttribute('x',i%2*620);e.setAttribute('y',50+Math.floor(i/2)*350);e.setAttribute('width',580);e.setAttribute('height',300);return e.outerHTML;});
 fs.writeFileSync(path.join(directory,'bell-plots.svg'),'<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="760"><rect width="1240" height="760" fill="white"/><text x="24" y="30" font-size="20" font-family="sans-serif">Bell final state · probability / amplitude / Z covariance / readout</text>'+panels.join('')+'</svg>');
 bell.dom.window.close();
 const interference=open(fs.readFileSync(path.join(directory,'interference.html'),'utf8'));
 setStep(interference,2);assert.match(interference.get('explanation').textContent,/Relative phases changed/);
 assert.equal(interference.get('tv').textContent,'0');
 assert.match(interference.get('phase-change').textContent,/3.141593/);
 setStep(interference,3);assert.equal(interference.get('expectation').textContent,'-1');
 checkSVG(interference);interference.dom.window.close();
 const qnn=open(fs.readFileSync(path.join(directory,'qnn.html'),'utf8'));
 assert.equal(qnn.get('parameters').children.length,5);
 const feature=[...qnn.get('parameters').children].find(e=>e.textContent.startsWith('feature: x'));
 feature.click();assert.equal(qnn.d.querySelectorAll('#parameters [aria-pressed="true"]').length,1);
 assert(qnn.d.querySelectorAll('#circuit rect[stroke-width="2"]').length>=2);
 qnn.get('granularity').value='block';qnn.get('next').click();assert.equal(qnn.get('step').value,'2');
 setStep(qnn,7);assert.match(qnn.get('block-label').textContent,/Readout rotations/);
 const qnnData=JSON.parse(qnn.get('trace-data').textContent);
 assert(Math.abs(Number(qnn.get('expectation').textContent)-qnnData.frames[7].expectation)<1e-6);
 checkSVG(qnn);qnn.dom.window.close();
 const hostile='</script><img src=x onerror="alert(1)">';
 payload.title=hostile;payload.operations[0].label=hostile;
 const escaped=open(replaceData(source,payload));setStep(escaped,1);
 assert.equal(escaped.get('title').textContent,hostile);assert.equal(escaped.d.querySelectorAll('img').length,0);
 assert.equal(escaped.d.querySelectorAll('script').length,2);escaped.dom.window.close();
 payload.operations=[];payload.blocks=[];payload.parameters=[];payload.frames=payload.frames.slice(0,1);
 const empty=open(replaceData(source,payload));assert.equal(empty.get('play').disabled,true);
 assert.equal(empty.get('next').disabled,true);assert.equal(empty.get('back').disabled,true);
 assert.equal(empty.get('status').textContent,'Step 0 / 0');checkSVG(empty);empty.dom.window.close();
 assert.deepEqual(errors,[]);
 console.log('Explainability UI passed: navigation, playback, blocks, parameter highlights, plots, readout, JSON download, empty circuit and injection safety.');
})().catch(e=>{console.error(e);process.exitCode=1;});
