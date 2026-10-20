const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM,VirtualConsole}=require('jsdom');
let tick=null;const errors=[];const vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(String(e)));
const dom=new JSDOM(fs.readFileSync(process.argv[2]||'.release-artifacts/glued-trees/glued_trees.html','utf8'),{
 runScripts:'dangerously',virtualConsole:vc,beforeParse(w){w.setInterval=fn=>{tick=fn;return 1;};w.clearInterval=()=>{tick=null;};}
});
const w=dom.window,$=id=>w.document.getElementById(id),d=w.gluedLesson.data;
assert.equal($('sweep').rows.length,d.results.length);assert.equal($('verification').rows.length,3);
assert.equal($('graph').querySelectorAll('circle').length,62);assert.match($('graph-info').textContent,/62 queries/);
$('graph').querySelector('circle').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
assert.match($('node-info').textContent,/Neighbor names/);
assert($('back').disabled);$('next').click();assert.equal(w.gluedLesson.frame,1);
$('depth').value=String(d.results.length-1);$('depth').dispatchEvent(new w.Event('change'));
const r=d.results.at(-1);assert.equal($('columns').textContent,String(r.columns));
assert.equal($('register').textContent,String(r.column_qubits));assert.match($('encoding').textContent,/not simulated here/);
$('time-range').value=String(r.times.length-1);$('time-range').dispatchEvent(new w.Event('input'));
assert($('next').disabled);assert.match($('exit-info').textContent,/Uniform-time averages/);
assert.match($('exit-info').textContent,/sampled peak/);
$('qbars').querySelectorAll('rect')[1].dispatchEvent(new w.MouseEvent('click',{bubbles:true}));
assert.match($('amplitude').textContent,/Column 1:/);assert.match($('amplitude').textContent,/da\/dt/);
$('play').click();tick();assert.equal(w.gluedLesson.frame,1);$('play').click();assert.equal(tick,null);
for(const e of w.document.querySelectorAll('svg,svg *'))for(const a of e.attributes)assert(!/NaN|Infinity/.test(a.value));
assert.equal(errors.length,0,errors.join('\n'));dom.window.close();
console.log('Glued-tree UI passed: graph names, depth/time navigation, probabilities, amplitudes, scaling, timing and finite SVG.');
