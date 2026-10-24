const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM,VirtualConsole}=require('jsdom');
const errors=[];let tick=null;
const vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(String(e)));
const dom=new JSDOM(fs.readFileSync(process.argv[2]||'.release-artifacts/maze-walk/walk.html','utf8'),{
 runScripts:'dangerously',virtualConsole:vc,beforeParse(w){w.setInterval=fn=>{tick=fn;return 1;};w.clearInterval=()=>{tick=null;};}
});
const w=dom.window,$=id=>w.document.getElementById(id),d=w.walkLesson.data;
const run=w.mazeRun;
assert.equal(run.step,0);assert.match($('run-c-status').textContent,/Walker at/);
for(const [a,b] of run.trail.slice(1).map((b,i)=>[run.trail[i],b])){
 assert(a===b||Math.abs(Math.floor(a/d.rows[0].length)-Math.floor(b/d.rows[0].length))+Math.abs(a%d.rows[0].length-b%d.rows[0].length)===1);
 assert.notEqual(d.rows[Math.floor(b/d.rows[0].length)][b%d.rows[0].length],'#');
}
const quantumExit=run.quantumExit;
let ticks=0;$('run-play').click();while(tick&&ticks++<=d.steps)tick();
assert(ticks<=d.steps+1);assert($('run-play').disabled);
assert.match($('run-status').textContent,/Run finished/);
if(quantumExit!==null){assert.match($('run-q-status').textContent,/collapsed onto G/);assert($('run-quantum').querySelector('rect[stroke="#51e3a3"]'));}
else assert.match($('run-q-status').textContent,/No exit/);
$('run-reset').click();assert.equal(run.step,0);assert(!($('run-play').disabled));
$('run-seed').value='-1';$('run-reset').click();assert.match($('run-status').textContent,/integer seed/);
$('run-seed').value='7';$('run-reset').click();assert.equal(run.quantumExit,quantumExit);
assert.equal($('qhit').textContent,'0.000%');assert($('back').disabled);
$('next').click();assert.equal(w.walkLesson.step,1);
assert.equal($('qhit').textContent,$('chit').textContent);
$('range').value=String(d.steps);$('range').dispatchEvent(new w.Event('input'));
assert($('next').disabled);assert.equal($('qhit').textContent,(d.quantum_hit*100).toFixed(3)+'%');
assert.match($('verdict').textContent,/not recovered routes/);
assert.equal($('results').rows.length,3);assert.match($('timing').textContent,/9 runs after 2 warmups/);
$('qgrid').querySelector('[aria-label^="Cell 0, 1:"]').dispatchEvent(new w.MouseEvent('click',{bubbles:true}));assert.match($('ports').textContent,/Cell \(0, 1\)/);
assert.match($('ports').textContent,/Coin outputs before shifting/);
$('play').click();tick();assert.equal(w.walkLesson.step,1);$('play').click();assert.equal(tick,null);
for(const e of w.document.querySelectorAll('svg,svg *'))for(const a of e.attributes)assert(!/NaN|Infinity/.test(a.value));
assert.equal(errors.length,0,errors.join('\n'));dom.window.close();
console.log('Quantum walk UI passed: navigation, exact data, cell amplitudes, playback, counts, timing and finite SVG.');
