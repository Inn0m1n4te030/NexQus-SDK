/* Exercise both the offline lesson and a real local Python/NexQus server. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {spawn}=require('node:child_process');
const {JSDOM,VirtualConsole}=require('jsdom');
const errors=[];
function open(source,url){
 let tick=null,blob=null,filename=null;
 const virtualConsole=new VirtualConsole();virtualConsole.on('jsdomError',e=>errors.push(String(e)));
 const dom=new JSDOM(source,{url,runScripts:'dangerously',virtualConsole,beforeParse(w){
  w.setInterval=fn=>{tick=fn;return 1;};w.clearInterval=()=>{tick=null;};
  w.fetch=(endpoint,options)=>fetch(new URL(endpoint,url),options);
  w.URL.createObjectURL=value=>{blob=value;return 'blob:maze-test';};w.URL.revokeObjectURL=()=>{};
  w.HTMLAnchorElement.prototype.click=function(){filename=this.download;};
 }});
 return {dom,w:dom.window,get:id=>dom.window.document.getElementById(id),tick:()=>tick&&tick(),
  filename:()=>filename,payload:()=>new Promise((resolve,reject)=>{const reader=new dom.window.FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsText(blob);})};
}
function change(ui,id,value){ui.get(id).value=value;ui.get(id).dispatchEvent(new ui.w.Event('change'));}
async function until(condition){for(let i=0;i<500;i++){if(condition())return;await new Promise(resolve=>setTimeout(resolve,10));}throw new Error('UI operation timed out');}
function checkSVG(ui){for(const svg of ui.w.document.querySelectorAll('svg'))for(const e of [svg,...svg.querySelectorAll('*')])for(const a of e.attributes)assert(!/NaN|Infinity/.test(a.value),a.name+': '+a.value);}
async function startServer(python){
 const child=spawn(python,['examples/quantum_maze.py','--no-browser','--port','0'],{stdio:['ignore','pipe','pipe']});
 const url=await new Promise((resolve,reject)=>{let output='',error='';const timer=setTimeout(()=>reject(new Error('Server startup timed out: '+error)),15000);
  child.stdout.on('data',chunk=>{output+=chunk;const match=output.match(/http:\/\/127\.0\.0\.1:\d+/);if(match){clearTimeout(timer);resolve(match[0]);}});
  child.stderr.on('data',chunk=>{error+=chunk;});child.on('error',reject);child.on('exit',code=>{clearTimeout(timer);reject(new Error('Server exited '+code+': '+error));});
 }).catch(error=>{child.kill();throw error;});
 return {child,url};
}
(async()=>{
 const source=fs.readFileSync(process.argv[2]||'.release-artifacts/maze/lesson.html','utf8');
 const offline=open(source,'file:///maze-lesson.html');
 assert.equal(offline.get('candidate-count').textContent,'64');assert.equal(offline.get('valid-count').textContent,'2');
 assert(offline.get('run').disabled);assert(offline.get('wall-tool').disabled);assert(!offline.get('measure').disabled);
 const prepared=offline.get('success').textContent;offline.get('next').click();
 assert.match(offline.get('stage-title').textContent,/Mark/);assert.equal(offline.get('success').textContent,prepared);
 assert.equal(offline.get('amplitudes').querySelectorAll('rect[fill="#bb2860"]').length,2);
 offline.get('next').click();assert.match(offline.get('stage-title').textContent,/Amplify/);
 offline.get('reset').click();offline.get('play').click();offline.tick();assert.equal(offline.w.mazeLesson.stage,1);
 offline.get('play').click();offline.tick();assert.equal(offline.w.mazeLesson.stage,1);
 offline.get('step-range').value='8';offline.get('step-range').dispatchEvent(new offline.w.Event('input'));
 assert.equal(offline.get('success').textContent,'99.9%');offline.get('measure').click();
 assert.match(offline.get('measurement').textContent,/valid exit route/);assert.match(offline.get('measurement').textContent,/Seed 7/);
 offline.get('show-bfs').checked=true;offline.get('show-bfs').dispatchEvent(new offline.w.Event('change'));
 assert.equal(offline.get('maze').querySelectorAll('polyline').length,2);
 offline.get('download').click();assert.equal(offline.filename(),'nexqus-maze-experiment.json');
 assert.equal(JSON.parse(await offline.payload()).resources.classical_route_checks,64);
 offline.get('lesson').click();assert.equal(offline.filename(),'nexqus-maze-circuit.html');assert.match(await offline.payload(),/maze_oracle/);
 checkSVG(offline);offline.dom.window.close();
 const server=await startServer(process.argv[3]||'python');
 try{
  const live=open(await (await fetch(server.url)).text(),server.url);
  assert(!live.get('wall-tool').disabled);
  live.get('maze').querySelector('[aria-label="Row 1, column 2, open"]').dispatchEvent(new live.w.MouseEvent('click',{bubbles:true}));
  assert.equal(live.w.mazeLesson.data,null);assert(live.get('measure').disabled);assert.equal(live.get('candidate-count').textContent,'—');
  live.get('run').click();await until(()=>live.w.mazeLesson.data!==null&&!live.get('run').disabled);
  assert.equal(live.w.mazeLesson.data.rows[0],'S#..');assert.equal(live.w.mazeLesson.data.valid_plans,1);
  live.get('next').click();const expected=live.w.mazeLesson.data.samples[1];
  live.get('measure').click();await until(()=>/Measured/.test(live.get('measurement').textContent));
  assert.match(live.get('measurement').textContent,new RegExp(expected.bits));assert.equal(live.get('seed').value,'8');
  live.get('seed').value='-1';live.get('measure').click();await until(()=>live.get('status').className.includes('error'));
  assert.match(live.get('status').textContent,/seed/);live.get('seed').value='7';
  change(live,'preset','Choice-budget challenge');live.get('run').click();await until(()=>live.w.mazeLesson.data!==null&&!live.get('run').disabled);
  assert.equal(live.w.mazeLesson.data.valid_plans,0);assert.match(live.get('coverage').textContent,/BFS found a path/);assert(live.get('next').disabled);assert(live.get('play').disabled);
  change(live,'preset','Blocked exit');live.get('run').click();await until(()=>live.w.mazeLesson.data!==null&&!live.get('run').disabled);
  assert.match(live.get('coverage').textContent,/No exit path exists/);
  change(live,'preset','Two corridors');live.get('start-tool').click();
  live.get('maze').querySelector('[aria-label="Row 1, column 2, open"]').dispatchEvent(new live.w.MouseEvent('click',{bubbles:true}));
  live.get('goal-tool').click();live.get('maze').querySelector('[aria-label="Row 4, column 1, open"]').dispatchEvent(new live.w.MouseEvent('click',{bubbles:true}));
  live.get('run').click();await until(()=>live.w.mazeLesson.data!==null&&!live.get('run').disabled);
  assert.equal(live.w.mazeLesson.data.rows[0],'.S..');assert.equal(live.w.mazeLesson.data.rows[3],'G##.');
  checkSVG(live);live.dom.window.close();
 }finally{server.child.kill();}
 assert.deepEqual(errors,[]);
 console.log('Maze UI passed: actual SDK server, maze editing, oracle rebuild, phase/probability navigation, playback, stage measurement, seeded samples, no-path/budget distinction, downloads and SVG validity.');
})().catch(error=>{console.error(error);process.exitCode=1;});
