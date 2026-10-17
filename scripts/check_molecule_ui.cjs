/* Exercise the actual offline viewer scripts without network resources. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {JSDOM,VirtualConsole}=require('jsdom');
const directory=process.argv[2]||'.release-artifacts/molecules',errors=[];
function open(source){let interval=null,blob=null,name=null;const vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(String(e)));
 const dom=new JSDOM(source,{runScripts:'dangerously',virtualConsole:vc,beforeParse(w){w.setInterval=fn=>{interval=fn;return 1;};w.clearInterval=()=>{interval=null;};w.URL.createObjectURL=b=>{blob=b;return 'blob:test';};w.URL.revokeObjectURL=()=>{};w.HTMLAnchorElement.prototype.click=function(){name=this.download;};}});
 return{dom,w:dom.window,d:dom.window.document,get:id=>dom.window.document.getElementById(id),tick:()=>interval&&interval(),name:()=>name,
  download:()=>new Promise((resolve,reject)=>{const reader=new dom.window.FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsText(blob);})};}
function change(ui,id,value){ui.get(id).value=value;ui.get(id).dispatchEvent(new ui.w.Event('change'));}
function finite(ui){for(const e of ui.d.querySelectorAll('#scene *'))for(const a of e.attributes)assert(!/NaN|Infinity/.test(a.value));}
function clickAtom(ui,index){ui.d.querySelector(`[data-atom="${index}"]`).dispatchEvent(new ui.w.MouseEvent('click',{bubbles:true}));}
(async()=>{
 const source=fs.readFileSync(path.join(directory,'gallery.html'),'utf8'),ui=open(source);
 assert.equal(ui.get('molecule').children.length,5);assert.equal(ui.get('atom-count').textContent,'2');
 clickAtom(ui,0);clickAtom(ui,1);assert.match(ui.get('measurement').textContent,/0.7350 Å/);
 change(ui,'molecule','1');assert.match(ui.get('title').textContent,/Methane/);assert.equal(ui.get('electron-count').textContent,'10');assert.equal(ui.get('mode-count').textContent,'18');assert.equal(ui.d.querySelectorAll('[data-atom]').length,5);assert.equal(ui.d.querySelectorAll('[data-bond]').length,4);
 clickAtom(ui,0);clickAtom(ui,1);assert.match(ui.get('measurement').textContent,/1.0900 Å/);
 change(ui,'measurement-mode','angle');clickAtom(ui,1);clickAtom(ui,0);clickAtom(ui,2);assert.match(ui.get('measurement').textContent,/109.471°/);
 const before=ui.get('scene').outerHTML;ui.get('rotate').click();ui.tick();assert.notEqual(ui.get('scene').outerHTML,before);ui.get('rotate').click();assert.equal(ui.get('rotate').getAttribute('aria-pressed'),'false');
 ui.get('reset').click();const initial=ui.get('scene').outerHTML;
 ui.get('scene').dispatchEvent(new ui.w.MouseEvent('pointerdown',{clientX:10,clientY:10,bubbles:true}));ui.get('scene').dispatchEvent(new ui.w.MouseEvent('pointermove',{clientX:50,clientY:30,bubbles:true}));ui.get('scene').dispatchEvent(new ui.w.MouseEvent('pointerup',{bubbles:true}));assert.notEqual(ui.get('scene').outerHTML,initial);
 ui.get('zoom').value='130';ui.get('zoom').dispatchEvent(new ui.w.Event('input'));finite(ui);
 change(ui,'representation','atoms');assert.equal(ui.d.querySelectorAll('[data-bond]').length,0);change(ui,'representation','ball');
 ui.get('save-xyz').click();const xyz=await ui.download();assert.equal(ui.name(),'nexqus-molecule.xyz');assert.equal(xyz.split('\n')[0],'5');assert.match(xyz,/C 0.0000000000/);
 ui.get('save-json').click();const payload=JSON.parse(await ui.download());assert.equal(payload.molecule.formula,'CH4');assert.equal(payload.molecule.atoms.length,5);assert.equal(payload.molecule.energies_hartree,undefined);
 ui.get('save-svg').click();const svg=await ui.download();assert.match(svg,/<svg/);assert.match(svg,/radialGradient/);assert(!/NaN|Infinity/.test(svg));
 change(ui,'measurement-mode','distance');ui.get('reset').click();fs.writeFileSync(path.join(directory,'methane-view.svg'),ui.get('scene').outerHTML);
 change(ui,'molecule','4');assert.equal(ui.get('atom-count').textContent,'12');assert.equal(ui.get('mode-count').textContent,'72');assert.equal(ui.d.querySelectorAll('[data-atom]').length,12);assert.equal(ui.d.querySelectorAll('[data-bond]').length,12);finite(ui);
 const h2=open(fs.readFileSync(path.join(directory,'h2_vqe.html'),'utf8'));assert.match(h2.get('result-status').textContent,/ENERGY RESULTS ATTACHED/);assert.equal(h2.get('energies').children.length,3);assert.match(h2.get('calculation-note').textContent,/STO-3G/);finite(h2);h2.dom.window.close();
 // Hostile labels must remain text, including HTML/script delimiters.
 const data=JSON.parse(ui.get('molecule-data').textContent);data.molecules[0].name='</script><img src=x onerror="window.injected=1">';
 const encoded=JSON.stringify(data).replace(/</g,'\\u003c').replace(/>/g,'\\u003e').replace(/&/g,'\\u0026');
 const hostile=open(source.replace(/(<script type="application\/json" id="molecule-data">)[\s\S]*?(<\/script>)/,(_,a,b)=>a+encoded+b));
 assert.equal(hostile.w.injected,undefined);assert.equal(hostile.d.querySelectorAll('img').length,0);assert.match(hostile.get('title').textContent,/<img/);hostile.dom.window.close();ui.dom.window.close();
 assert.deepEqual(errors,[]);console.log('Molecular viewer: selection, distances, angles, rotation, drag, zoom, representations, exports and injection checks passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
