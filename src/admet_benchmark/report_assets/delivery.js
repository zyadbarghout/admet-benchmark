// Native script assets also work under file://, without a server or network.
window.reportChunks = {};
const assetPromises = new Map();
function loadAsset(url, key) {
  if (!assetPromises.has(url)) assetPromises.set(url, new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.onload = () => { const result = window.reportChunks[key]; delete window.reportChunks[key]; script.remove(); resolve(result); };
    script.onerror = () => { assetPromises.delete(url); script.remove(); reject(new Error('Cannot load local report asset: '+url)); };
    document.head.append(script);
  }));
  return assetPromises.get(url);
}
async function ensureMap(view) {
  if (view && !view.coordinates && view.asset_url) Object.assign(view, await loadAsset(view.asset_url, view.id));
}
function reportLoading(message) {
  const node=$('loading-state');node.hidden=!message;node.textContent=message||'';
  $('workspace').inert=Boolean(message);
  document.querySelector('.task-tabs').inert=Boolean(message);
}
async function readyMap() {
  await ensureMap(dashboard.atlas?.views.find(v=>v.representation===$('representation').value && v.method===$('map-technique').value));
}
async function loadRecipeAssets(recipe) {
  if(recipe.report_signature!==dashboard.report_signature) return;
  await ensureMap(dashboard.atlas?.views.find(v=>v.representation===recipe.controls?.representation && v.method===recipe.controls?.['map-technique']));
}
async function startReport() {
  reportLoading('Loading molecular observations…');
  const packed=await loadAsset(dashboard.record_asset,'records');
  dashboard.records=packed.rows.map(([values, predictions])=>{
    const record=Object.fromEntries(packed.columns.map((key,i)=>[key,values[i]]));
    record.predictions=Object.fromEntries(Object.entries(predictions).map(([model, p])=>[model,Object.fromEntries(packed.prediction_fields.flatMap((key,i)=>p[i]===null?[]:[[key,p[i]]]))]));
    return record;
  });
  assetPromises.delete(dashboard.record_asset);
  await ensureMap(dashboard.atlas?.views.find(v=>v.method==='tsne') || dashboard.atlas?.views[0]);
  initialise(); initialiseWorkspace(); initialiseScientific(); initialiseViewTools();
  for(const id of ['representation','map-technique']) $(id).addEventListener('change',async event=>{
    if(event.reportReady)return;
    event.stopImmediatePropagation();
    reportLoading('Loading the selected chemical map…');
    try{
      if(id==='representation')refreshMapTechniques();
      await readyMap();
      const ready=new Event('change');ready.reportReady=true;$(id).dispatchEvent(ready);
    }catch(error){$('view-status').textContent=error.message}
    finally{reportLoading('')}
  },true);
  reportLoading('');
  document.documentElement.dataset.reportReady='true';
}
startReport().catch(error=>{reportLoading(error.message);$('loading-state').setAttribute('role','alert')});
