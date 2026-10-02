// Task workspaces and chemical-map controls. All analyses use saved observations.
let taskMode = dashboard.endpoints.some(e=>e.task==='regression')?'regression':'classification';
let inspectorPage=0, inspectorTimer, selectedNode=null;
const representationLabels={morgan:'Morgan fingerprints',rdkit_descriptors:'RDKit descriptors',morgan_rdkit:'Morgan + RDKit'};
const foldPalette=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#6F4E7C','#4B5D16','#A23B43','#555555'];
const bandPalette=['#287D8E','#D89B27','#BC4565'];
const atlasChemicalIndex=new Map((dashboard.atlas?.chemical_ids||[]).map((id,i)=>[id,i]));
const responsibilityCache=new Map();
function currentMap(){return dashboard.atlas?.views.find(v=>v.representation===$('representation').value&&v.method===$('map-technique').value)}
function setTaskMode(mode, redraw=true){
  if(!['regression','classification','merged'].includes(mode))throw new Error('Unknown task workspace');
  taskMode=mode;selectedNode=null;selectedIds.clear();inspectorPage=0;
  document.querySelectorAll('.task-tabs button').forEach(button=>{button.setAttribute('aria-selected',String(button.dataset.task===mode));button.tabIndex=button.dataset.task===mode?0:-1});
  $('workspace').setAttribute('aria-labelledby','tab-'+mode);
  document.querySelectorAll('#endpoint-list input').forEach(input=>input.parentElement.hidden=mode!=='merged'&&endpointById.get(input.value).task!==mode);
  if(!selectedEndpoints().length)document.querySelectorAll('#endpoint-list input').forEach(input=>{if(mode==='merged'||endpointById.get(input.value).task===mode)input.checked=true});
  if(redraw){$('colour').value=mode==='classification'?'prediction':'residual';['preprocessing','strategy'].forEach(id=>$(id).value='__all__');['fold','repeat'].forEach(id=>$(id).value='all');$('search').value='';refreshEndpointDependent()}
}
function refreshMapTechniques(){
  const previous=$('map-technique').value,views=dashboard.atlas?.views.filter(v=>v.representation===$('representation').value)||[];
  $('map-technique').replaceChildren();
  views.forEach(v=>$('map-technique').add(new Option(v.method==='tsne'?'t-SNE':v.method.toUpperCase(),v.method)));
  if(!views.length)$('map-technique').add(new Option(dashboard.embedding.includes('PCA')?'PCA':'t-SNE','saved'));
  if([...$('map-technique').options].some(o=>o.value===previous))$('map-technique').value=previous;
}
function applyMapCoordinates(){
  const view=currentMap();
  if(view){dashboard.records.forEach(r=>{const coordinate=view.coordinates[atlasChemicalIndex.get(r.chemical_id)];[r.x,r.y]=coordinate});dashboard.embedding_axes=view.axes}
  $('map-note').textContent=view?`${representationLabels[view.representation]} · ${view.note} Continuous preprocessing: ${view.preprocessing.pca_dimensions} PCA components, ${(100*view.preprocessing.pca_variance_retained).toFixed(1)}% reference variance retained. ${view.method==='gtm'?`Reference: ${dashboard.atlas.reference_scope}. ${view.fit.iterations} EM updates; ${view.fit.converged?'converged':'stopped: '+view.fit.stop_reason}.`:''}`:'Saved map. Regenerate with --map-methods tsne gtm pca --representations morgan rdkit_descriptors morgan_rdkit to enable all alternatives.';
}
function thresholdSettings(){
  const low=Number($('threshold-low').value),high=Number($('threshold-high').value),kind=$('threshold-kind').value;
  const valid=$('threshold-low').value!==''&&$('threshold-high').value!==''&&Number.isFinite(low)&&Number.isFinite(high)&&low<high&&(kind!=='absolute_error'||low>=0);
  return {low,high,kind,valid};
}
function bandIndex(value,{low,high}){return value<low?0:value<high?1:2}
function bandLabels(settings){return [`Low < ${fmt(settings.low)}`,`Medium [${fmt(settings.low)}, ${fmt(settings.high)})`,`High ≥ ${fmt(settings.high)}`]}
function drawSpace(){
  applyMapCoordinates();
  const colourOptions=[...$('colour').options],available=evaluationRecords();
  colourOptions.forEach(option=>{option.disabled=option.value==='prediction'?taskMode!=='classification':option.value==='uncertainty'?!available.some(r=>Number.isFinite(r.predictions[$('model').value]?.uncertainty)):false});
  if($('colour').selectedOptions[0]?.disabled)$('colour').value=taskMode==='classification'?'prediction':'residual';
  drawSpaceBase();
  const graph=$('space'),model=$('model').value,rowsById=new Map(dashboard.records.map(r=>[r.id,r]));
  const byFold=$('fold-display').value==='colour',settings=thresholdSettings();
  const bands=$('threshold-colour').value==='on'&&diagnosticEndpoint()?.task==='regression'&&settings.valid;
  $('map-legend').replaceChildren();
  if(byFold||bands){
    const keys=[...new Set(evaluationRecords().map(r=>contextKey(r)))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
    renderLegend('map-legend',byFold?keys.map(key=>contextLabel(key,keys)):bandLabels(settings),byFold?foldPalette:bandPalette,byFold?keys:null);
    const traces=graph.data.map(trace=>{
      if(!trace.uid?.startsWith('eval-'))return trace;
      const rows=trace.customdata.map(id=>rowsById.get(id));
      if(bands&&!byFold&&rows.some(r=>r.endpoint!==diagnosticEndpoint().id))return trace;
      return {...trace,marker:{...trace.marker,showscale:false,color:rows.map(r=>byFold?foldPalette[keys.indexOf(contextKey(r))%foldPalette.length]:bandPalette[bandIndex(r.predictions[model][settings.kind],settings)])},text:rows.map(r=>`${escapeHtml(r.molecule_id)}<br>repeat ${r.repeat} · fold ${r.fold}${bands?'<br>'+bandLabels(settings)[bandIndex(r.predictions[model][settings.kind],settings)]:''}`)};
    });
    Plotly.react(graph,traces,graph.layout,{responsive:true,displaylogo:false});
  }
}
function drawDiagnostics(){
  drawDiagnosticsBase();
  const endpoint=diagnosticEndpoint(),rows=activeDiagnosticRows(),model=$('model').value,classification=endpoint?.task==='classification';
  $('diagnostic-e').hidden=!classification;$('diagnostic-f').hidden=!classification;
  $('diagnostic-title').textContent=`${endpoint?.label||'Endpoint'} · ${model||'select a model'} · ${classification?'classification':'regression'} diagnostics`;
  if(!classification)return;
  if(!rows.length){['diagnostic-e','diagnostic-f'].forEach(id=>emptyPlot(id,'No observations for this endpoint and model.'));return}
  // A supplied per-observation threshold takes priority over the endpoint default.
  const matrix=[[0,0],[0,0]];
  rows.forEach(r=>{const prediction=r.predictions[model],threshold=decisionThreshold(r,model);matrix[r.target_model][prediction.prediction>=threshold?1:0]++});
  plot('diagnostic-e',[{type:'heatmap',z:matrix,x:['Predicted 0','Predicted 1'],y:['Observed 0','Observed 1'],colorscale:'Blues',text:matrix.map(row=>row.map(String)),texttemplate:'%{text}',hovertemplate:'%{y}<br>%{x}<br>%{z} observations<extra></extra>'}],{title:'Confusion matrix · saved threshold policy',margin:{l:90,r:25,t:50,b:60}});
  const scores=rows.map(r=>r.predictions[model].prediction),labels=rows.map(r=>r.target_model);
  const brier=mean(scores.map((v,i)=>(v-labels[i])**2)),logloss=-mean(scores.map((v,i)=>labels[i]*Math.log(Math.max(1e-15,v))+(1-labels[i])*Math.log(Math.max(1e-15,1-v))));
  plot('diagnostic-f',[0,1].map(label=>({type:'violin',name:`Observed ${label}`,y:rows.filter(r=>r.target_model===label).map(r=>r.predictions[model].prediction),points:'all',box:{visible:true},meanline:{visible:false},marker:{color:label?'#BC4565':'#287D8E'}})),{title:`Probability distributions · Brier ${fmt(brier)} · log loss ${fmt(logloss)}`,margin:{l:50,r:20,t:70,b:50},yaxis:{title:'Predicted probability',range:[0,1]}});
}
function emptyPlot(id,message){plot(id,[],{annotations:[{text:message,showarrow:false,xref:'paper',yref:'paper',x:.5,y:.5}],xaxis:{visible:false},yaxis:{visible:false},margin:{l:20,r:20,t:30,b:20}})}
function contextLabel(key,keys){const suffix=key.split(' / ').at(-1);return keys.filter(k=>k.split(' / ').at(-1)===suffix).length===1?suffix:`Context ${keys.indexOf(key)+1} · ${suffix}`}
function contextAxis(keys){return {tickmode:'array',tickvals:keys,ticktext:keys.map(key=>contextLabel(key,keys)),title:{text:'Repeat / fold · full context on hover'}}}
function renderLegend(id,labels,palette,titles=null){$(id).replaceChildren();labels.forEach((label,i)=>{const span=document.createElement('span');span.title=titles?.[i]||label;const dot=document.createElement('i');dot.style.background=palette[i%palette.length];span.append(dot,document.createTextNode(label));$(id).append(span)})}
function contextKey(r){return `${r.preprocessing} / ${r.split_strategy} / ${r.split} / r${r.repeat} f${r.fold}`}
function drawThresholds(){
  const endpoint=diagnosticEndpoint(),rows=activeDiagnosticRows(),model=$('model').value;
  $('threshold-panel').hidden=endpoint?.task!=='regression';
  if(endpoint?.task!=='regression')return;
  const settings=thresholdSettings();
  if(!settings.valid){$('threshold-summary').textContent='Enter finite ordered cutoffs (lower < upper); absolute-error cutoffs must be nonnegative.';['threshold-distribution','threshold-folds'].forEach(id=>emptyPlot(id,'Apply valid cutoffs to view error bands.'));return}
  const counts=[0,0,0];rows.forEach(r=>counts[bandIndex(r.predictions[model][settings.kind],settings)]++);
  const labels=bandLabels(settings);
  $('threshold-summary').textContent=`${endpoint.label} · ${endpoint.target_transform} evaluation scale · ${rows.length} observations / ${new Set(rows.map(r=>r.chemical_id)).size} molecules. ${labels.map((label,i)=>`${label}: ${counts[i]}`).join(' · ')}. Bands are exploratory; equality enters the higher band.`;
  if(!rows.length){['threshold-distribution','threshold-folds'].forEach(id=>emptyPlot(id,'No selected regression observations.'));return}
  plot('threshold-distribution',counts.map((_,band)=>({type:'histogram',x:rows.filter(r=>bandIndex(r.predictions[model][settings.kind],settings)===band).map(r=>r.predictions[model][settings.kind]),name:labels[band],marker:{color:bandPalette[band]},opacity:.8})),{title:'Selected error distribution with cutoffs',barmode:'stack',xaxis:{title:settings.kind==='residual'?'Prediction − observed':'Absolute error'},yaxis:{title:'Observations'},margin:{l:55,r:20,t:55,b:80},legend:{orientation:'h'},shapes:[settings.low,settings.high].map(x=>({type:'line',x0:x,x1:x,y0:0,y1:1,yref:'paper',line:{color:'#15231f',dash:'dot'}}))});
  const groups=[...new Set(rows.map(contextKey))];
  plot('threshold-folds',counts.map((_,band)=>({type:'scatter',mode:'lines+markers',name:labels[band],x:groups,y:groups.map(key=>{const group=rows.filter(r=>contextKey(r)===key);return group.filter(r=>bandIndex(r.predictions[model][settings.kind],settings)===band).length/group.length}),line:{color:bandPalette[band]}})),{title:'Fraction in each error band by fold',xaxis:contextAxis(groups),yaxis:{title:'Fraction of observations',range:[0,1]},legend:{orientation:'h'},margin:{l:55,r:20,t:55,b:110}});
}
function sortedContexts(rows){return [...new Set(rows.map(contextKey))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}))}
function drawFoldEvidence(){
  const rows=activeDiagnosticRows(),model=$('model').value,endpoint=diagnosticEndpoint(),keys=sortedContexts(rows);
  $('fold-evidence-note').textContent=`${keys.length} repeat/fold contexts; ${rows.length} observations and ${new Set(rows.map(r=>r.chemical_id)).size} unique molecules. Coordinates are shared across folds. Each metric point retains its saved context; overlapping training sets do not create independent experiments.`;
  $('fold-panel').classList.toggle('fold-facets',$('fold-display').value==='facets');
  const metrics=metricRows(endpoint,$('metric').value).filter(r=>r.model===model);
  const metricKeys=metrics.map(r=>`${r.preprocessing} / ${r.split_strategy} / ${r.partition} / r${r.repeat} f${r.fold}`);
  plot('fold-results',[{type:'scatter',mode:'markers',x:metrics.map(r=>`${r.preprocessing} / ${r.split_strategy} / ${r.partition} / r${r.repeat} f${r.fold}`),y:metrics.map(r=>r.value),marker:{color:'#16617a',size:10},error_y:{type:'data',array:metrics.map(r=>Number.isFinite(r.ci_upper)?Math.max(0,r.ci_upper-r.value):0),arrayminus:metrics.map(r=>Number.isFinite(r.ci_lower)?Math.max(0,r.value-r.ci_lower):0),symmetric:false},customdata:metrics.map(r=>r.n_molecules),hovertemplate:'%{x}<br>metric %{y:.4f}<br>%{customdata} molecules<extra></extra>'}],{title:`Full-fold ${$('metric').value||'metrics'} · ${model}`,xaxis:contextAxis(metricKeys),yaxis:{title:$('metric').value||'Metric'},margin:{l:60,r:25,t:55,b:140}});
  if($('fold-display').value==='facets')drawFoldMaps(keys,rows,model);
  else plot('fold-errors',keys.map((key,i)=>({type:'violin',name:contextLabel(key,keys),meta:key,hovertemplate:'%{meta}<br>Error %{y:.4f}<extra></extra>',y:rows.filter(r=>contextKey(r)===key).map(r=>r.predictions[model].absolute_error),points:'all',box:{visible:true},meanline:{visible:false},marker:{color:foldPalette[i%10],size:4},line:{color:foldPalette[i%10]}})),{title:'Filtered molecule errors in every fold',showlegend:false,xaxis:{tickangle:-35},yaxis:{title:'Absolute error'},margin:{l:60,r:20,t:55,b:140}});
  requestAnimationFrame(()=>['fold-results','fold-errors'].forEach(id=>Plotly.Plots.resize($(id))));
}
function drawFoldMaps(keys,rows,model){
  const count=Math.min(10,keys.length),columns=Math.min(5,Math.max(1,count)),nrows=Math.ceil(count/columns),layout={title:'Shared chemical map · up to 10 fold panels',showlegend:false,annotations:[],margin:{l:35,r:20,t:75,b:40}},traces=[];
  if(!count){emptyPlot('fold-errors','No folds in this cohort.');return}
  const allX=dashboard.records.map(r=>r.x),allY=dashboard.records.map(r=>r.y),xr=[Math.min(...allX),Math.max(...allX)],yr=[Math.min(...allY),Math.max(...allY)];
  keys.slice(0,10).forEach((key,i)=>{const col=i%columns,row=Math.floor(i/columns),axis=i?String(i+1):'',xDomain=[col/columns+.02,(col+1)/columns-.02],yDomain=[1-(row+1)/nrows+.07,1-row/nrows-.06],group=rows.filter(r=>contextKey(r)===key);layout['xaxis'+axis]={domain:xDomain,anchor:'y'+axis,range:xr,showticklabels:false,zeroline:false};layout['yaxis'+axis]={domain:yDomain,anchor:'x'+axis,range:yr,showticklabels:false,zeroline:false};layout.annotations.push({xref:'paper',yref:'paper',x:(xDomain[0]+xDomain[1])/2,y:yDomain[1]+.025,text:escapeHtml(contextLabel(key,keys)),showarrow:false});traces.push({type:'scatter',mode:'markers',xaxis:'x'+axis,yaxis:'y'+axis,x:group.map(r=>r.x),y:group.map(r=>r.y),customdata:group.map(r=>r.id),marker:{size:5,color:foldPalette[i%10]},hovertemplate:`${escapeHtml(key)}<br>%{customdata}<extra></extra>`})});
  plot('fold-errors',traces,layout);
}
function drawUncertainty(rows){
  $('uncertainty-panel').hidden=false;$('uncertainty-link').hidden=false;
  const model=$('model').value,endpoint=diagnosticEndpoint(),scored=rows.filter(r=>Number.isFinite(r.predictions[model]?.uncertainty));
  const intervals=endpoint.task==='regression'?rows.filter(r=>Number.isFinite(r.predictions[model]?.prediction_lower)&&Number.isFinite(r.predictions[model]?.prediction_upper)):[];
  $('uq-status').textContent=`${scored.length}/${rows.length} selected observations have submitted uncertainty scores; ${intervals.length} have regression prediction intervals. ${!scored.length&&!intervals.length?'This run has no UQ outputs. Nothing is inferred from map entropy or fold count.':'Each curve retains its repeat/fold context. GTM entropy describes map projection, not prediction confidence.'}`;
  const groups=sortedContexts(scored);
  if(scored.length){
    plot('uncertainty-a',groups.map((key,i)=>{const group=scored.filter(r=>contextKey(r)===key);return {type:'scattergl',mode:'markers',name:contextLabel(key,groups),meta:key,x:group.map(r=>r.predictions[model].uncertainty),y:group.map(r=>r.predictions[model].absolute_error),customdata:group.map(r=>r.id),marker:{color:foldPalette[i%10],size:6,opacity:.7},hovertemplate:'Uncertainty %{x:.4f}<br>Error %{y:.4f}<extra>%{meta}</extra>'}}),{title:'Uncertainty versus realized error',xaxis:{title:'Submitted score · larger = less certain'},yaxis:{title:'Absolute error'},showlegend:groups.length<=10,legend:{orientation:'h',y:-.22},margin:{l:60,r:20,t:55,b:70}});
    plot('uncertainty-b',groups.map((key,i)=>{const group=scored.filter(r=>contextKey(r)===key).sort((a,b)=>a.predictions[model].uncertainty-b.predictions[model].uncertainty),x=[],y=[];let sum=0;group.forEach((r,j)=>{sum+=r.predictions[model].absolute_error;if(j===group.length-1||r.predictions[model].uncertainty!==group[j+1].predictions[model].uncertainty){x.push((j+1)/group.length);y.push(sum/(j+1))}});return{type:'scatter',mode:'lines',name:contextLabel(key,groups),meta:key,x,y,line:{color:foldPalette[i%10]},hovertemplate:'Fraction retained %{x:.2f}<br>MAE %{y:.4f}<extra>%{meta}</extra>'}}),{title:'Error-retention curve · each fold',xaxis:{title:'Fraction retained · most certain first',range:[0,1]},yaxis:{title:'Mean retained absolute error'},showlegend:false,margin:{l:60,r:20,t:55,b:70}});
  }else ['uncertainty-a','uncertainty-b'].forEach(id=>emptyPlot(id,'No submitted uncertainty scores in this cohort.'));
  if(intervals.length){
    const keys=sortedContexts(intervals),confidence=endpoint.interval_confidence;
    const traces=[{type:'scatter',mode:'markers',name:'Observed coverage',x:keys,y:keys.map(key=>{const group=intervals.filter(r=>contextKey(r)===key);return mean(group.map(r=>Number(r.target_model>=intervalBounds(r.predictions[model])[0]&&r.target_model<=intervalBounds(r.predictions[model])[1])))}),marker:{color:'#16617a',size:10}}];
    if(Number.isFinite(confidence))traces.push({type:'scatter',mode:'lines',name:'Declared nominal coverage',x:keys,y:keys.map(()=>confidence),line:{dash:'dash',color:'#BC4565'}});
    plot('uncertainty-c',traces,{title:'Empirical interval coverage by fold',yaxis:{title:'Observed fraction covered',range:[-.025,1.025]},xaxis:contextAxis(keys),legend:{orientation:'h',y:-.25},margin:{l:60,r:20,t:55,b:140}});
    plot('uncertainty-d',keys.map((key,i)=>({type:'violin',name:contextLabel(key,keys),meta:key,hovertemplate:'%{meta}<br>Width %{y:.4f}<extra></extra>',y:intervals.filter(r=>contextKey(r)===key).map(r=>intervalBounds(r.predictions[model])[1]-intervalBounds(r.predictions[model])[0]),points:'all',box:{visible:true},meanline:{visible:false},line:{color:foldPalette[i%10]}})),{title:'Prediction-interval widths by fold',yaxis:{title:'Width on evaluation scale'},xaxis:{title:'Repeat / fold · full context on hover'},showlegend:false,margin:{l:60,r:20,t:55,b:140}});
  }else ['uncertainty-c','uncertainty-d'].forEach(id=>emptyPlot(id,endpoint.task==='classification'?'Probability calibration is shown in classification diagnostics. No prediction sets were supplied.':'No regression prediction intervals were supplied.'));
}
function renderInspector(){
  const query=$('inspector-search').value.trim().toLowerCase(),scope=$('inspector-scope').value;
  const source=scope==='visible'?selectedEvaluation():dashboard.records.filter(r=>taskMode==='merged'||r.task===taskMode);
  const byChemical=new Map();source.forEach(r=>{const text=`${r.molecule_id} ${r.sample_id} ${r.original_sample_id||''} ${r.smiles} ${r.canonical_smiles}`.toLowerCase();if(!query||text.includes(query)){if(!byChemical.has(r.chemical_id))byChemical.set(r.chemical_id,[]);byChemical.get(r.chemical_id).push(r)}});
  const groups=[...byChemical.values()],pages=Math.max(1,Math.ceil(groups.length/8));inspectorPage=Math.min(inspectorPage,pages-1);
  $('inspector-results').replaceChildren();
  groups.slice(inspectorPage*8,inspectorPage*8+8).forEach(group=>{const r=group.find(row=>row.endpoint===diagnosticEndpoint()?.id)||group[0],button=document.createElement('button');button.type='button';button.className='inspector-result';button.innerHTML=`<strong>${escapeHtml(r.molecule_id)}</strong><span>${escapeHtml(r.canonical_smiles)}</span><small>${group.length} matching observations · ${escapeHtml(r.endpoint_label)}</small>`;button.addEventListener('click',()=>{showMolecule(r.id);$('details').insertAdjacentHTML('beforeend',`<br>${group.length} matching observation(s); this record: repeat ${r.repeat}, fold ${r.fold}`)});$('inspector-results').append(button)});
  $('inspector-count').textContent=`${groups.length} matches · ${inspectorPage+1}/${pages}`;$('inspector-prev').disabled=inspectorPage===0;$('inspector-next').disabled=inspectorPage>=pages-1;
  if(!groups.length)$('inspector-results').textContent='No matching molecules. Change the query or search scope.';
}
function gtmResponsibilities(view){
  if(!responsibilityCache.has(view.id)){const bytes=Uint8Array.from(atob(view.responsibilities_b64),c=>c.charCodeAt(0));const data=new DataView(bytes.buffer),values=new Float32Array(bytes.length/4);for(let i=0;i<values.length;i++)values[i]=data.getFloat32(i*4,true);responsibilityCache.set(view.id,values)}
  return responsibilityCache.get(view.id);
}
function moleculeSummaries(rows,model){
  const groups=new Map();rows.forEach(r=>{if(!groups.has(r.chemical_id))groups.set(r.chemical_id,[]);groups.get(r.chemical_id).push(r)});
  return [...groups].map(([id,group])=>({id,group,error:median(group.map(r=>r.predictions[model].absolute_error)),residual:median(group.map(r=>r.predictions[model].residual)),uncertainty:median(group.map(r=>r.predictions[model].uncertainty).filter(Number.isFinite))}));
}
function selectGTMNode(node){
  const view=currentMap();if(!view)return;
  const model=$('model').value,rows=endpointRecords(diagnosticEndpoint().id).filter(r=>r.predictions[model]);
  const chosen=rows.filter(r=>view.mode[atlasChemicalIndex.get(r.chemical_id)]===node);
  if(!chosen.length){$('gtm-note').textContent='No molecules have this node as their most likely assignment.';return}
  selectedNode=node;selectedIds=new Set(chosen.map(r=>r.id));drawSpace();
}
function drawGTM(){
  const view=currentMap();$('gtm-panel').hidden=view?.method!=='gtm';if(view?.method!=='gtm')return;
  const model=$('model').value,endpoint=diagnosticEndpoint(),rows=endpointRecords(endpoint.id).filter(r=>r.predictions[model]),molecules=moleculeSummaries(rows,model),kind=$('gtm-landscape').value,settings=thresholdSettings(),size=view.fit.config.nodes,K=view.grid.length,R=gtmResponsibilities(view),population=new Float64Array(K),weighted=new Float64Array(K),weight=new Float64Array(K);
  molecules.forEach(m=>{const start=atlasChemicalIndex.get(m.id)*K,value=kind==='uncertainty'?m.uncertainty:m.error;for(let k=0;k<K;k++){const r=R[start+k];population[k]+=r;if(Number.isFinite(value)){weighted[k]+=r*value;weight[k]+=r}}});
  const minimum=Math.max(0,Number($('gtm-support').value)||0);
  $('gtm-note').textContent=`${molecules.length} unique molecules for ${endpoint.label}. ${selectedNode===null?'No node selected.':`Node ${selectedNode} selected; clear it to restore the full cohort.`} Landscapes retain the full pre-node cohort. Soft support is summed probability mass; it is not an integer molecule count.`;
  const pies=kind==='bands'&&endpoint.task==='regression'&&settings.valid;
  $('gtm-pies').hidden=!pies;$('gtm-landscape-chart').hidden=pies;
  $('gtm-band-legend').hidden=!pies;if(pies)renderLegend('gtm-band-legend',bandLabels(settings),bandPalette);
  if(pies){drawGTMPies(view,molecules,settings);return}
  if(kind==='bands'){$('gtm-note').textContent+=' Regression error bands require a regression endpoint and valid cutoffs.'}
  const z=Array.from({length:size},(_,row)=>Array.from({length:size},(_,col)=>{const k=row*size+col;return population[k]<minimum?null:kind==='density'?population[k]:weight[k]>0?weighted[k]/weight[k]:null}));
  const axis=Array.from({length:size},(_,i)=>i);
  plot('gtm-landscape-chart',[{type:'heatmap',x:axis,y:axis,z,colorscale:kind==='density'?'Blues':'Viridis',customdata:Array.from({length:size},(_,row)=>Array.from({length:size},(_,col)=>row*size+col)),hovertemplate:'Node (%{x}, %{y})<br>Value %{z:.4f}<extra></extra>',colorbar:{title:{text:kind==='density'?'Soft support':kind==='uncertainty'?'Submitted UQ':'Mean molecule error'}}}],{title:kind==='density'?'Responsibility-weighted molecule density':kind==='uncertainty'?'Responsibility-weighted uncertainty':'Responsibility-weighted error',xaxis:{title:'GTM column'},yaxis:{title:'GTM row'},margin:{l:60,r:40,t:55,b:50}});
  const node=$('gtm-landscape-chart');node.removeAllListeners?.('plotly_click');node.on('plotly_click',event=>{const index=event.points?.[0]?.customdata;if(Number.isInteger(index))selectGTMNode(index)});
}
function drawGTMPies(view,molecules,settings){
  const groups=Array.from({length:view.grid.length},()=>[]);molecules.forEach(m=>groups[view.mode[atlasChemicalIndex.get(m.id)]].push(m));
  const maximum=Math.max(1,...groups.map(g=>g.length)),log=$('gtm-count-scale').value==='log',container=$('gtm-pies');container.style.gridTemplateColumns=`repeat(${view.fit.config.nodes}, minmax(28px,1fr))`;container.replaceChildren();
  for(let row=view.fit.config.nodes-1;row>=0;row--)for(let col=0;col<view.fit.config.nodes;col++){
    const node=row*view.fit.config.nodes+col,group=groups[node],counts=[0,0,0];group.forEach(m=>counts[bandIndex(settings.kind==='residual'?m.residual:m.error,settings)]++);
    const button=document.createElement('button');button.type='button';button.className='gtm-node';button.disabled=!group.length;button.setAttribute('aria-label',`Node ${node}: ${group.length} molecules; low ${counts[0]}, medium ${counts[1]}, high ${counts[2]}`);button.title=button.getAttribute('aria-label');button.setAttribute('aria-pressed',String(selectedNode===node));
    const total=group.length,first=total?counts[0]/total*100:0,second=total?(counts[0]+counts[1])/total*100:0,scale=total?(log?Math.log1p(total)/Math.log1p(maximum):Math.sqrt(total/maximum)):0;
    button.innerHTML=`<span style="width:${15+scale*70}%;height:${15+scale*70}%;background:conic-gradient(${bandPalette[0]} 0 ${first}%,${bandPalette[1]} ${first}% ${second}%,${bandPalette[2]} ${second}% 100%)"></span>`;
    button.addEventListener('click',()=>selectGTMNode(node));container.append(button);
  }
}
function initialiseWorkspace(){
  document.querySelectorAll('.task-tabs button').forEach(button=>{button.disabled=button.dataset.task!=='merged'&&!dashboard.endpoints.some(e=>e.task===button.dataset.task);button.addEventListener('click',()=>setTaskMode(button.dataset.task));button.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();const buttons=[...document.querySelectorAll('.task-tabs button:not(:disabled)')],i=buttons.indexOf(button),next=buttons[(i+(event.key==='ArrowRight'?1:buttons.length-1))%buttons.length];next.focus();setTaskMode(next.dataset.task)})});
  const reps=[...new Set(dashboard.atlas?.views.map(v=>v.representation)||[])];
  $('representation').replaceChildren();(reps.length?reps:['saved']).forEach(value=>$('representation').add(new Option(representationLabels[value]||'Saved features',value)));refreshMapTechniques();
  const defaultView=dashboard.atlas?.views.find(v=>v.method==='tsne');if(defaultView){$('representation').value=defaultView.representation;refreshMapTechniques();$('map-technique').value=defaultView.method}
  $('representation').addEventListener('change',()=>{refreshMapTechniques();selectedNode=null;selectedIds.clear();drawSpace()});$('map-technique').addEventListener('change',()=>{selectedNode=null;selectedIds.clear();drawSpace()});
  $('fold-display').addEventListener('change',drawSpace);
  $('apply-thresholds').addEventListener('click',()=>{drawSpace();$('view-status').textContent='Exploratory error bands updated; saved fold metrics are unchanged.'});
  ['threshold-colour','threshold-kind'].forEach(id=>$(id).addEventListener('change',drawSpace));
  ['gtm-landscape','gtm-support','gtm-count-scale'].forEach(id=>$(id).addEventListener('change',drawGTM));$('clear-node').addEventListener('click',()=>{selectedNode=null;selectedIds.clear();drawSpace()});
  $('inspector-search').addEventListener('input',()=>{clearTimeout(inspectorTimer);inspectorPage=0;inspectorTimer=setTimeout(renderInspector,120)});$('inspector-scope').addEventListener('change',()=>{inspectorPage=0;renderInspector()});
  $('inspector-prev').addEventListener('click',()=>{inspectorPage--;renderInspector()});$('inspector-next').addEventListener('click',()=>{inspectorPage++;renderInspector()});
  if(taskMode==='classification')$('colour').value='prediction';
  setTaskMode(taskMode,false);drawSpace();
}
