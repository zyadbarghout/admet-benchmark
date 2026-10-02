// Diagnostics backed by registered OOF artifacts, not map-derived confidence.
function intervalBounds(p){return $('interval-method').value==='absolute'&&Number.isFinite(p.absolute_lower)?[p.absolute_lower,p.absolute_upper]:[p.prediction_lower,p.prediction_upper]}
function scientificMatches(record){const filter=$('scientific-cohort').value,p=record.predictions[$('model').value];return filter==='all'||filter==='novel'&&p?.novel_scaffold===1||filter==='cliff'&&p?.property_cliff===1||filter==='distant'&&Number.isFinite(p?.nearest_train_similarity)&&p.nearest_train_similarity<.5}
function drawScientific(){
  const endpoint=diagnosticEndpoint(),rows=activeDiagnosticRows(),model=$('model').value;
  $('scientific-panel').hidden=!endpoint.cv_evidence;
  $('interval-method-control').hidden=endpoint.task!=='regression'||!rows.some(r=>Number.isFinite(r.predictions[model].absolute_lower));
  if(!endpoint.cv_evidence)return;
  const available=rows.filter(r=>Number.isFinite(r.predictions[model].nearest_train_similarity));
  $('scientific-note').textContent=`${rows.length} OOF observations / ${new Set(rows.map(r=>r.chemical_id)).size} chemicals. Reference neighbours use each fold's fitting molecules only. ${endpoint.uq_description}. Scaffold-shift coverage is empirical; post-hoc filtered coverage is descriptive.`;
  const keys=sortedContexts(available);
  plot('applicability-error',keys.map((key,i)=>{const group=available.filter(r=>contextKey(r)===key);return {type:'scattergl',mode:'markers',name:contextLabel(key,keys),x:group.map(r=>r.predictions[model].nearest_train_similarity),y:group.map(r=>r.predictions[model].absolute_error),customdata:group.map(r=>r.id),marker:{size:6,color:foldPalette[i%10],opacity:.6},hovertemplate:'Closest training Tanimoto %{x:.3f}<br>Error %{y:.4f}<extra>%{fullData.name}</extra>'}}),{title:'Error versus original training similarity',xaxis:{title:'Morgan Tanimoto · closest fitting molecule',range:[0,1]},yaxis:{title:'Absolute prediction error'},showlegend:false,margin:{l:65,r:20,t:50,b:65}});
  const bins=[0,.25,.5,.75,1.000001],groups=bins.slice(0,-1).map((lower,i)=>available.filter(r=>r.predictions[model].nearest_train_similarity>=lower&&r.predictions[model].nearest_train_similarity<bins[i+1]));
  const x=groups.map((g,i)=>`${bins[i]}–${Math.min(1,bins[i+1])} (n=${g.length})`);
  const covered=r=>{const p=r.predictions[model];if(endpoint.task==='classification')return r.target_model===0?p.set_0:p.set_1;const [lo,hi]=intervalBounds(p);return Number(r.target_model>=lo&&r.target_model<=hi)};
  plot('applicability-coverage',[{type:'scatter',mode:'markers',x,y:groups.map(g=>g.length>=10?mean(g.map(covered)):null),marker:{size:10,color:'#16617a'},name:'Observed coverage'},{type:'scatter',mode:'lines',x,y:x.map(()=>endpoint.interval_confidence),name:'Nominal reference',line:{dash:'dash',color:'#BC4565'}}],{title:'Coverage by training similarity · minimum 10 observations',xaxis:{title:'Training similarity / selected observation count'},yaxis:{title:'Interval or set coverage',range:[-.025,1.025]},legend:{orientation:'h',y:-.3},margin:{l:65,r:20,t:60,b:100}});
  const cliff=available.filter(r=>r.predictions[model].property_cliff===1);
  $('cliff-note').textContent=endpoint.task==='regression'?`Property-cliff stress cohort: ${cliff.length} observations. Producer-supplied labels compare a similar fitting molecule with the measured target using the property-gap cutoff recorded in the study protocol. This label uses the evaluation outcome after prediction; it is never a predictor or model-selection signal.`:'Property-cliff annotations are supplied for regression only. Classification disagreement is assessed through probability, set and error diagnostics.';
  const sorted=[...cliff].sort((a,b)=>b.predictions[model].absolute_error-a.predictions[model].absolute_error).slice(0,12);
  $('cliff-table').innerHTML='<thead><tr><th>Molecule</th><th>Fold</th><th>Error</th><th>Largest similar-pair gap</th></tr></thead><tbody>'+sorted.map(r=>`<tr><td>${escapeHtml(r.molecule_id)}</td><td>${r.fold}</td><td>${fmt(r.predictions[model].absolute_error)}</td><td>${fmt(r.predictions[model].max_similar_property_gap)}</td></tr>`).join('')+'</tbody>';
  drawDecision(rows,endpoint,model);
  if(endpoint.task==='classification'&&rows.some(r=>Number.isFinite(r.predictions[model].set_0)))drawPredictionSets(rows,endpoint,model);
}
function drawDecision(rows,endpoint,model){
  $('decision-panel').hidden=endpoint.task!=='regression';if(endpoint.task!=='regression')return;
  const raw=$('property-cutoff').value,value=Number(raw);
  if(!raw||!Number.isFinite(value)){$('decision-note').textContent=`Enter a scientifically meaningful property cutoff on the ${endpoint.evaluation_units || endpoint.target_transform}. This is separate from residual/error cutoffs.`;emptyPlot('decision-chart','Set a property cutoff to inspect interval decisions.');return}
  const groups=['Below','Crosses cutoff','Above'];const data=groups.map(()=>[]);
  rows.forEach(r=>{const [lo,hi]=intervalBounds(r.predictions[model]);if(Number.isFinite(lo)&&Number.isFinite(hi))data[hi<value?0:lo>value?2:1].push(r)});
  $('decision-note').textContent=`Cutoff ${value} · ${endpoint.evaluation_units || endpoint.target_transform} · ${groups.map((g,i)=>`${g}: ${data[i].length}`).join(' · ')}. Equality counts as crossing. These intervals do not supply a probability of exceeding the cutoff.`;
  plot('decision-chart',groups.map((name,i)=>({type:'violin',name,y:data[i].map(r=>r.predictions[model].prediction),customdata:data[i].map(r=>r.id),points:'all',box:{visible:true},meanline:{visible:false},marker:{color:bandPalette[i],size:5}})),{title:'Predictions grouped by interval decision',yaxis:{title:'Predicted property · evaluation scale'},showlegend:false,shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:value,y1:value,line:{dash:'dash',color:'#15231f'}}],margin:{l:65,r:20,t:50,b:60}});
}
function drawPredictionSets(rows,endpoint,model){
  const available=rows.filter(r=>Number.isFinite(r.predictions[model].set_0)),keys=sortedContexts(available);
  const coverage=group=>mean(group.map(r=>r.target_model===0?r.predictions[model].set_0:r.predictions[model].set_1));
  plot('uncertainty-c',[{type:'scatter',mode:'markers',name:'Set coverage',x:keys,y:keys.map(k=>coverage(available.filter(r=>contextKey(r)===k))),marker:{color:'#16617a',size:10}},{type:'scatter',mode:'lines',name:'Nominal reference',x:keys,y:keys.map(()=>endpoint.interval_confidence),line:{dash:'dash',color:'#BC4565'}}],{title:'Class-conditional conformal set coverage',xaxis:contextAxis(keys),yaxis:{title:'True label included',range:[-.025,1.025]},legend:{orientation:'h',y:-.3},margin:{l:60,r:20,t:50,b:105}});
  const names=['Empty','{0}','{1}','{0,1}'],count=[0,0,0,0];available.forEach(r=>{const p=r.predictions[model];count[p.set_0+2*p.set_1]++});
  plot('uncertainty-d',[{type:'scatter',mode:'markers',x:names,y:count,marker:{size:14,color:'#16617a'},hovertemplate:'%{x}<br>%{y} observations<extra></extra>'}],{title:'Prediction-set outcomes · retained observations',xaxis:{title:'Set membership'},yaxis:{title:'Observation count'},margin:{l:65,r:20,t:50,b:65}});
  $('uq-status').textContent+=` ${available.length} observations have conformal prediction sets; ${count[1]+count[2]} are singletons and ${count[3]} include both labels.`;
  plot('uncertainty-b',keys.map((key,i)=>{const group=available.filter(r=>contextKey(r)===key).sort((a,b)=>a.predictions[model].uncertainty-b.predictions[model].uncertainty),x=[],y=[];let wrong=0;group.forEach((r,j)=>{wrong+=Number(Number(r.predictions[model].prediction>=decisionThreshold(r,model))!==r.target_model);if(j===group.length-1||r.predictions[model].uncertainty!==group[j+1].predictions[model].uncertainty){x.push((j+1)/group.length);y.push(wrong/(j+1))}});return {type:'scatter',mode:'lines',name:contextLabel(key,keys),x,y,line:{color:foldPalette[i%10]}}}),{title:'Classification risk versus fraction retained',xaxis:{title:'Fraction retained · lowest entropy first',range:[0,1]},yaxis:{title:'Misclassification fraction',range:[0,1]},showlegend:false,margin:{l:60,r:20,t:50,b:65}});
}
function initialiseScientific(){
  ['interval-method','scientific-cohort'].forEach(id=>$(id).addEventListener('change',()=>{selectedNode=null;selectedIds.clear();drawSpace()}));
  $('apply-property-cutoff').addEventListener('click',drawScientific);
  document.querySelectorAll('.task-tabs button').forEach(button=>button.addEventListener('click',()=>{$('scientific-cohort').value='all';if(diagnosticEndpoint()?.cv_evidence){['preprocessing','strategy'].forEach(id=>{const first=[...$(id).options].find(o=>o.value!=='__all__');if(first)$(id).value=first.value});refreshFoldControls();refreshMetrics()}drawSpace()}));
  if(dashboard.endpoints.some(e=>e.cv_evidence)){
    const setFirst=id=>{const options=[...$(id).options].filter(o=>o.value!=='__all__');if(options.length)$(id).value=options[0].value};
    setFirst('preprocessing');setFirst('strategy');refreshFoldControls();refreshMetrics();drawSpace();
  }
}
