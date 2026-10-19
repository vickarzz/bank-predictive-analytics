"""Build the static live demo for GitHub Pages: docs/index.html.

One self-contained HTML page (data embedded as compact JSON, Plotly from a CDN), so it is
served by GitHub Pages with no server, never sleeps, and works wherever github.io works.

    python tools/build_static_demo.py                      # uses streamlit_app/demo_data (20k sample)
    python tools/build_static_demo.py --gold data/gold     # or your own pipeline output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def codes(s: pd.Series):
    cats = sorted(s.astype(str).unique())
    m = {c: i for i, c in enumerate(cats)}
    return cats, [m[v] for v in s.astype(str)]


def build(gold: Path) -> dict:
    cust = pd.read_csv(gold / "dim_customer.csv")
    churn = pd.read_csv(gold / "fact_churn_score.csv").merge(cust[["customer_id", "segment", "region"]], on="customer_id")
    loans = pd.read_csv(gold / "fact_loan_risk.csv")
    daily = pd.read_csv(gold / "fact_remittance_daily.csv")
    fc = pd.read_csv(gold / "fact_remittance_forecast.csv")
    mm = pd.read_csv(gold / "model_metrics.csv")
    fi = pd.read_csv(gold / "feature_importance.csv")

    seg, seg_c = codes(churn.segment)
    reg, reg_c = codes(churn.region)
    band, band_c = codes(churn.risk_band)
    sig, sig_c = codes(churn.primary_signal)
    churn_cols = {
        "id": churn.customer_id.tolist(), "seg": seg_c, "reg": reg_c, "band": band_c, "sig": sig_c,
        "dec": churn.risk_decile.tolist(), "p": churn.churn_probability.round(5).tolist(),
        "bar": churn.balance_at_risk_sar.round(0).astype(int).tolist(),
    }
    prod, prod_c = codes(loans["product"])
    sect, sect_c = codes(loans.employer_sector)
    grade, grade_c = codes(loans.risk_grade)
    dbr, dbr_c = codes(loans.dbr_band)
    loan_cols = {
        "id": loans.loan_id.tolist(), "prod": prod_c, "sect": sect_c, "grade": grade_c, "dbr": dbr_c,
        "dbrv": loans.dbr.round(4).tolist(), "pd": loans.pd_12m.round(5).tolist(), "bureau": loans.bureau_score.tolist(),
        "ead": loans.ead_sar.round(0).astype(int).tolist(), "el": loans.expected_loss_sar.round(0).astype(int).tolist(),
    }
    corr = sorted(daily.corridor.unique())
    rem = {}
    for c in corr:
        d = daily[daily.corridor == c].sort_values("date")
        f = fc[fc.corridor == c].sort_values("date")
        rem[c] = {"a": d.amount_sar.round(0).astype(int).tolist(),
                  "bt": [None if pd.isna(v) else int(round(v)) for v in d.backtest_forecast_sar],
                  "f": f.forecast_amount_sar.round(0).astype(int).tolist(),
                  "lo": f.lower_p10_sar.round(0).astype(int).tolist(), "hi": f.upper_p90_sar.round(0).astype(int).tolist()}
    first = daily[daily.corridor == corr[0]].sort_values("date")
    return {
        "dict": {"seg": seg, "reg": reg, "band": band, "sig": sig, "prod": prod, "sect": sect, "grade": grade, "dbr": dbr},
        "churn": churn_cols, "loans": loan_cols,
        "rem": {"dates": first.date.tolist(), "fdates": fc[fc.corridor == corr[0]].sort_values("date").date.tolist(),
                "isbt": first.is_backtest_period.astype(bool).tolist(), "corr": rem},
        "metrics": mm[["model_name", "dataset", "metric", "value", "model_version"]].round(4).values.tolist(),
        "fi": fi[["model_name", "feature", "importance"]].round(4).values.tolist(),
        "n_customers": int(len(cust)),
    }


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bank Predictive Analytics — live demo</title>
<meta name="description" content="Churn, credit risk and remittance forecasting for a synthetic Saudi retail bank — interactive demo of the pipeline's gold output.">
<script src="assets/plotly-2.35.2.min.js"></script>  <!-- bundled (MIT) so the page has no external dependency -->
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--line:#e6e5df;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--high:#d03b3b;--med:#fab219;--low:#0ca30c}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,"Segoe UI",system-ui,-apple-system,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:28px 20px 40px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.01em} .lede{color:var(--ink2);margin:0 0 4px;max-width:880px}
.links{font-size:13px;color:var(--muted);margin-bottom:18px} .links a{color:var(--s1);text-decoration:none} .links a:hover{text-decoration:underline}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--grid);margin-bottom:18px;overflow-x:auto}
.tab{background:none;border:0;padding:10px 14px;font:inherit;color:var(--ink2);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
.tab[aria-selected=true]{color:var(--s1);border-color:var(--s1);font-weight:600}
.panel{display:none} .panel.on{display:block}
.filters{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px} .filters label{font-size:12px;color:var(--ink2);display:flex;flex-direction:column;gap:4px}
select{font:inherit;padding:6px 8px;border:1px solid var(--grid);border-radius:6px;background:#fff;min-width:170px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px} .kpi b{display:block;font-size:24px;font-weight:600}
.kpi span{font-size:12px;color:var(--ink2)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px} @media(max-width:820px){.grid2{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px;min-width:0}
.card h3{font-size:15px;margin:0 0 6px} .note{font-size:12px;color:var(--muted);margin:4px 0 0}
.chart{height:300px} .chart.tall{height:360px}
table{border-collapse:collapse;width:100%;font-size:12.5px} th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--ink2);font-weight:600} td.n{text-align:right;font-variant-numeric:tabular-nums} .tbl{overflow-x:auto;max-height:420px}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11.5px;font-weight:600}
.pill.High{background:#fbe3e3;color:#a12a2a}.pill.Medium{background:#fdf0d2;color:#8a5a00}.pill.Low{background:#dcf3dc;color:#0b6b0b}
footer{margin-top:24px;font-size:12px;color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <h1>Bank Predictive Analytics</h1>
  <p class="lede">Churn, credit risk and remittance forecasting for a synthetic Saudi retail bank. Models are trained and validated out of time by the pipeline; this page shows its gold-layer output for a __N__-customer sample.</p>
  <div class="links"><a href="https://github.com/vickarzz/bank-predictive-analytics">Source code on GitHub</a> · all data is synthetic</div>
  <div class="tabs" role="tablist">
    <button class="tab" role="tab" aria-selected="true" data-t="over">Overview</button>
    <button class="tab" role="tab" aria-selected="false" data-t="churn">Churn risk</button>
    <button class="tab" role="tab" aria-selected="false" data-t="credit">Credit risk</button>
    <button class="tab" role="tab" aria-selected="false" data-t="rem">Remittance forecast</button>
    <button class="tab" role="tab" aria-selected="false" data-t="model">Model health</button>
  </div>

  <section class="panel on" id="p-over">
    <div class="kpis" id="k-over"></div>
    <div class="grid2">
      <div class="card"><h3>Balance at risk by segment (SAR)</h3><div class="chart" id="c-seg"></div></div>
      <div class="card"><h3>Expected loss by risk grade (SAR)</h3><div class="chart" id="c-grade"></div></div>
    </div>
    <div class="card"><h3>Total daily remittance — last 180 days and next 90 days (SAR)</h3><div class="chart" id="c-over-rem"></div></div>
  </section>

  <section class="panel" id="p-churn">
    <div class="filters"><label>Segment<select id="f-seg"></select></label><label>Region<select id="f-reg"></select></label></div>
    <div class="kpis" id="k-churn"></div>
    <div class="grid2">
      <div class="card"><h3>High-risk customers by primary signal</h3><div class="chart" id="c-sig"></div></div>
      <div class="card"><h3>Average churn probability by risk decile</h3><div class="chart" id="c-dec"></div>
        <p class="note">Colour = risk band: red High (decile 10), amber Medium (8–9), green Low.</p></div>
    </div>
    <div class="card"><h3>Retention worklist — highest churn probability first (top 50)</h3><div class="tbl" id="t-churn"></div></div>
  </section>

  <section class="panel" id="p-credit">
    <div class="filters"><label>Product<select id="f-prod"></select></label><label>Employer sector<select id="f-sect"></select></label></div>
    <div class="kpis" id="k-credit"></div>
    <div class="grid2">
      <div class="card"><h3>EAD by risk grade (SAR)</h3><div class="chart" id="c-ead"></div></div>
      <div class="card"><h3>Average 12-month PD by DBR band</h3><div class="chart" id="c-dbr"></div></div>
    </div>
    <div class="card"><h3>Watchlist — largest expected loss first (top 50)</h3><div class="tbl" id="t-credit"></div></div>
  </section>

  <section class="panel" id="p-rem">
    <div class="filters"><label>Corridor<select id="f-corr"></select></label></div>
    <div class="kpis" id="k-rem"></div>
    <div class="card"><h3>Daily remittance: actual, backtest and forecast with P10–P90 band (SAR)</h3><div class="chart tall" id="c-rem"></div>
      <p class="note">Peaks: Saudi payday window (~27th), Saturdays and the days before Eid. Last 180 days of history shown.</p></div>
    <div class="card"><h3>90-day forecast by corridor (SAR)</h3><div class="chart" id="c-corr"></div></div>
  </section>

  <section class="panel" id="p-model">
    <div class="kpis" id="k-model"></div>
    <div class="grid2">
      <div class="card"><h3>Permutation importance <select id="f-model" style="min-width:140px;margin-left:8px"><option value="churn">Churn</option><option value="credit_risk">Credit risk</option></select></h3>
        <div class="chart tall" id="c-fi"></div><p class="note">Drop in AUC when the feature is shuffled (out-of-time sample).</p></div>
      <div class="card"><h3>All metrics</h3><div class="tbl" id="t-metrics"></div></div>
    </div>
  </section>

  <footer>Built with Python, scikit-learn, MLflow, Airflow / Databricks and Power BI · Ahmad Zulfikar Rafsanjani · static page generated by tools/build_static_demo.py</footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const C = getComputedStyle(document.documentElement), V = n => C.getPropertyValue(n).trim();
const S1=V('--s1'), S2=V('--s2'), INK2=V('--ink2'), MUTED=V('--muted'), GRID=V('--grid'), BAND={High:V('--high'),Medium:V('--med'),Low:V('--low')};
const fmtInt = v => Math.round(v).toLocaleString('en-US');
const sar = v => { const a=Math.abs(v); return 'SAR ' + (a>=1e9?(v/1e9).toFixed(2)+'bn':a>=1e6?(v/1e6).toFixed(1)+'M':a>=1e3?(v/1e3).toFixed(1)+'K':fmtInt(v)); };
const pct = (v,d=1) => (v*100).toFixed(d)+'%';
const sum = a => a.reduce((x,y)=>x+y,0), mean = a => a.length? sum(a)/a.length : 0;
const kpis = (id, items) => document.getElementById(id).innerHTML = items.map(([v,l])=>`<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join('');
const layout = (extra={}) => Object.assign({margin:{l:8,r:16,t:8,b:30},paper_bgcolor:'#fff',plot_bgcolor:'#fff',
  font:{family:'Inter, Segoe UI, sans-serif',size:12,color:INK2},showlegend:false,bargap:.35,
  xaxis:{showgrid:false,linecolor:GRID,tickfont:{color:MUTED},automargin:true},yaxis:{gridcolor:GRID,zeroline:false,tickfont:{color:MUTED},automargin:true},
  hoverlabel:{bgcolor:'#fff',font:{color:'#0b0b0b'}}}, extra);
const cfg = {displayModeBar:false,responsive:true};
function hbar(el, labels, values, fmt, colors){ Plotly.react(el,[{type:'bar',orientation:'h',y:labels,x:values,marker:{color:colors||S1,cornerradius:4},
  hovertemplate:'%{y}<br>%{x:'+fmt+'}<extra></extra>'}], layout({yaxis:{autorange:'reversed',automargin:true,tickfont:{color:INK2}},xaxis:{gridcolor:GRID,tickfont:{color:MUTED},automargin:true}}), cfg); }
function vbar(el, labels, values, fmt, colors, yfmt){ Plotly.react(el,[{type:'bar',x:labels,y:values,marker:{color:colors||S1,cornerradius:4},
  hovertemplate:'%{x}<br>%{y:'+fmt+'}<extra></extra>'}], layout({xaxis:{type:'category',showgrid:false,linecolor:GRID,tickfont:{color:INK2}},yaxis:{gridcolor:GRID,tickformat:yfmt||'',tickfont:{color:MUTED},automargin:true}}), cfg); }
function groupSum(keys, vals){ const m=new Map(); keys.forEach((k,i)=>m.set(k,(m.get(k)||0)+vals[i])); return m; }
function fillSelect(id, opts){ document.getElementById(id).innerHTML = '<option value="">All</option>' + opts.map((o,i)=>`<option value="${i}">${o}</option>`).join(''); }

// ---------------- tabs ----------------
const drawn = {};
document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab').forEach(x=>x.setAttribute('aria-selected', x===b));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on', p.id==='p-'+b.dataset.t));
  if(!drawn[b.dataset.t]){ ({over:drawOver,churn:drawChurn,credit:drawCredit,rem:drawRem,model:drawModel})[b.dataset.t](); drawn[b.dataset.t]=1; }
  window.dispatchEvent(new Event('resize'));
});

const ch = D.churn, ln = D.loans, dict = D.dict, N = ch.p.length, L = ln.pd.length;
const metric = (m,k) => (D.metrics.find(r=>r[0]===m && r[2]===k)||[0,0,0,NaN])[3];
const rowsWhere = (n, pred) => { const r=[]; for(let i=0;i<n;i++) if(pred(i)) r.push(i); return r; };
const totalRem = () => { const corr=Object.values(D.rem.corr); const n=D.rem.dates.length, nf=D.rem.fdates.length;
  const a=Array(n).fill(0), bt=Array(n).fill(null), f=Array(nf).fill(0), lo=Array(nf).fill(0), hi=Array(nf).fill(0);
  corr.forEach(c=>{ c.a.forEach((v,i)=>a[i]+=v); c.bt.forEach((v,i)=>{ if(v!==null) bt[i]=(bt[i]||0)+v; }); c.f.forEach((v,i)=>f[i]+=v); c.lo.forEach((v,i)=>lo[i]+=v); c.hi.forEach((v,i)=>hi[i]+=v); });
  return {a,bt,f,lo,hi}; };

// ---------------- overview ----------------
function drawOver(){
  const t = totalRem();
  kpis('k-over', [[fmtInt(ch.band.filter(b=>dict.band[b]==='High').length),'High-risk customers'],[sar(sum(ch.bar)),'Balance at risk'],
    [sar(sum(ln.ead)),'Credit exposure (EAD)'],[sar(sum(ln.el)),'Expected loss'],[sar(sum(t.f)),'Remittance next 90 days'],[metric('churn','auc').toFixed(2),'Churn model AUC (out-of-time)']]);
  const seg = [...groupSum(ch.seg.map(i=>dict.seg[i]), ch.bar)].sort((a,b)=>b[1]-a[1]);
  hbar('c-seg', seg.map(x=>x[0]), seg.map(x=>x[1]), ',.0f');
  const gr = groupSum(ln.grade.map(i=>dict.grade[i]), ln.el), gl = [...gr.keys()].sort();
  vbar('c-grade', gl, gl.map(g=>gr.get(g)), ',.0f');
  const n=D.rem.dates.length, s=Math.max(0,n-180);
  Plotly.react('c-over-rem',[{x:D.rem.dates.slice(s),y:t.a.slice(s),name:'Actual',line:{color:S1,width:2},hovertemplate:'%{y:,.0f}'},
    {x:D.rem.fdates,y:t.f,name:'Forecast',line:{color:S2,width:2},hovertemplate:'%{y:,.0f}'}],
    layout({showlegend:true,legend:{orientation:'h',y:1.12,x:0},hovermode:'x unified',yaxis:{gridcolor:GRID,tickformat:'~s',tickfont:{color:MUTED},automargin:true}}),cfg);
}

// ---------------- churn ----------------
fillSelect('f-seg', dict.seg); fillSelect('f-reg', dict.reg);
function drawChurn(){
  const fs=document.getElementById('f-seg').value, fr=document.getElementById('f-reg').value;
  const idx = rowsWhere(N, i => (fs===''||ch.seg[i]==+fs) && (fr===''||ch.reg[i]==+fr));
  const p = idx.map(i=>ch.p[i]), high = idx.filter(i=>dict.band[ch.band[i]]==='High');
  kpis('k-churn', [[fmtInt(idx.length),'Customers scored'],[fmtInt(high.length),'High-risk customers'],[fmtInt(sum(p)),'Expected churners (3M)'],
    [sar(sum(idx.map(i=>ch.bar[i]))),'Balance at risk'],[pct(mean(p),2),'Avg churn probability']]);
  const sg=[...groupSum(high.map(i=>dict.sig[ch.sig[i]]), high.map(()=>1))].sort((a,b)=>b[1]-a[1]);
  hbar('c-sig', sg.map(x=>x[0]), sg.map(x=>x[1]), ',.0f');
  const dec=[...Array(10).keys()].map(k=>{ const r=idx.filter(i=>ch.dec[i]===k+1); return [k+1, mean(r.map(i=>ch.p[i])), r.length?dict.band[ch.band[r[0]]]:'Low']; });
  vbar('c-dec', dec.map(d=>String(d[0])), dec.map(d=>d[1]), '.2%', dec.map(d=>BAND[d[2]]), '.0%');
  const top = idx.sort((a,b)=>ch.p[b]-ch.p[a]).slice(0,50);
  document.getElementById('t-churn').innerHTML = '<table><thead><tr><th>Customer ID</th><th>Segment</th><th>Region</th><th>Risk band</th><th>Primary signal</th><th class="n">Churn probability</th><th class="n">Balance at risk (SAR)</th></tr></thead><tbody>' +
    top.map(i=>`<tr><td>${ch.id[i]}</td><td>${dict.seg[ch.seg[i]]}</td><td>${dict.reg[ch.reg[i]]}</td><td><span class="pill ${dict.band[ch.band[i]]}">${dict.band[ch.band[i]]}</span></td><td>${dict.sig[ch.sig[i]]}</td><td class="n">${pct(ch.p[i])}</td><td class="n">${fmtInt(ch.bar[i])}</td></tr>`).join('') + '</tbody></table>';
}
['f-seg','f-reg'].forEach(id=>document.getElementById(id).onchange=drawChurn);

// ---------------- credit ----------------
fillSelect('f-prod', dict.prod); fillSelect('f-sect', dict.sect);
function drawCredit(){
  const fp=document.getElementById('f-prod').value, fs=document.getElementById('f-sect').value;
  const idx = rowsWhere(L, i => (fp===''||ln.prod[i]==+fp) && (fs===''||ln.sect[i]==+fs));
  const ead=sum(idx.map(i=>ln.ead[i])), el=sum(idx.map(i=>ln.el[i]));
  kpis('k-credit', [[fmtInt(idx.length),'Live loans'],[sar(ead),'EAD'],[sar(el),'Expected loss'],[pct(ead?el/ead:0,2),'EL rate'],[fmtInt(idx.filter(i=>ln.dbrv[i]>0.45).length),'Loans above 45% DBR']]);
  const g=groupSum(idx.map(i=>dict.grade[ln.grade[i]]), idx.map(i=>ln.ead[i])), gl=[...g.keys()].sort();
  vbar('c-ead', gl, gl.map(k=>g.get(k)), ',.0f');
  const order=['<=25%','25-33%','33-45%','>45%'], lab=['≤25%','25–33%','33–45%','>45%'];
  const pdv=order.map(b=>mean(idx.filter(i=>dict.dbr[ln.dbr[i]]===b).map(i=>ln.pd[i])));
  vbar('c-dbr', lab, pdv, '.2%', null, '.0%');
  const top=idx.sort((a,b)=>ln.el[b]-ln.el[a]).slice(0,50);
  document.getElementById('t-credit').innerHTML = '<table><thead><tr><th>Loan ID</th><th>Product</th><th>Grade</th><th>DBR band</th><th>Employer sector</th><th class="n">Bureau</th><th class="n">PD 12M</th><th class="n">EAD (SAR)</th><th class="n">Expected loss (SAR)</th></tr></thead><tbody>' +
    top.map(i=>`<tr><td>${ln.id[i]}</td><td>${dict.prod[ln.prod[i]]}</td><td>${dict.grade[ln.grade[i]]}</td><td>${dict.dbr[ln.dbr[i]]}</td><td>${dict.sect[ln.sect[i]]}</td><td class="n">${ln.bureau[i]}</td><td class="n">${pct(ln.pd[i])}</td><td class="n">${fmtInt(ln.ead[i])}</td><td class="n">${fmtInt(ln.el[i])}</td></tr>`).join('') + '</tbody></table>';
}
['f-prod','f-sect'].forEach(id=>document.getElementById(id).onchange=drawCredit);

// ---------------- remittance ----------------
const corrNames = Object.keys(D.rem.corr);
document.getElementById('f-corr').innerHTML = '<option value="">All corridors</option>' + corrNames.map(c=>`<option>${c}</option>`).join('');
function drawRem(){
  const c=document.getElementById('f-corr').value, t = c ? D.rem.corr[c] : totalRem();
  const n=D.rem.dates.length, s=Math.max(0,n-180);
  const btIdx=D.rem.isbt.map((b,i)=>b?i:-1).filter(i=>i>=0);
  const mapeOf = x => mean(btIdx.map(i=>Math.abs(x.bt[i]-x.a[i])/x.a[i]));
  // "All corridors" = average of the per-corridor errors, the same definition the pipeline reports
  const mape = c ? mapeOf(t) : mean(corrNames.map(k=>mapeOf(D.rem.corr[k])));
  kpis('k-rem', [[sar(sum(t.f)),'Forecast next 90 days'],[sar(sum(t.lo)),'Low case (P10)'],[sar(sum(t.hi)),'High case (P90)'],[pct(mape),'Backtest MAPE (90 days)']]);
  Plotly.react('c-rem',[
    {x:D.rem.fdates,y:t.hi,line:{width:0},hoverinfo:'skip',showlegend:false},
    {x:D.rem.fdates,y:t.lo,line:{width:0},fill:'tonexty',fillcolor:'rgba(235,104,52,.15)',name:'P10–P90 band',hovertemplate:'P10 %{y:,.0f}<extra></extra>'},
    {x:D.rem.dates.slice(s),y:t.a.slice(s),name:'Actual',line:{color:S1,width:2},hovertemplate:'Actual %{y:,.0f}<extra></extra>'},
    {x:D.rem.dates.slice(s),y:t.bt.slice(s),name:'Backtest',line:{color:INK2,width:2,dash:'dot'},hovertemplate:'Backtest %{y:,.0f}<extra></extra>'},
    {x:D.rem.fdates,y:t.f,name:'Forecast',line:{color:S2,width:2},hovertemplate:'Forecast %{y:,.0f}<extra></extra>'}],
    layout({showlegend:true,legend:{orientation:'h',y:1.1,x:0},hovermode:'x unified',yaxis:{gridcolor:GRID,tickformat:'~s',tickfont:{color:MUTED},automargin:true}}),cfg);
  const totals=corrNames.map(k=>[k,sum(D.rem.corr[k].f)]).sort((a,b)=>b[1]-a[1]);
  const pal=[S1,S2,V('--s3'),V('--s4'),'#e87ba4','#008300'];
  hbar('c-corr', totals.map(x=>x[0]), totals.map(x=>x[1]), ',.0f', totals.map(x=>pal[corrNames.indexOf(x[0])]));
}
document.getElementById('f-corr').onchange=drawRem;

// ---------------- model health ----------------
function drawModel(){
  kpis('k-model', [[metric('churn','auc').toFixed(3),'Churn AUC (out-of-time)'],[metric('credit_risk','auc').toFixed(3),'Credit AUC (out-of-time)'],
    [metric('credit_risk','gini').toFixed(3),'Credit Gini'],[pct(metric('remittance_forecast','mape'),2),'Remittance MAPE (backtest)']]);
  const m=document.getElementById('f-model').value;
  const r=D.fi.filter(x=>x[0]===m).sort((a,b)=>b[2]-a[2]).slice(0,10);
  hbar('c-fi', r.map(x=>x[1]), r.map(x=>x[2]), '.4f');
  document.getElementById('t-metrics').innerHTML = '<table><thead><tr><th>Model</th><th>Dataset</th><th>Metric</th><th class="n">Value</th><th>Version</th></tr></thead><tbody>' +
    D.metrics.map(x=>`<tr><td>${x[0]}</td><td>${x[1]}</td><td>${x[2]}</td><td class="n">${(+x[3]).toFixed(4)}</td><td>${x[4]}</td></tr>`).join('') + '</tbody></table>';
}
document.getElementById('f-model').onchange=drawModel;

drawOver(); drawn.over=1;
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "streamlit_app" / "demo_data"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "index.html"))
    a = ap.parse_args()
    data = build(Path(a.gold))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__N__", f"{data['n_customers']:,}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    (out.parent / ".nojekyll").write_text("")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
