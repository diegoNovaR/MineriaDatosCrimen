"""
src/dashboard/dashboard_generator.py  —  v4

Cambios v4:
  - Sin límite de muestreo: se carga TODO el dataset
  - Canvas con renderizado por densidad: agrupa puntos cercanos en zoom-out
    para mantener 60fps con 150k-500k registros
  - Franja horaria corregida: Madrugada 0-5h, Mañana 6-11h,
    Tarde 12-17h, Noche 18-23h
  - Modo Franja convive con Hora×Hora y Día×Día
  - Filtro por mes dinámico
"""

import os
import json
import webbrowser
import pandas as pd

RUTA_DASHBOARD = os.path.join("outputs", "dashboard")

CITY_THEMES = {
    "chicago": {
        "label": "CHICAGO",
        "font_display": "Bebas Neue", "font_mono": "IBM Plex Mono", "font_body": "IBM Plex Sans",
        "font_url": "https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap",
        "bg": "#0a0a0f", "panel": "#11111a", "border": "#1e1e2e",
        "city_color": "#4f8eff", "city_color_dim": "#4f8eff55",
        "badge_style": "font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:3px;",
    },
    "philadelphia": {
        "label": "Philadelphia",
        "font_display": "Playfair Display", "font_mono": "Source Code Pro", "font_body": "Source Sans 3",
        "font_url": "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Code+Pro:wght@400;600&family=Source+Sans+3:wght@300;400;600&display=swap",
        "bg": "#0d0a06", "panel": "#141008", "border": "#2a2010",
        "city_color": "#f0c040", "city_color_dim": "#f0c04055",
        "badge_style": "font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:900;",
    },
    "san_francisco": {
        "label": "San Francisco",
        "font_display": "DM Serif Display", "font_mono": "DM Mono", "font_body": "DM Sans",
        "font_url": "https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap",
        "bg": "#060d14", "panel": "#0a1520", "border": "#0f2233",
        "city_color": "#f5a623", "city_color_dim": "#f5a62355",
        "badge_style": "font-family:'DM Serif Display',serif;font-size:1.4rem;",
    },
}

CATEGORIA_LABELS = {
    "agresion": "AGRESIÓN", "robo_simple": "ROBO SIMPLE",
    "daño_propiedad": "DAÑO PROPIEDAD", "robo_vehiculo": "ROBO VEHÍCULO",
    "warrant_otros": "WARRANT / OTROS", "fraude_engaño": "FRAUDE / ENGAÑO",
    "allanamiento": "ALLANAMIENTO", "drogas": "DROGAS",
    "robo_con_violencia": "ROBO CON VIOLENCIA", "violacion_armas": "VIOLACIÓN ARMAS",
    "agresion_sexual": "AGRESIÓN SEXUAL", "delito_contra_menores": "DELITO CONTRA MENORES",
}

MESES_NOMBRES = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
    7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre",
}

# Franjas corregidas: 0-5 / 6-11 / 12-17 / 18-23
FRANJAS = [
    ("madrugada", "MADRUGADA", list(range(0, 6)),  "#7b5cfa", "🌑"),
    ("manana",    "MAÑANA",    list(range(6, 12)),  "#f5a623", "🌅"),
    ("tarde",     "TARDE",     list(range(12, 18)), "#00c9a7", "☀️"),
    ("noche",     "NOCHE",     list(range(18, 24)), "#4f8eff", "🌙"),
]

def _hora_a_franja(h: int) -> str:
    if h <= 5:  return "madrugada"
    if h <= 11: return "manana"
    if h <= 17: return "tarde"
    return "noche"


# ── Sin muestreo: carga TODO el dataset ──────────────────────────────────────

def _df_to_js_array(df: pd.DataFrame, ciudad: str) -> str:
    print(f"\n    [data] {ciudad}: shape={df.shape}")
    print(f"    [data] columnas: {list(df.columns)}")

    # ── Conversión forzada de tipos ──────────────────────────────────────────
    # Cubre el caso más común: columnas leídas como string desde CSV
    df = df.copy()
    for col in ["latitud", "longitud"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hora" in df.columns:
        df["hora"] = pd.to_numeric(df["hora"], errors="coerce")
    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
    if "es_peligroso" in df.columns:
        # puede venir como True/False string, 0/1, o bool
        if df["es_peligroso"].dtype == object:
            df["es_peligroso"] = df["es_peligroso"].map(
                {"True": True, "False": False, "1": True, "0": False,
                 "true": True, "false": False}
            ).fillna(False)

    # ── Diagnóstico post-conversión ──────────────────────────────────────────
    for col in ["latitud", "longitud", "hora"]:
        if col in df.columns:
            n_null = df[col].isna().sum()
            sample = df[col].dropna().head(3).tolist()
            print(f"    [data] {col}: dtype={df[col].dtype}, nulos={n_null:,}, muestra={sample}")
        else:
            print(f"    [WARN] columna '{col}' NO EXISTE en el DataFrame de {ciudad}")

    df_clean = df.dropna(subset=["latitud", "longitud", "hora"]).copy()
    n_total  = len(df_clean)
    print(f"    [data] {ciudad}: {n_total:,} registros válidos — cargando SIN muestreo")
    if n_total == 0:
        print(f"    [ERROR] {ciudad}: 0 registros después de dropna. "
              f"Verifica que latitud/longitud/hora no sean todos NaN.")
        return "[]"
    if n_total > 300_000:
        print(f"    [warn] {ciudad}: dataset muy grande ({n_total:,}). "
              f"El HTML puede pesar >50 MB.")

    tiene_desc  = "descripcion" in df_clean.columns
    tiene_det   = "tipo_delito_detalle" in df_clean.columns
    tiene_fecha = "fecha" in df_clean.columns

    records = []
    for _, row in df_clean.iterrows():
        cat  = str(row.get("categoria_delito","")) if pd.notna(row.get("categoria_delito")) else ""
        det  = str(row.get("tipo_delito_detalle","")) if tiene_det and pd.notna(row.get("tipo_delito_detalle")) else cat
        desc = str(row.get("descripcion",""))[:60] if tiene_desc and pd.notna(row.get("descripcion")) else ""
        hora = int(row["hora"]) if pd.notna(row["hora"]) else 0
        mes  = int(row["mes"])  if pd.notna(row.get("mes")) else 1
        pel  = bool(row["es_peligroso"])

        if tiene_fecha and pd.notna(row["fecha"]):
            fecha_str = str(row["fecha"])[:10]
        else:
            anio = int(row["anio"]) if pd.notna(row.get("anio")) else 2025
            fecha_str = f"{anio}-{mes:02d}-01"

        records.append({
            "lat": round(float(row["latitud"]), 5),   # 5 dec = ~1m precisión
            "lon": round(float(row["longitud"]), 5),
            "h":   hora,
            "f":   fecha_str,
            "m":   mes,
            "fr":  _hora_a_franja(hora),
            "cat": cat,
            "det": det,
            "dsc": desc,
            "p":   1 if pel else 0,
        })

    # Serialización compacta (sin espacios)
    return json.dumps(records, ensure_ascii=False, separators=(',', ':'))


def _bounding_box(df: pd.DataFrame):
    la, lb = df["latitud"].min(),  df["latitud"].max()
    lo, lp = df["longitud"].min(), df["longitud"].max()
    pla, plo = (lb-la)*0.02, (lp-lo)*0.02
    return {"latMin":round(la-pla,5),"latMax":round(lb+pla,5),
            "lonMin":round(lo-plo,5),"lonMax":round(lp+plo,5)}


# ── HTML ──────────────────────────────────────────────────────────────────────

def _render_html(ciudad: str, df: pd.DataFrame) -> str:
    t = CITY_THEMES.get(ciudad, CITY_THEMES["chicago"])

    # Conversión de tipos antes de cualquier cálculo
    df = df.copy()
    for col in ["latitud", "longitud"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hora" in df.columns:
        df["hora"] = pd.to_numeric(df["hora"], errors="coerce")
    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
    if "es_peligroso" in df.columns and df["es_peligroso"].dtype == object:
        df["es_peligroso"] = df["es_peligroso"].map(
            {"True": True, "False": False, "1": True, "0": False,
             "true": True, "false": False}
        ).fillna(False)

    bb       = _bounding_box(df)
    js_data  = _df_to_js_array(df, ciudad)
    n_total  = len(df.dropna(subset=["latitud","longitud"]))
    n_danger = int(df["es_peligroso"].sum()) if "es_peligroso" in df.columns else 0

    meses_presentes = sorted(df["mes"].dropna().unique().astype(int).tolist()) if "mes" in df.columns else list(range(1,13))
    meses_js  = json.dumps({str(m): MESES_NOMBRES[m] for m in meses_presentes})
    franjas_js = json.dumps([
        {"id":fid,"label":flabel,"horas":horas,"color":color,"icono":icono}
        for fid,flabel,horas,color,icono in FRANJAS
    ], ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{t['label']} — Crime Temporal Map 2025</title>
<link href="{t['font_url']}" rel="stylesheet"/>
<style>
:root{{
  --bg:{t['bg']};--panel:{t['panel']};--bord:{t['border']};
  --danger:#ff3b5c;--safe:#00c9a7;--acc:{t['city_color']};
  --text:#e8e8f0;--muted:#5a5a7a;
  --fm:'{t['font_mono']}',monospace;--fb:'{t['font_body']}',sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:var(--fb);
      height:100vh;display:flex;flex-direction:column;overflow:hidden;}}
header{{background:var(--panel);border-bottom:1px solid var(--bord);
        padding:8px 16px;display:flex;align-items:center;gap:12px;flex-shrink:0;flex-wrap:wrap;}}
.badge{{{t['badge_style']}color:var(--acc);line-height:1;}}
.sub{{font-family:var(--fm);font-size:.6rem;color:var(--muted);letter-spacing:2px;margin-top:2px;}}
.leg{{display:flex;gap:12px;align-items:center;padding:0 8px;}}
.ld{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:3px;}}
.li{{font-size:.66rem;color:var(--muted);display:flex;align-items:center;}}
.info-bar{{display:flex;gap:5px;align-items:center;margin-left:auto;flex-wrap:wrap;}}
.ib-pill{{font-family:var(--fm);font-size:.57rem;padding:2px 7px;border-radius:10px;
          background:#ffffff0a;color:var(--muted);border:1px solid var(--bord);}}
.stats{{display:flex;gap:14px;margin-left:10px;}}
.sv{{font-family:var(--fm);font-size:.95rem;font-weight:600;}}
.sv.d{{color:var(--danger);}}.sv.s{{color:var(--safe);}}
.sl-lbl{{font-size:.57rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}}
.main{{display:flex;flex:1;overflow:hidden;}}
#mw{{flex:1;position:relative;overflow:hidden;background:#04040c;}}
canvas#mc{{width:100%;height:100%;display:block;cursor:crosshair;}}
#franja-badge{{position:absolute;top:10px;left:10px;z-index:10;
               font-family:var(--fm);font-size:.7rem;letter-spacing:2px;
               padding:4px 11px;border-radius:3px;border:1px solid;
               backdrop-filter:blur(6px);pointer-events:none;display:none;
               text-transform:uppercase;font-weight:600;}}
#perf-badge{{position:absolute;bottom:10px;left:10px;z-index:10;
             font-family:var(--fm);font-size:.55rem;color:var(--muted);
             background:var(--panel)cc;border:1px solid var(--bord);
             padding:3px 8px;border-radius:2px;pointer-events:none;}}
.side{{width:250px;background:var(--panel);border-left:1px solid var(--bord);
       display:flex;flex-direction:column;overflow:hidden;flex-shrink:0;}}
.st{{padding:10px 13px;font-family:var(--fm);font-size:.58rem;color:var(--muted);
     letter-spacing:2px;border-bottom:1px solid var(--bord);text-transform:uppercase;}}
#feed{{flex:1;overflow-y:auto;padding:6px;}}
#feed::-webkit-scrollbar{{width:3px;}}
#feed::-webkit-scrollbar-thumb{{background:var(--bord);}}
.fi{{padding:5px 7px;border-left:2px solid transparent;margin-bottom:3px;
     border-radius:2px;background:#13131f;animation:fi .2s ease;}}
.fi.d{{border-color:var(--danger);}}.fi.s{{border-color:var(--safe);}}
.fc{{font-family:var(--fm);font-size:.61rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;}}
.fc.d{{color:var(--danger);}}.fc.s{{color:var(--safe);}}
.fm{{font-size:.58rem;color:var(--muted);margin-top:2px;}}
@keyframes fi{{from{{opacity:0;transform:translateX(5px)}}to{{opacity:1;transform:none}}}}
.ctrl{{background:var(--panel);border-top:1px solid var(--bord);padding:7px 16px;flex-shrink:0;}}
.cr{{display:flex;align-items:center;gap:6px;margin-bottom:5px;flex-wrap:wrap;}}
.cl{{font-family:var(--fm);font-size:.58rem;color:var(--muted);letter-spacing:1px;
     text-transform:uppercase;white-space:nowrap;}}
.mb{{display:flex;gap:4px;flex-wrap:wrap;}}
.btn{{font-family:var(--fm);font-size:.57rem;padding:3px 8px;border:1px solid var(--bord);
      background:transparent;color:var(--muted);cursor:pointer;border-radius:2px;
      transition:all .15s;text-transform:uppercase;letter-spacing:1px;}}
.btn.on{{background:var(--acc);border-color:var(--acc);color:#000;}}
.btn.don{{background:var(--danger);border-color:var(--danger);color:#fff;}}
.btn.son{{background:var(--safe);border-color:var(--safe);color:#000;}}
.fbtn{{font-family:var(--fm);font-size:.57rem;padding:3px 9px;border-radius:2px;
       border:1px solid var(--bord);background:transparent;color:var(--muted);
       cursor:pointer;transition:all .15s;text-transform:uppercase;letter-spacing:1px;}}
.fbtn.on{{color:#000!important;font-weight:600;}}
.mes-row{{display:flex;align-items:center;gap:4px;padding:5px 0 3px;
          border-top:1px solid var(--bord);flex-wrap:wrap;}}
.mbtn{{font-family:var(--fm);font-size:.55rem;padding:2px 5px;border:1px solid var(--bord);
       background:transparent;color:var(--muted);cursor:pointer;border-radius:2px;transition:all .15s;}}
.mbtn.on{{background:var(--acc);border-color:var(--acc);color:#000;font-weight:600;}}
#sl{{flex:1;-webkit-appearance:none;height:3px;border-radius:2px;
     background:var(--bord);outline:none;cursor:pointer;min-width:80px;}}
#sl::-webkit-slider-thumb{{-webkit-appearance:none;width:12px;height:12px;
  border-radius:50%;background:var(--acc);cursor:pointer;}}
#sv{{font-family:var(--fm);font-size:.76rem;color:var(--acc);min-width:88px;text-align:right;}}
.pb{{width:27px;height:27px;border-radius:50%;border:1px solid var(--acc);
     background:transparent;color:var(--acc);cursor:pointer;font-size:.8rem;
     display:flex;align-items:center;justify-content:center;transition:all .15s;}}
.pb:hover{{background:var(--acc);color:#000;}}
.spb{{font-family:var(--fm);font-size:.57rem;padding:2px 6px;border:1px solid var(--bord);
      background:transparent;color:var(--muted);cursor:pointer;border-radius:2px;transition:all .15s;}}
.spb.on{{border-color:var(--acc);color:var(--acc);}}
.sep{{width:1px;height:16px;background:var(--bord);margin:0 2px;flex-shrink:0;}}
#tip{{position:fixed;background:{t['panel']}ee;border:1px solid var(--bord);
      border-radius:4px;padding:6px 10px;font-size:.66rem;pointer-events:none;
      display:none;z-index:100;max-width:215px;line-height:1.5;}}
.tc{{font-weight:600;text-transform:uppercase;font-family:var(--fm);letter-spacing:1px;font-size:.62rem;}}
.tc.d{{color:var(--danger);}}.tc.s{{color:var(--safe);}}
.tr{{color:var(--muted);font-size:.6rem;}}
</style>
</head>
<body>
<header>
  <div>
    <div class="badge">{t['label']}</div>
    <div class="sub">CRIME TEMPORAL MAP · 2025</div>
  </div>
  <div class="leg">
    <span class="li"><span class="ld" style="background:var(--danger)"></span>Peligroso</span>
    <span class="li"><span class="ld" style="background:var(--safe)"></span>No Peligroso</span>
  </div>
  <div class="info-bar">
    <span class="ib-pill">{n_total:,} registros totales</span>
    <span class="ib-pill" style="color:var(--danger)">{n_danger:,} peligrosos</span>
    <span class="ib-pill" id="pill-mes" style="color:var(--acc)">Todos los meses</span>
    <span class="ib-pill" id="pill-franja" style="display:none"></span>
  </div>
  <div class="stats">
    <div><div class="sv d" id="cd">0</div><div class="sl-lbl">Peligrosos</div></div>
    <div><div class="sv s" id="cs">0</div><div class="sl-lbl">No Peligros.</div></div>
    <div><div class="sv" id="ct" style="color:var(--text)">0</div><div class="sl-lbl">Visibles</div></div>
  </div>
</header>

<div class="main">
  <div id="mw">
    <canvas id="mc"></canvas>
    <div id="franja-badge"></div>
    <div id="perf-badge">cargando...</div>
  </div>
  <div class="side">
    <div class="st">▶ feed de crímenes</div>
    <div id="feed"></div>
  </div>
</div>

<div class="ctrl">
  <div class="cr">
    <span class="cl">Modo:</span>
    <div class="mb">
      <button class="btn on" id="mh" onclick="setMode('hora')">Hora×Hora</button>
      <button class="btn"    id="md" onclick="setMode('dia')">Día×Día</button>
      <button class="btn"    id="mf" onclick="setMode('franja')">Franja</button>
    </div>
    <div class="sep"></div>
    <span class="cl">Filtro:</span>
    <div class="mb">
      <button class="btn on" id="fa" onclick="setFilter('all')">Todos</button>
      <button class="btn"    id="fd" onclick="setFilter('danger')">Peligrosos</button>
      <button class="btn"    id="fs" onclick="setFilter('safe')">No peligrosos</button>
    </div>
    <span style="margin-left:auto"></span>
    <span class="cl">Vel:</span>
    <button class="spb"    onclick="setSpeed(2000,this)">0.5×</button>
    <button class="spb on" onclick="setSpeed(800,this)">1×</button>
    <button class="spb"    onclick="setSpeed(300,this)">3×</button>
    <button class="spb"    onclick="setSpeed(80,this)">10×</button>
  </div>

  <div id="franja-row" class="cr" style="display:none;border-top:1px solid var(--bord);padding-top:5px;">
    <span class="cl">Franja:</span>
    <div class="mb" id="franja-btns"></div>
  </div>

  <div class="mes-row">
    <span class="cl">Mes:</span>
    <button class="mbtn on" id="m-all" onclick="setMes(0,this)">TODOS</button>
    <span id="mes-btns"></span>
  </div>

  <div class="cr" style="margin-bottom:0;margin-top:4px;">
    <button class="pb" id="pb" onclick="togglePlay()">▶</button>
    <input type="range" id="sl" min="0" max="23" value="0" oninput="onSlider(this.value)"/>
    <span id="sv">00:00 h</span>
  </div>
</div>
<div id="tip"></div>

<script>
// ── Dataset completo ──────────────────────────────────────────────────────────
const ALL       = {js_data};
const BB        = {json.dumps(bb)};
const CAT_LABEL = {json.dumps(CATEGORIA_LABELS, ensure_ascii=False)};
const MESES_MAP = {meses_js};
const FRANJAS   = {franjas_js};
const MESES_ES  = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"];

console.log(`[dataset] ${{ALL.length.toLocaleString()}} registros cargados`);
document.getElementById('perf-badge').textContent =
  `${{ALL.length.toLocaleString()}} pts totales`;

// ── Índices por hora para acceso O(1) ────────────────────────────────────────
// Evita recorrer todo ALL en cada frame del modo Hora×Hora
const IDX_HORA = {{}};
for(let h=0;h<24;h++) IDX_HORA[h]=[];
ALL.forEach((d,i)=> IDX_HORA[d.h].push(i));

// Índice por fecha
const IDX_FECHA = {{}};
ALL.forEach((d,i)=>{{
  if(!IDX_FECHA[d.f]) IDX_FECHA[d.f]=[];
  IDX_FECHA[d.f].push(i);
}});

// Índice por franja
const IDX_FRANJA = {{}};
FRANJAS.forEach(fr=> IDX_FRANJA[fr.id]=[]);
ALL.forEach((d,i)=> IDX_FRANJA[d.fr].push(i));

// Índice por mes (para filtro de mes)
const IDX_MES = {{}};
for(let m=1;m<=12;m++) IDX_MES[m]=new Set();
ALL.forEach((d,i)=> {{ if(IDX_MES[d.m]) IDX_MES[d.m].add(i); }});

// ── Botones de franjas ────────────────────────────────────────────────────────
(function(){{
  const cont=document.getElementById('franja-btns');
  FRANJAS.forEach((fr,i)=>{{
    const b=document.createElement('button');
    b.className='fbtn'+(i===0?' on':'');
    b.id='fr-'+fr.id;
    b.style.borderColor=fr.color;
    if(i===0){{b.style.background=fr.color;b.style.color='#000';}}
    else b.style.color=fr.color;
    b.textContent=fr.icono+' '+fr.label;
    b.dataset.color=fr.color;
    b.onclick=()=>setFranja(fr.id,b);
    cont.appendChild(b);
  }});
}})();

// ── Botones de meses ──────────────────────────────────────────────────────────
(function(){{
  const cont=document.getElementById('mes-btns');
  Object.entries(MESES_MAP).forEach(([num,nombre])=>{{
    const b=document.createElement('button');
    b.className='mbtn'; b.id='m-'+num;
    b.textContent=nombre.slice(0,3).toUpperCase(); b.title=nombre;
    b.onclick=()=>setMes(parseInt(num),b);
    cont.appendChild(b);
  }});
}})();

// ── Fechas para modo día ──────────────────────────────────────────────────────
const FECHAS_ALL=[...new Set(ALL.map(d=>d.f))].sort();
let FECHAS=FECHAS_ALL;
function fechaLabel(f){{
  const[y,m,d]=f.split('-');
  return `${{parseInt(d)}} ${{MESES_ES[parseInt(m)]}} ${{y}}`;
}}

// ── Estado ────────────────────────────────────────────────────────────────────
let mode='hora', filter='all', mesActivo=0, franjaActiva=FRANJAS[0].id;
let playing=false, pos=0, maxPos=23, speed=800;
let timer=null, vis=[], hover=null;
let frameReq=null;

const canvas=document.getElementById('mc');
const ctx=canvas.getContext('2d');

function resize(){{
  const w=document.getElementById('mw');
  canvas.width =w.clientWidth *devicePixelRatio;
  canvas.height=w.clientHeight*devicePixelRatio;
  canvas.style.width =w.clientWidth +'px';
  canvas.style.height=w.clientHeight+'px';
  schedDraw();
}}
window.addEventListener('resize',resize);
setTimeout(resize,60);

function xy(lat,lon){{
  const w=canvas.width,h=canvas.height;
  return[
    ((lon-BB.lonMin)/(BB.lonMax-BB.lonMin))*w,
    (1-(lat-BB.latMin)/(BB.latMax-BB.latMin))*h
  ];
}}

// ── Obtener índices filtrados (usa índices pre-calculados) ────────────────────
function getIndices(){{
  let base;
  if(mode==='hora')        base=IDX_HORA[pos];
  else if(mode==='dia')    base=IDX_FECHA[FECHAS[pos]]||[];
  else                     base=IDX_FRANJA[franjaActiva];

  // Filtro mes
  if(mesActivo!==0){{
    const mset=IDX_MES[mesActivo];
    base=base.filter(i=>mset.has(i));
  }}
  // Filtro peligrosidad
  if(filter==='danger')      base=base.filter(i=>ALL[i].p===1);
  else if(filter==='safe')   base=base.filter(i=>ALL[i].p===0);
  return base;
}}

// ── Renderizado con densidad adaptativa ──────────────────────────────────────
// Si hay >5000 puntos visibles, agrupa por celda de grilla para mantener fps
const DENSITY_THRESHOLD = 5000;
const GRID_CELLS        = 120;   // resolución de agrupación

function draw(){{
  const w=canvas.width, h=canvas.height;
  ctx.clearRect(0,0,w,h);

  // Grid
  ctx.strokeStyle='rgba(20,20,40,0.4)'; ctx.lineWidth=.5;
  for(let i=0;i<=10;i++){{
    ctx.beginPath();ctx.moveTo(i*w/10,0);ctx.lineTo(i*w/10,h);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,i*h/10);ctx.lineTo(w,i*h/10);ctx.stroke();
  }}

  const indices=getIndices();
  vis=indices.map(i=>ALL[i]);

  const franjaObj=FRANJAS.find(f=>f.id===franjaActiva);
  let nd=0,ns=0;
  const t0=performance.now();

  if(vis.length>DENSITY_THRESHOLD){{
    // Modo densidad: agrupar en grilla y dibujar una celda por grupo
    const cellW=w/GRID_CELLS, cellH=h/GRID_CELLS;
    const gridD={{}}, gridS={{}};
    vis.forEach(d=>{{
      const[px,py]=xy(d.lat,d.lon);
      const cx=Math.floor(px/cellW), cy=Math.floor(py/cellH);
      const key=cx+','+cy;
      if(d.p){{ gridD[key]=(gridD[key]||0)+1; nd++; }}
      else   {{ gridS[key]=(gridS[key]||0)+1; ns++; }}
    }});
    // Dibujar celdas de peligrosos
    Object.entries(gridD).forEach(([key,cnt])=>{{
      const[cx,cy]=key.split(',').map(Number);
      const px=(cx+.5)*cellW, py=(cy+.5)*cellH;
      const r=Math.min(1.5+Math.log2(cnt+1)*1.8, cellW*.9)*devicePixelRatio;
      ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2);
      const alpha=Math.min(0.4+cnt/30,0.95);
      ctx.fillStyle=`rgba(255,59,92,${{alpha}})`; ctx.fill();
    }});
    // Dibujar celdas de no peligrosos
    const safeCol=mode==='franja'?(franjaObj?franjaObj.color:'#00c9a7'):'#00c9a7';
    const sr=parseInt(safeCol.slice(1,3),16),
          sg=parseInt(safeCol.slice(3,5),16),
          sb=parseInt(safeCol.slice(5,7),16);
    Object.entries(gridS).forEach(([key,cnt])=>{{
      const[cx,cy]=key.split(',').map(Number);
      const px=(cx+.5)*cellW, py=(cy+.5)*cellH;
      const r=Math.min(1.2+Math.log2(cnt+1)*1.5, cellW*.9)*devicePixelRatio;
      ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2);
      const alpha=Math.min(0.3+cnt/40,0.85);
      ctx.fillStyle=`rgba(${{sr}},${{sg}},${{sb}},${{alpha}})`; ctx.fill();
    }});
  }} else {{
    // Modo individual: dibuja cada punto con glow
    vis.forEach(d=>{{
      const[px,py]=xy(d.lat,d.lon);
      const col=d.p?'#ff3b5c':(mode==='franja'&&franjaObj?franjaObj.color:'#00c9a7');
      const r=(d.p?4.2:3.2)*devicePixelRatio;
      ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2);
      ctx.fillStyle=col+(d.p?'99':'77'); ctx.fill();
      const g=ctx.createRadialGradient(px,py,0,px,py,r*2.4);
      g.addColorStop(0,col+'33'); g.addColorStop(1,col+'00');
      ctx.beginPath(); ctx.arc(px,py,r*2.4,0,Math.PI*2);
      ctx.fillStyle=g; ctx.fill();
      if(d.p) nd++; else ns++;
    }});
  }}

  // Hover ring
  if(hover){{
    const[hx,hy]=xy(hover.lat,hover.lon);
    ctx.beginPath(); ctx.arc(hx,hy,9*devicePixelRatio,0,Math.PI*2);
    ctx.strokeStyle=hover.p?'#ff3b5c':'#00c9a7';
    ctx.lineWidth=1.5*devicePixelRatio; ctx.stroke();
  }}

  // Barra progreso día
  if(mode==='dia'&&FECHAS.length>1){{
    const pct=pos/(FECHAS.length-1);
    ctx.fillStyle='rgba(255,255,255,0.05)';
    ctx.fillRect(0,h-4*devicePixelRatio,w,4*devicePixelRatio);
    ctx.fillStyle='{t['city_color']}';
    ctx.fillRect(0,h-4*devicePixelRatio,w*pct,4*devicePixelRatio);
  }}

  const ms=Math.round(performance.now()-t0);
  document.getElementById('cd').textContent=nd.toLocaleString();
  document.getElementById('cs').textContent=ns.toLocaleString();
  document.getElementById('ct').textContent=vis.length.toLocaleString();
  document.getElementById('perf-badge').textContent=
    `${{vis.length.toLocaleString()}} pts · ${{ms}}ms · ${{ALL.length.toLocaleString()}} total`;
}}

function schedDraw(){{
  if(frameReq) cancelAnimationFrame(frameReq);
  frameReq=requestAnimationFrame(draw);
}}

// ── Label slider ──────────────────────────────────────────────────────────────
function labelPos(){{
  const el=document.getElementById('sv');
  if(mode==='hora')      el.textContent=String(pos).padStart(2,'0')+':00 h';
  else if(mode==='dia')  el.textContent=FECHAS[pos]?fechaLabel(FECHAS[pos]):'—';
  else                   el.textContent='—';
}}

function updateFeed(){{
  let tl;
  if(mode==='hora')     tl=String(vis[0]?.h??pos).padStart(2,'0')+':00';
  else if(mode==='dia') tl=FECHAS[pos]?fechaLabel(FECHAS[pos]):'—';
  else{{ const fr=FRANJAS.find(f=>f.id===franjaActiva); tl=fr?fr.icono+' '+fr.label:franjaActiva; }}
  document.getElementById('feed').innerHTML=
    vis.slice(0,14).map(d=>`
      <div class="fi ${{d.p?'d':'s'}}">
        <div class="fc ${{d.p?'d':'s'}}">${{CAT_LABEL[d.cat]||d.cat}}</div>
        <div class="fm">${{d.det}}${{d.dsc?' · '+d.dsc:''}} · ${{tl}}</div>
      </div>`).join('');
}}

// ── Setters ───────────────────────────────────────────────────────────────────
function setMode(m){{
  mode=m; pos=0;
  ['mh','md','mf'].forEach(id=>document.getElementById(id).classList.remove('on'));
  document.getElementById(m==='hora'?'mh':m==='dia'?'md':'mf').classList.add('on');
  const fr=document.getElementById('franja-row');
  const sl=document.getElementById('sl');
  const badge=document.getElementById('franja-badge');
  if(m==='franja'){{
    fr.style.display='flex'; sl.style.opacity='.3'; sl.disabled=true;
    badge.style.display='block'; updateFranjaBadge();
  }}else{{
    fr.style.display='none'; sl.style.opacity='1'; sl.disabled=false;
    badge.style.display='none';
    document.getElementById('pill-franja').style.display='none';
    maxPos=m==='hora'?23:Math.max(0,FECHAS.length-1);
    sl.max=maxPos; sl.value=0;
  }}
  labelPos(); schedDraw(); updateFeed();
}}

function setFranja(fid,btn){{
  franjaActiva=fid;
  document.querySelectorAll('.fbtn').forEach(b=>{{
    b.classList.remove('on');
    b.style.background='transparent'; b.style.color=b.dataset.color;
  }});
  btn.classList.add('on');
  btn.style.background=btn.dataset.color; btn.style.color='#000';
  updateFranjaBadge(); schedDraw(); updateFeed();
}}

function updateFranjaBadge(){{
  const fr=FRANJAS.find(f=>f.id===franjaActiva);
  if(!fr)return;
  const badge=document.getElementById('franja-badge');
  const h0=fr.horas[0].toString().padStart(2,'0');
  const h1=(fr.horas[fr.horas.length-1]+1).toString().padStart(2,'0');
  badge.textContent=fr.icono+'  '+fr.label+'  '+h0+'h – '+h1+'h';
  badge.style.color=fr.color; badge.style.borderColor=fr.color;
  badge.style.background=fr.color+'18';
  const pill=document.getElementById('pill-franja');
  pill.style.display='inline-block'; pill.style.color=fr.color;
  pill.textContent=fr.icono+' '+fr.label;
}}

function setFilter(f){{
  filter=f;
  document.getElementById('fa').className='btn'+(f==='all'?' on':'');
  document.getElementById('fd').className='btn'+(f==='danger'?' don':'');
  document.getElementById('fs').className='btn'+(f==='safe'?' son':'');
  schedDraw(); updateFeed();
}}

function setMes(m,btn){{
  mesActivo=m;
  document.querySelectorAll('.mbtn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById('pill-mes').textContent=
    m===0?'Todos los meses':MESES_MAP[String(m)];
  FECHAS=m===0?FECHAS_ALL
    :[...new Set(ALL.filter(d=>d.m===m).map(d=>d.f))].sort();
  if(mode==='dia'){{
    maxPos=Math.max(0,FECHAS.length-1);
    const sl=document.getElementById('sl');
    sl.max=maxPos; pos=0; sl.value=0;
  }}
  labelPos(); schedDraw(); updateFeed();
}}

function onSlider(v){{ if(mode==='franja')return; pos=+v; labelPos(); schedDraw(); updateFeed(); }}

function togglePlay(){{
  if(mode==='franja')return;
  playing=!playing;
  document.getElementById('pb').textContent=playing?'⏸':'▶';
  if(playing) animate(); else clearTimeout(timer);
}}

function animate(){{
  if(!playing)return;
  pos=(pos+1)%(maxPos+1);
  document.getElementById('sl').value=pos;
  labelPos(); schedDraw(); updateFeed();
  timer=setTimeout(animate,speed);
}}

function setSpeed(s,btn){{
  speed=s;
  document.querySelectorAll('.spb').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
}}

// ── Tooltip ───────────────────────────────────────────────────────────────────
canvas.addEventListener('mousemove',e=>{{
  // En modo densidad el hover no funciona sobre puntos individuales
  if(vis.length>DENSITY_THRESHOLD){{
    hover=null; document.getElementById('tip').style.display='none'; return;
  }}
  const r=canvas.getBoundingClientRect();
  const mx=(e.clientX-r.left)*devicePixelRatio;
  const my=(e.clientY-r.top)*devicePixelRatio;
  let best=null, bd=18*devicePixelRatio;
  vis.forEach(d=>{{
    const[x,y]=xy(d.lat,d.lon);
    const dist=Math.hypot(x-mx,y-my);
    if(dist<bd){{bd=dist;best=d;}}
  }});
  const tip=document.getElementById('tip');
  if(best){{
    hover=best;
    tip.style.display='block';
    tip.style.left=(e.clientX+14)+'px';
    tip.style.top=(e.clientY-8)+'px';
    const fr=FRANJAS.find(f=>f.id===best.fr);
    tip.innerHTML=`
      <div class="tc ${{best.p?'d':'s'}}">${{CAT_LABEL[best.cat]||best.cat}}</div>
      <div class="tr">Tipo: ${{best.det}}</div>
      ${{best.dsc?'<div class="tr">Desc: '+best.dsc+'</div>':''}}
      <div class="tr">Hora: ${{String(best.h).padStart(2,'0')}}:00 · ${{fr?fr.icono+' '+fr.label:''}} · ${{MESES_ES[best.m]}}</div>
      <div class="tr">Lat ${{best.lat.toFixed(4)}} / Lon ${{best.lon.toFixed(4)}}</div>
      <div class="tr" style="margin-top:3px;color:${{best.p?'#ff3b5c':'#00c9a7'}};font-weight:600">
        ${{best.p?'⚠ PELIGROSO':'✓ NO PELIGROSO'}}</div>`;
  }}else{{hover=null;tip.style.display='none';}}
  schedDraw();
}});
canvas.addEventListener('mouseleave',()=>{{
  hover=null; document.getElementById('tip').style.display='none'; schedDraw();
}});

schedDraw(); updateFeed();
</script>
</body>
</html>"""


# ── Index ─────────────────────────────────────────────────────────────────────

def _render_index(ciudades_generadas: list) -> str:
    cards_html = ""
    for ciudad in ciudades_generadas:
        t = CITY_THEMES.get(ciudad, CITY_THEMES["chicago"])
        cards_html += f"""
  <a class="card" href="{ciudad}.html" style="--cc:{t['city_color']};--ccd:{t['city_color_dim']}">
    <div class="card-top">
      <canvas class="dots" id="dots-{ciudad}"></canvas>
      <div class="card-city" style="{t['badge_style']} color:var(--cc)">{t['label']}</div>
    </div>
    <div class="card-body">
      <div class="card-arrow"><span></span> ABRIR DASHBOARD →</div>
    </div>
  </a>"""
    dots_js = "\n".join(
    "  animDots('dots-" + c + "', '" +
    CITY_THEMES.get(c, CITY_THEMES['chicago'])['city_color'] +
    "');"
    for c in ciudades_generadas
)
    return f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Crime Dashboard — Urban Analysis 2025</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#08080f;color:#e0e0f0;font-family:'IBM Plex Sans',sans-serif;
      min-height:100vh;display:flex;flex-direction:column;
      align-items:center;justify-content:center;padding:40px 20px;}}
h1{{font-family:'Bebas Neue',sans-serif;font-size:3.2rem;letter-spacing:6px;
    text-align:center;margin-bottom:6px;
    background:linear-gradient(135deg,#fff 40%,#ff3b5c);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.sub{{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#4a4a6a;
      letter-spacing:3px;text-align:center;text-transform:uppercase;margin-bottom:44px;}}
.cards{{display:flex;gap:22px;flex-wrap:wrap;justify-content:center;}}
.card{{width:240px;background:#0e0e1a;border:1px solid #1a1a2e;border-radius:4px;
       overflow:hidden;text-decoration:none;color:#e0e0f0;
       transition:transform .2s,border-color .2s;cursor:pointer;}}
.card:hover{{transform:translateY(-5px);border-color:var(--cc);}}
.card-top{{height:110px;position:relative;overflow:hidden;
           display:flex;align-items:flex-end;padding:12px;
           background:linear-gradient(135deg,#050518,#0a0a28);}}
.dots{{position:absolute;top:0;left:0;right:0;bottom:0;}}
.card-city{{position:relative;z-index:1;line-height:1;}}
.card-body{{padding:12px;}}
.card-arrow{{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:#4a4a6a;
             letter-spacing:2px;display:flex;align-items:center;gap:6px;}}
.card-arrow span{{flex:1;height:1px;background:#1a1a2e;}}
.leg{{display:flex;gap:18px;margin-top:36px;justify-content:center;}}
.li{{display:flex;align-items:center;gap:5px;font-size:.66rem;color:#4a4a6a;
     font-family:'IBM Plex Mono',monospace;letter-spacing:1px;text-transform:uppercase;}}
.ld{{width:8px;height:8px;border-radius:50%;}}
</style></head><body>
<h1>URBAN CRIME MAP</h1>
<div class="sub">Análisis Temporal · Peligroso vs No Peligroso · 2025</div>
<div class="cards">{cards_html}</div>
<div class="leg">
  <span class="li"><span class="ld" style="background:#ff3b5c"></span>Peligroso</span>
  <span class="li"><span class="ld" style="background:#00c9a7"></span>No peligroso</span>
</div>
<script>
function animDots(id,col){{
  const c=document.getElementById(id); if(!c)return;
  const ctx=c.getContext('2d');
  const W=c.parentElement.clientWidth,H=c.parentElement.clientHeight;
  c.width=W;c.height=H;
  const hex=col.replace('#','');
  const r=parseInt(hex.slice(0,2),16),g=parseInt(hex.slice(2,4),16),b=parseInt(hex.slice(4,6),16);
  const pts=Array.from({{length:55}},()=>({{
    x:Math.random()*W,y:Math.random()*H,
    vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4,
    r:1.2+Math.random()*2,
    col:`rgba(${{r}},${{g}},${{b}},${{.3+Math.random()*.4}})`
  }}));
  (function step(){{
    ctx.clearRect(0,0,W,H);
    pts.forEach(p=>{{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=p.col;ctx.fill();
    }});
    requestAnimationFrame(step);
  }})();
}}
{dots_js}
</script></body></html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def generar_dashboard(datasets: dict) -> None:
    os.makedirs(RUTA_DASHBOARD, exist_ok=True)
    print(f"\n{'='*55}\n  GENERANDO DASHBOARD INTERACTIVO\n{'='*55}")
    ciudades_ok = []
    for ciudad, df in datasets.items():
        if "latitud" not in df.columns or "longitud" not in df.columns:
            print(f"  [SKIP] {ciudad}: sin columnas latitud/longitud"); continue
        if ciudad not in CITY_THEMES:
            print(f"  [SKIP] {ciudad}: sin tema definido"); continue
        html = _render_html(ciudad, df)
        ruta = os.path.join(RUTA_DASHBOARD, f"{ciudad}.html")
        with open(ruta,"w",encoding="utf-8") as f: f.write(html)
        n = len(df.dropna(subset=["latitud","longitud"]))
        p = int(df["es_peligroso"].sum()) if "es_peligroso" in df.columns else 0
        print(f"  [OK] {ciudad:<18} → {ruta}  ({n:,} pts · {p:,} peligrosos)")
        ciudades_ok.append(ciudad)
    if not ciudades_ok:
        print("  [ERROR] No se generó ningún dashboard."); return
    idx_html = _render_index(ciudades_ok)
    idx_ruta = os.path.join(RUTA_DASHBOARD, "index.html")
    with open(idx_ruta,"w",encoding="utf-8") as f: f.write(idx_html)
    print(f"  [OK] index            → {idx_ruta}")
    print(f"\n  ✔ Dashboard listo en: {RUTA_DASHBOARD}/\n  Ciudades: {ciudades_ok}")
    resp = input("\n  ¿Abrir el dashboard en el navegador? (s/n): ").strip().lower()
    if resp=="s":
        webbrowser.open(f"file://{os.path.abspath(idx_ruta)}")
        print("  [Browser] Abriendo index.html ...")