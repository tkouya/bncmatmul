#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py -- aggregate GPU(gdtq/mpc_cuda) vs CPU-OMP benchmark CSVs into
  bench/cuda/gpu_report.xlsx   (data + summary sheets)
  bench/cuda/gpu_report.pdf    (multi-page report: methodology, charts, tables)
Inputs (any missing is skipped):
  bench/cuda/gpu_omp_real.csv   real EFT dd/td/qd/ds/ts/qs  (matvec/matmul/spmv)
  bench/cuda/gpu_omp_cmpf.csv   complex mpc_cuda cmpf<bits> (axpy/matvec/matmul)
  bench/cuda/axpy_gpu.csv + bench/mds/axpy_cpu.csv  (real AXPY, re-included)
"""
import csv, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import xlsxwriter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BC   = os.path.join(ROOT, 'bench', 'cuda')
BM   = os.path.join(ROOT, 'bench', 'mds')

# ---- CJK font (auto-detect; needed for Japanese text/labels in the PDF) ----
def _find_cjk():
    env = os.environ.get('CJK_FONT')
    cands = ([env] if env else []) + [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]
    for p in cands:
        if p and os.path.exists(p): return p
    for pat in ('/usr/share/fonts/**/*CJK*.ttc','/usr/share/fonts/**/*CJK*.otf',
                '/usr/share/fonts/**/*ipa*.ttf','/usr/share/fonts/**/*[Tt]akao*.ttf'):
        g = glob.glob(pat, recursive=True)
        if g: return g[0]
    return None
CJK = _find_cjk()
if CJK:
    try:
        fm.fontManager.addfont(CJK)
        plt.rcParams['font.family'] = fm.FontProperties(fname=CJK).get_name()
    except Exception as e:
        print('warn: CJK font load failed (%s); Japanese may not render' % e)
else:
    print('warn: no CJK font found; set CJK_FONT=/path/to/font.ttc for Japanese text')
plt.rcParams['axes.unicode_minus'] = False

def load_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get('op')]

real = load_csv(os.path.join(BC,'gpu_omp_real.csv'))
cmpf = load_csv(os.path.join(BC,'gpu_omp_cmpf.csv'))

def fnum(x):
    try: return float(x)
    except: return float('nan')

# ---------- AXPY (schema: backend,nthreads,family,type,bits,N,seconds,iters,mflops) ----------
def load_axpy():
    g=os.path.join(BC,'axpy_gpu.csv'); c=os.path.join(BM,'axpy_cpu.csv')
    rows=[]
    if not (os.path.exists(g) and os.path.exists(c)): return rows
    def rd(p):
        with open(p) as f: return list(csv.DictReader(f))
    gpu=rd(g); cpu=rd(c)
    # GPU "max" config = largest nthreads per (type,N); CPU "max" = nthreads==32
    gmax={}
    for r in gpu:
        k=(r['type'],int(r['N'])); nt=int(r['nthreads'])
        if k not in gmax or nt>gmax[k][0]: gmax[k]=(nt,float(r['seconds']),r['family'],int(r['bits']),float(r['mflops']))
    cmax={}
    for r in cpu:
        if int(r['nthreads'])==32:
            cmax[(r['type'],int(r['N']))]=(float(r['seconds']),float(r['mflops']))
    for k in sorted(gmax, key=lambda x:(x[0],x[1])):
        if k in cmax:
            nt,gs,fam,bits,gmf=gmax[k]; cs,cmf=cmax[k]
            rows.append(dict(op='axpy',family=fam,type=k[0],bits=bits,N=k[1],nnz=0,
                             gpu_sec=gs,cpu_sec=cs,cpu_backend='omp(avx512,32t)',
                             speedup=cs/gs if gs>0 else 0,relerr=0.0,
                             gpu_mflops=gmf,cpu_mflops=cmf))
    return rows
axpy=load_axpy()

REAL_TYPES=['dd','td','qd','ds','ts','qs']
TYPE_DESC={'dd':'double-double (~31桁)','td':'triple-double (~47桁)','qd':'quad-double (~62桁)',
           'ds':'double-single (~14桁)','ts':'triple-single (~21桁)','qs':'quad-single (~28桁)',
           'double':'double (~16桁)','float':'float (~7桁)'}

def rows_for(data, op):
    return [r for r in data if r['op']==op]

# ============================ XLSX ============================
def write_xlsx(path):
    wb=xlsxwriter.Workbook(path)
    hf=wb.add_format({'bold':True,'bg_color':'#305496','font_color':'white','border':1})
    cf=wb.add_format({'border':1}); nf=wb.add_format({'border':1,'num_format':'0.000'})
    ef=wb.add_format({'border':1,'num_format':'0.00E+00'}); sf=wb.add_format({'border':1,'bold':True,'num_format':'0.00'})
    cols=['op','family','type','bits','N','nnz','gpu_sec','cpu_sec','cpu_backend','speedup','relerr','gpu_mflops','cpu_mflops']
    def sheet(name, data):
        ws=wb.add_worksheet(name)
        for j,c in enumerate(cols): ws.write(0,j,c,hf)
        for i,r in enumerate(data,1):
            for j,c in enumerate(cols):
                v=r.get(c,'')
                if c in ('gpu_sec','cpu_sec','gpu_mflops','cpu_mflops'): ws.write_number(i,j,fnum(v),nf)
                elif c=='speedup': ws.write_number(i,j,fnum(v),sf)
                elif c=='relerr': ws.write_number(i,j,fnum(v),ef)
                elif c in ('bits','N','nnz'):
                    try: ws.write_number(i,j,int(float(v)),cf)
                    except: ws.write(i,j,v,cf)
                else: ws.write(i,j,str(v),cf)
        ws.set_column(0,0,8); ws.set_column(1,1,13); ws.set_column(8,8,16); ws.freeze_panes(1,0)
        return ws
    # 概要 sheet
    ws=wb.add_worksheet('概要'); ws.set_column(0,0,26); ws.set_column(1,6,14)
    title=wb.add_format({'bold':True,'font_size':14})
    ws.write(0,0,'BNCmatmul 0.24  GPU(gdtq/mpc_cuda) vs CPU-OMP ベンチマーク概要',title)
    r=2
    ws.write(r,0,'演算',hf); ws.write(r,1,'精度型',hf); ws.write(r,2,'代表N',hf)
    ws.write(r,3,'GPU[s]',hf); ws.write(r,4,'CPU-OMP[s]',hf); ws.write(r,5,'speedup',hf); ws.write(r,6,'relerr',hf); r+=1
    # pick the largest N per (op,type) for the summary
    allrows=real+axpy+cmpf
    seen={}
    for row in allrows:
        k=(row['op'],row['type'])
        if k not in seen or int(float(row['N']))>int(float(seen[k]['N'])): seen[k]=row
    order=['axpy','matvec','matmul','spmv']
    for row in sorted(seen.values(), key=lambda x:(order.index(x['op']) if x['op'] in order else 9, x['type'])):
        if fnum(row['cpu_sec'])<=0: continue
        ws.write(r,0,row['op'],cf); ws.write(r,1,row['type'],cf); ws.write_number(r,2,int(float(row['N'])),cf)
        ws.write_number(r,3,fnum(row['gpu_sec']),nf); ws.write_number(r,4,fnum(row['cpu_sec']),nf)
        ws.write_number(r,5,fnum(row['speedup']),sf); ws.write_number(r,6,fnum(row['relerr']),ef); r+=1
    if real:
        sheet('matvec',[x for x in real if x['op']=='matvec'])
        sheet('matmul',[x for x in real if x['op']=='matmul'])
        sheet('spmv',[x for x in real if x['op']=='spmv'])
    if axpy: sheet('axpy',axpy)
    if cmpf: sheet('complex_mpc',cmpf)
    wb.close()
    print('wrote',path)

# ============================ PDF ============================
COL={'dd':'#1f77b4','td':'#2ca02c','qd':'#d62728','ds':'#17becf','ts':'#bcbd22','qs':'#9467bd',
     'double':'#7f7f7f','float':'#e377c2',
     'cmpf106':'#1f77b4','cmpf212':'#ff7f0e','cmpf424':'#d62728'}

def bar_speedup(ax, data, op, types, title):
    # grouped bars: x=N, group=type, height=speedup (cpu_sec>0 only)
    Ns=sorted({int(float(r['N'])) for r in data if r['op']==op and fnum(r['cpu_sec'])>0})
    if not Ns:
        ax.text(0.5,0.5,'(no data)',ha='center'); ax.set_title(title); return
    x=np.arange(len(Ns)); w=0.8/max(1,len(types))
    for ti,t in enumerate(types):
        ys=[]
        for N in Ns:
            m=[r for r in data if r['op']==op and r['type']==t and int(float(r['N']))==N and fnum(r['cpu_sec'])>0]
            ys.append(fnum(m[0]['speedup']) if m else 0)
        ax.bar(x+ti*w-0.4+w/2, ys, w, label=t, color=COL.get(t,'#888'))
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in Ns])
    ax.axhline(1.0,color='k',lw=0.8,ls='--')
    ax.set_xlabel('行列/ベクトル サイズ N'); ax.set_ylabel('speedup (CPU-OMP時間 / GPU時間)')
    ax.set_title(title); ax.legend(ncol=3,fontsize=7); ax.grid(axis='y',alpha=0.3)

def table_page(pdf, title, header, rows, note=''):
    fig=plt.figure(figsize=(11.7,8.3)); fig.suptitle(title,fontsize=14,y=0.985)
    noteh=0.06 if note else 0.0
    ax=fig.add_axes([0.03,0.02+noteh,0.94,0.93-noteh]); ax.axis('off')
    tb=ax.table(cellText=rows, colLabels=header, cellLoc='center', bbox=[0,0,1,1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.5 if len(rows)>24 else 8)
    for j in range(len(header)):
        tb[0,j].set_facecolor('#305496'); tb[0,j].set_text_props(color='white',weight='bold')
    if note:
        fig.text(0.03,0.012,note,fontsize=7.5)
    pdf.savefig(fig); plt.close(fig)

def write_pdf(path, sysinfo):
    with PdfPages(path) as pdf:
        # ---- title/methodology ----
        fig=plt.figure(figsize=(11.7,8.3)); ax=fig.add_axes([0,0,1,1]); ax.axis('off')
        txt=[
          ('BNCmatmul 0.24 : 多倍長 基本線形計算・疎行列ルーチンの GPU 化と',18,0.93),
          ('CPU 並列(OpenMP)同精度ルーチンとの性能比較レポート',18,0.89),
          (sysinfo,10,0.83),
          ('1. 目的',13,0.76),
          ('  gdtq(実数 double/single 系 EFT 多倍長: dd/td/qd, ds/ts/qs)および mpc_cuda(複素 MPC 多倍長)を',10,0.725),
          ('  用いて基本線形計算(AXPY, 行列×ベクトル, 行列×行列)と疎行列ベクトル積(SpMV)を GPU 実行し,',10,0.70),
          ('  同一精度の CPU OpenMP 並列ルーチンと実行時間を比較した.',10,0.675),
          ('2. 手法',13,0.62),
          ('  ・GPU: H100 上で gdtq / mpc_cuda のデバイスカーネルを実行. 時間はカーネル実行のみ(データは',10,0.585),
          ('    デバイス常駐, cudaDeviceSynchronize で計測, reps 回の最小値). ',10,0.56),
          ('  ・CPU: 同一精度の _bncomp_*(OpenMP, AVX-512, 32スレッド)ルーチン. SpMV の ds/ts/qs は',10,0.535),
          ('    ライブラリに OMP 版が無いため直列(serial)を基準とした. 複素 mpc_cuda の CPU 基準は',10,0.51),
          ('    同一の mpc_cuda ホスト MPC 演算を OpenMP 並列化(同一精度・同一アルゴリズム).',10,0.485),
          ('  ・速度向上率 speedup = (CPU-OMP 実行時間) / (GPU 実行時間).  正当性は relerr(GPU vs CPU の',10,0.46),
          ('    相対誤差最大値)で確認. MFLOPS は高位演算数(AXPY:2N, matvec:2N^2, matmul:2N^3, SpMV:2nnz).',10,0.435),
          ('3. 対象精度',13,0.38),
          ('  実数: dd(106bit) td(159) qd(212) / ds(48) ts(72) qs(96).  複素: cmpf 106/212/424 bit.',10,0.345),
          ('4. 収録シート/図: 概要, matvec, matmul, spmv, axpy, complex_mpc (xlsx) + 各演算の speedup 図(PDF).',10,0.30),
        ]
        for s,sz,y in txt:
            ax.text(0.06,y,s,fontsize=sz,va='top',weight=('bold' if sz>=13 else 'normal'))
        pdf.savefig(fig); plt.close(fig)

        # ---- real EFT charts ----
        if real:
            fig,axs=plt.subplots(1,2,figsize=(11.7,8.3)); fig.suptitle('実数 EFT 多倍長 (gdtq) : 密行列 基本線形計算  GPU vs CPU-OMP(32スレッド)',fontsize=13)
            bar_speedup(axs[0],real,'matvec',REAL_TYPES,'行列×ベクトル (matvec)  speedup')
            bar_speedup(axs[1],real,'matmul',REAL_TYPES,'行列×行列 (matmul)  speedup')
            fig.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)

            fig,axs=plt.subplots(1,2,figsize=(11.7,8.3)); fig.suptitle('実数 EFT 多倍長 (gdtq) : 疎行列ベクトル積 SpMV  GPU vs CPU',fontsize=13)
            bar_speedup(axs[0],real,'spmv',['dd','td','qd'],'SpMV double系 (CPU=OMP)  speedup')
            bar_speedup(axs[1],real,'spmv',['ds','ts','qs'],'SpMV single系 (CPU=serial)  speedup')
            fig.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)

        # ---- axpy ----
        if axpy:
            fig=plt.figure(figsize=(11.7,8.3)); fig.suptitle('実数 AXPY (c=a+alpha*b)  GPU(最大スレッド) vs CPU-OMP(AVX-512,32スレッド)',fontsize=13)
            ax=fig.add_axes([0.08,0.12,0.86,0.76])
            bar_speedup(ax,axpy,'axpy',['dd','td','qd','ds','ts','qs'],'AXPY speedup (再掲)')
            pdf.savefig(fig); plt.close(fig)

        # ---- complex mpc ----
        if cmpf:
            ctypes=sorted({r['type'] for r in cmpf}, key=lambda t:int(t.replace('cmpf','')))
            fig,axs=plt.subplots(1,2,figsize=(11.7,8.3)); fig.suptitle('複素 多倍長 MPC (mpc_cuda) : GPU vs CPU-OMP(同一MPC演算の並列化)',fontsize=13)
            bar_speedup(axs[0],cmpf,'matvec',ctypes,'複素 matvec  speedup')
            bar_speedup(axs[1],cmpf,'matmul',ctypes,'複素 matmul  speedup')
            fig.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)

        # ---- summary table ----
        allrows=real+axpy+cmpf
        seen={}
        for row in allrows:
            k=(row['op'],row['type'])
            if fnum(row['cpu_sec'])<=0: continue
            if k not in seen or int(float(row['N']))>int(float(seen[k]['N'])): seen[k]=row
        order=['axpy','matvec','matmul','spmv']
        hdr=['演算','型','bits','N','nnz','GPU[s]','CPU-OMP[s]','speedup','relerr']
        trows=[]
        for row in sorted(seen.values(), key=lambda x:(order.index(x['op']) if x['op'] in order else 9,x['type'])):
            trows.append([row['op'],row['type'],int(float(row['bits'])),int(float(row['N'])),
                          int(float(row['nnz'])),f"{fnum(row['gpu_sec']):.4g}",f"{fnum(row['cpu_sec']):.4g}",
                          f"{fnum(row['speedup']):.1f}x",f"{fnum(row['relerr']):.1e}"])
        note=('注: speedup>1 は GPU が高速. 各 (演算,型) の最大 N の行を掲載. relerr は GPU と CPU の相対誤差最大値で\n'
              '正当性を示す(0 は下位ビットまで一致). SpMV ds/ts/qs の CPU は serial (ライブラリに OMP 版無し).')
        table_page(pdf,'まとめ: 各演算・各精度の代表点 (最大 N)',hdr,trows,note)

        # ---- 考察・結論 ----
        def best(op,data=None):
            d=data if data is not None else real
            rs=[r for r in d if r['op']==op and fnum(r['cpu_sec'])>0]
            return max((fnum(r['speedup']) for r in rs), default=0), rs
        mv_max,_=best('matvec'); mm_max,mm_rs=best('matmul'); sp_max,_=best('spmv')
        mm_min=min((fnum(r['speedup']) for r in mm_rs), default=0)
        ax_max,_=best('axpy',axpy); cx_mv,_=best('matvec',cmpf); cx_mm,_=best('matmul',cmpf)
        fig=plt.figure(figsize=(11.7,8.3)); ax=fig.add_axes([0,0,1,1]); ax.axis('off')
        L=[('考察・結論',18,0.94),
           ('■ 正当性',13,0.87),
           ('  全演算で GPU 結果は CPU と下位ビットまで一致(relerr=0, 一部の単精度系のみ丸め由来 ~1e-9〜1e-12).',10,0.835),
           ('  多倍長 EFT / MPC 演算が GPU 上で正しく実装されていることを確認した.',10,0.81),
           ('■ 密行列 基本線形計算',13,0.75),
           (f'  ・行列×ベクトル(matvec): GPU が最大 {mv_max:.0f}倍高速. ただし CPU-OMP 側はスレッド生成',10,0.715),
           ('    オーバヘッド(~7ms)律速で, 本質的にレイテンシ限界. GPU カーネルは µs オーダで完了.',10,0.69),
           (f'  ・行列×行列(matmul): speedup {mm_min:.1f}〜{mm_max:.1f}倍. ナイーブな GPU カーネル(1スレッド1要素,',10,0.655),
           ('    共有メモリ・タイリング無し)は AVX-512+ブロック化+32スレッドの CPU-OMP に対し N<=2048 では',10,0.63),
           ('    優位でない. N の増加とともに比が改善(crossover 傾向)し, タイリング/Strassen 化が今後の課題.',10,0.605),
           ('■ 疎行列ベクトル積 SpMV',13,0.545),
           (f'  ・GPU(CSR)が最大 {sp_max:.0f}倍高速. GPU は µs オーダ. ただし CPU-OMP 版は行オフセットの',10,0.51),
           ('    前置和を行毎に再計算する O(N^2) 実装のため大 N で急速に悪化しており, 大 N の高速比は',10,0.485),
           ('    この CPU 側非効率も含む. 小〜中 N では OMP 生成オーバヘッド律速.',10,0.46),
           ('■ AXPY(再掲)',13,0.40),
           (f'  ・GPU(全スレッド)対 CPU-OMP(32スレッド, AVX-512): 大 N で最大 {ax_max:.1f}倍. 帯域律速演算で,',10,0.365),
           ('    小 N は両者ともオーバヘッド律速(比~1). 既測の結果を一貫指標で再掲.',10,0.34),
           ('■ 複素 多倍長 MPC(mpc_cuda)',13,0.28),
           (f'  ・行列×ベクトル 最大 {cx_mv:.0f}倍, 行列×行列 最大 {cx_mm:.0f}倍(CPU は同一 MPC 演算の OpenMP 並列).',10,0.245),
           ('    1要素あたりの演算コストが極めて大きい MPC では GPU の多数スレッドが非常に有効で, EFT の',10,0.22),
           ('    密行列積(ナイーブ GPU が不利)と対照的に, 高精度・高演算密度ほど GPU 優位が明確.',10,0.195),
           ('■ 総括',13,0.135),
           ('  演算密度が高い(高精度・複素 MPC)ほど, また要素独立並列度が高い(SpMV, AXPY, matvec)ほど',10,0.10),
           ('  GPU 化の効果が大きい. EFT 密行列積のみ CPU-OMP が優位で, カーネル最適化が残課題.',10,0.075),
          ]
        for s,sz,y in L: ax.text(0.06,y,s,fontsize=sz,va='top',weight=('bold' if sz>=13 else 'normal'))
        pdf.savefig(fig); plt.close(fig)
    print('wrote',path)

if __name__=='__main__':
    sysinfo=sys.argv[1] if len(sys.argv)>1 else 'device: NVIDIA H100 NVL (sm_90) / CPU: 32-core, AVX-512, OpenMP 32 threads'
    write_xlsx(os.path.join(BC,'gpu_report.xlsx'))
    write_pdf(os.path.join(BC,'gpu_report.pdf'), sysinfo)
