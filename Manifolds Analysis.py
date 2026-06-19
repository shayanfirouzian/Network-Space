import pandas as pd

d50 = pd.read_csv(".\global exhaustive manifold (N=50).csv")
d100 = pd.read_csv(".\global exhaustive manifold (N=100).csv")
d150 = pd.read_csv(".\global exhaustive manifold (N=150).csv")

def full_report(df_in, N):
    nd = df_in[df_in['rho'] > 0.01].copy()
    total = len(nd)
    er  = nd[(nd['eta']<=0.10)&(nd['c']<=0.20)]
    wst = nd[(nd['eta']<=0.10)&(nd['c']>0.20)&(nd['c']<=0.50)]
    wsd = nd[(nd['eta']<=0.10)&(nd['c']>0.50)]
    sf  = nd[(nd['eta']>0.10)&(nd['eta']<=0.20)]
    cp  = nd[nd['eta']>0.20]
    
    # N=50 Hierarchical Paradox count (rho>0.05 filter)
    hier_strict = nd[(nd['eta']>0.10)&(nd['c']>0.20)&(nd['rho']>0.05)]
    hier_all    = nd[(nd['eta']>0.10)&(nd['c']>0.20)]

    print(f"\n{'='*70}")
    print(f"N = {N}  |  Active manifold (ρ>0.01): {total:,}")
    print(f"{'='*70}")
    rows = [("I",  "Erdős-Rényi",          er,  "η≤0.10, c≤0.20"),
            ("II", "WS-Transitional",      wst, "η≤0.10, 0.20<c≤0.50"),
            ("III","WS-Dense-Clustered",   wsd, "η≤0.10, c>0.50"),
            ("IV", "BA-Scale-Free",        sf,  "0.10<η≤0.20"),
            ("V",  "BA-Core-Periphery",    cp,  "η>0.20")]
    print(f"\n{'#':<4} {'Archetype':<25} {'Condition':<25} {'n':>8} {'%':>6}"
          f"  {'ρ̄':>5} {'η̄':>5} {'c̄':>5} {'μ̄':>5}  {'τ_c':>7} {'RI':>6}")
    print("-"*105)
    for num, name, sub, cond in rows:
        r,e,c,m = (sub['rho'].mean(),sub['eta'].mean(),
                   sub['c'].mean(),sub['mu'].mean())
        tau = 1/((N-1)*r*(1+e)) if r>0 else 9.999
        ri  = c
        print(f"{num:<4} {name:<25} {cond:<25} {len(sub):>8,} {len(sub)/total*100:>6.1f}"
              f"  {r:>5.3f} {e:>5.3f} {c:>5.3f} {m:>5.3f}  {tau:>7.4f} {ri:>6.3f}")
    print(f"\n  MECE: {sum(len(s) for _,_,s,_ in rows)}/{total} = "
          f"{sum(len(s) for _,_,s,_ in rows)/total*100:.1f}%")
    print(f"  Hierarchical Paradox (η>0.10, c>0.20, ρ>0.01): {len(hier_all):,} pts")
    print(f"  Hierarchical Paradox (η>0.10, c>0.20, ρ>0.05): {len(hier_strict):,} pts")
    return nd

nd50  = full_report(d50,50)
nd100 = full_report(d100,100)
nd150 = full_report(d150,150)

#%%
# --- Sensitivity analysis for the revised partition ---
print("\n\n=== SENSITIVITY ANALYSIS ===\n\n")

for N in [50, 100, 150]:
    print(f"{'='*70}")
    print(f"N = {N}")
    print(f"{'='*70}\n")
    nd=[]
    exec(f'nd = nd{int(N)}')
    if len(nd) == 0:
        print('Error! Dataframe was not attributed.')
    else:
        print("Primary thresholds (η_τ, c_τ): perturbation ±0.02")
        print(f"{'η_τ':<6} {'c_τ':<6}  {'I-ER':>7} {'II-WST':>8} {'III-WSD':>9} {'IV-SF':>7} {'V-CP':>6}")

        for eta_t in [0.08, 0.10, 0.12]:
            for c_t in [0.18, 0.20, 0.22]:
                er_  = nd[(nd['eta']<=eta_t)&(nd['c']<=c_t)]
                wst_ = nd[(nd['eta']<=eta_t)&(nd['c']>c_t)&(nd['c']<=0.50)]
                wsd_ = nd[(nd['eta']<=eta_t)&(nd['c']>0.50)]
                sf_  = nd[(nd['eta']>eta_t)&(nd['eta']<=0.20)]
                cp_  = nd[nd['eta']>0.20]
                N_nd = len(nd)
                print(f"{eta_t:<6.2f} {c_t:<6.2f}  {len(er_)/N_nd*100:>7.1f}"
                      f"{len(wst_)/N_nd*100:>8.1f}{len(wsd_)/N_nd*100:>9.1f}"
                      f"{len(sf_)/N_nd*100:>7.1f}{len(cp_)/N_nd*100:>6.1f}")
        print("\nInternal WS threshold (c_WS): perturbation ±0.05")

        print(f"{'c_WS':<6}  {'II-WST':>8} {'III-WSD':>9}  RI_ratio")
        for c_ws in [0.40, 0.45, 0.50, 0.55, 0.60]:
            ws  = nd[(nd['eta']<=0.10)&(nd['c']>0.20)]
            wst_= ws[ws['c']<=c_ws]
            wsd_= ws[ws['c']>c_ws]
            ri_t = wst_['c'].mean() if len(wst_)>0 else 0
            ri_d = wsd_['c'].mean() if len(wsd_)>0 else 0
            ratio = ri_d/ri_t if ri_t>0 else 0
            print(f"{c_ws:<6.2f}  {len(wst_)/len(nd)*100:>8.1f}{len(wsd_)/len(nd)*100:>9.1f}"
                  f"  RI_ratio={ratio:.2f}x")
        print('\n')