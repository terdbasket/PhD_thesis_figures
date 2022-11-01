from Functions.General_functions import *
from scipy import interpolate
m_B = 1.8266e-26

M_pB = m_B*const.m_p/(m_B+const.m_p)

csv_path = os.path.join(os.getcwd(), 'CSVs')

K_E_axis = np.logspace(0,3,num = 100)
T_E_axis = np.logspace(0,3,num = 100)

D_D_cs_df = pd.read_csv(os.path.join(csv_path, 'D-D_cross_sec.csv'), header=None, names = ['E kV', 'cs Barns'])
D_T_cs_df = pd.read_csv(os.path.join(csv_path, 'D-T_cross_sec.csv'), header=None, names = ['E kV', 'cs Barns'])
D_He_cs_df = pd.read_csv(os.path.join(csv_path, 'D-He_cross_sec.csv'), header=None, names = ['E kV', 'cs Barns'])
p_B_cs_df = pd.read_csv(os.path.join(csv_path, 'p-B_cs.csv'), header = None, names = ['E kV', 'cs Barns'])

D_D_re_df = pd.read_csv(os.path.join(csv_path, 'D-D_reactivity.csv'), header = None, names = ['E kV', 'cs cm'])
D_T_re_df = pd.read_csv(os.path.join(csv_path, 'D-T_reactivity.csv'), header = None, names = ['E kV', 'cs cm'])
D_He_re_df = pd.read_csv(os.path.join(csv_path, 'D-He_reactivity.csv'), header = None, names = ['E kV', 'cs cm'])

D_D_cs_E = np.asarray(D_D_cs_df['E kV'])
D_D_cs = np.asarray(D_D_cs_df['cs Barns'])*1e-28
D_D_cs_interp = interpolate.interp1d(D_D_cs_E, D_D_cs, assume_sorted = True)

D_T_cs_E = np.asarray(D_T_cs_df['E kV'])
D_T_cs = np.asarray(D_T_cs_df['cs Barns'])*1e-28
D_T_cs_interp = scipy.interpolate.interp1d(D_T_cs_E, D_T_cs)

D_He_cs_E = np.asarray(D_He_cs_df['E kV'])
D_He_cs = np.asarray(D_He_cs_df['cs Barns'])*1e-28
D_He_cs_interp = scipy.interpolate.interp1d(D_He_cs_E, D_He_cs)

p_B_cs_E = np.asarray(p_B_cs_df['E kV'])
p_B_cs_E_SI = p_B_cs_E*1000*const.e
p_B_cs = np.asarray(p_B_cs_df['cs Barns'])*1e-28
p_B_cs_interp = scipy.interpolate.interp1d(p_B_cs_E, p_B_cs)
p_B_cs_SI_interp = scipy.interpolate.interp1d(p_B_cs_E_SI, p_B_cs)

D_D_re_TE = np.asarray(D_D_re_df['E kV'])
D_D_re = np.asarray(D_D_re_df['cs cm'])*1e-6
D_D_re_interp = scipy.interpolate.interp1d(D_D_re_TE, D_D_re)


D_T_re_TE = np.asarray(D_T_re_df['E kV'])
D_T_re = np.asarray(D_T_re_df['cs cm'])*1e-6
D_T_re_interp = scipy.interpolate.interp1d(D_T_re_TE, D_T_re)

D_He_re_TE = np.asarray(D_He_re_df['E kV'])
D_He_re = np.asarray(D_He_re_df['cs cm'])*1e-6
D_He_re_interp = scipy.interpolate.interp1d(D_He_re_TE, D_He_re)

p_B_re = []
start_time = timeit.default_timer()
print('\nStarting p-11B reactivity integration.')
i = 0


for T in T_E_axis*1000*const.e/const.k:
    prefactor = np.sqrt(8*const.k*T/(np.pi*M_pB))*(1.0/(const.k*T))**2
    int_func = lambda E: E*p_B_cs_SI_interp(E)*np.exp(-E/(const.k*T)) if E <= p_B_cs_E_SI[-1] else E*p_B_cs[-1]*np.exp(-E/(const.k*T))
    reactivity, err = scipy.integrate.quad(int_func, min(p_B_cs_E_SI), 1e6*const.e, limit = 2000)

    p_B_re.append(prefactor*reactivity)
    i+=1
    if i%int(len(T_E_axis)/10) == 0:
        print('Calculating up to {} temps out of {} took {:.3f} s'.format(i, len(T_E_axis), timeit.default_timer()-start_time))
p_B_re = np.asarray(p_B_re)
print(p_B_re)
fig, ax = plt.subplots(ncols=2, figsize = (8,4))

ax[0].loglog(D_D_cs_E, D_D_cs, 'k-', label = 'D-D')
ax[0].loglog(D_T_cs_E, D_T_cs, 'r--', label = 'D-T')
ax[0].loglog(D_He_cs_E, D_He_cs, 'b:', label = 'D-$^3$He')
ax[0].loglog(p_B_cs_E, p_B_cs, color = 'grey', linestyle = 'dashdot', label = 'p-$^{11}$B')
ax[0].set_xlabel('K$_\mathregular{rel}$ (keV)', fontsize = 12)
ax[0].set_ylabel('$\sigma(v)$ (m$^{2}$)', fontsize = 12)
ax[0].set_title('Common fusion cross-sections', fontsize = 14)
ax[0].legend(loc = 'upper left')
ax[0].set_xlim(1, 1000, emit = False)
ax[0].set_ylim(1e-5*1e-28, 10*1e-28, emit = False)

ax[1].loglog(D_D_re_TE, D_D_re, 'k-', label = 'D-D')
ax[1].loglog(D_T_re_TE, D_T_re, 'r--', label = 'D-T')
ax[1].loglog(D_He_re_TE, D_He_re, 'b:', label = 'D-$^3$He')
ax[1].loglog(T_E_axis, p_B_re, color = 'grey', linestyle = 'dashdot', label = 'p-$^{11}$B')
ax[1].set_xlabel('T$_E$ (keV)', fontsize = 12)
ax[1].set_ylabel('$<\sigma(v)v>$ (m$^3$s$^{-1}$)', fontsize = 12)
ax[1].set_title('Maxwellian plasma fusion reactivity', fontsize = 14)
ax[1].legend(loc = 'upper left')
ax[1].set_xlim(1,1000, emit = False)
ax[1].set_ylim(1e-26, 1e-21, emit = False)

fig.tight_layout()


fig_path = os.path.join(os.getcwd(), 'Final_Figures')
fig_name = 'fusion_cross_secs_and_reactivities.eps'

plt.savefig(os.path.join(fig_path, fig_name), format = 'eps')

plt.show()
plt.close()