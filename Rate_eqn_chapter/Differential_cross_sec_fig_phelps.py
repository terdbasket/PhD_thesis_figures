from Functions.General_functions import *

csv_path = os.path.join(os.path.dirname(os.getcwd()), 'Rate_eqn_chapter' ,'CSV_files')
I_exp_df = pd.read_csv(os.path.join(csv_path, 'ArII_ArI_elastic_diff_cross_sec.csv'), header = None, names = ['theta', 'I'])
I_exp_theta = np.asarray(I_exp_df['theta'])
I_exp = np.asarray(I_exp_df['I'])

full_theta = np.linspace(0,180, num=500)
energy_loss = lambda theta: np.cos(theta*np.pi/180)**2

sigma_iso = lambda eps, a: 2e-19/(np.sqrt(eps)*(1+eps)) + 3e-19*eps/((1+eps/3.0)**a)

ratio = sigma_iso(2.7, 2)/sigma_iso(2.7,2.3)

place = find_nearest_index_in_array(I_exp_theta, 90, sorted_array=True)
I_val = I_exp[place]*ratio


I_iso = lambda theta: I_val*np.sin(theta*np.pi/180)

plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots()

plot1 = ax.plot(I_exp_theta, I_exp, color = 'k', linestyle = 'solid', label = 'I$_\mathregular{tot}$')
plot2 = ax.plot(full_theta, I_iso(full_theta), color = 'k', linestyle = '--', label = 'I$_\mathregular{i}$')
ax.set_xlabel('Centre of mass scattering angle - \u03B8 (deg)', fontsize = 12)
ax.set_xlim(0,180, emit = False)
ax.set_ylim(0, 40, emit = False)
ax.set_ylabel('I(\u03B8)sin(\u03B8) ($10^{-20}$ $\mathregular{m}^2/\mathregular{sr}$)\n0.5I$_\mathregular{lab}$(\u03D5)sin(\u03D5)  $\mathregular{m}^2/\mathregular{sr}$)', fontsize = 12)

ax2 = ax.twinx()

plot3 = ax2.plot(full_theta, energy_loss(full_theta/2.0), linestyle = '-.', color = 'k',  label = '\u03B5$_\mathregular{f}$/\u03B5$_\mathregular{i}$')
ax2.set_ylabel('Ion energy ratio', fontsize = 12)
ax2.set_ylim(0,2)
sec_ax = ax2.secondary_xaxis('top', functions = (lambda x: x/2.0, lambda x: 2*x))
sec_ax.set_xlabel('Laboratory scattering angle - \u03D5 (deg)', fontsize = 12)
sec_ax.set_xlim(0, 90, emit = False)

lns = plot1 + plot2 + plot3
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, numpoints = 3, handlelength = 4, markerfirst = False)

figurepath = os.path.join(os.path.dirname(os.getcwd()), 'Rate_eqn_chapter', 'Final_figures')

fig.subplots_adjust(hspace=0.05, top = 0.85, left = 0.15, right = 0.85)
fig.suptitle('Argon ion elastic differential cross sections', y = 1, fontsize = 14)

plt.savefig(figurepath+'/Phelps_ArII_ArI_diff_cross_sec.eps', format = 'eps')
plt.show()
plt.close()