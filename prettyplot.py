import matplotlib.pyplot as plt
import numpy as np

"""
Collection of pretty plot functions.
"""
class PrettyPlot(object):
	"""docstring for ."""

	def __init__(self, arg):
		super(, self).__init__()
		self.arg = arg
		return

	"""
	Temporary
	"""
	def fillbetween (self, **kwargs):
		# Error margin as area
		if kwargs.get('eflag', True):
			upper_error = y[:,0] + y[:,2]
			lower_error = y[:,0] - y[:,2]
			lim_idx = (lower_error <= 0)
			lower_error[lim_idx] = 0

			self.plt.plot (x, upper_error, color='blue', dashes=[8,4], alpha=0.25, label='upper error margin')
			self.plt.plot (x, lower_error, color='blue', dashes=[8,4], alpha=0.25, label='lower error margin')
			self.plt.fill_between (x, upper_error, lower_error, facecolor='blue', alpha=0.15, label='error margin')
		return

	"""
	Scatter Plot with Polynomial Trendline
	"""
	def ScatterLinear (self, x, y, **kwargs):
		self.fig, self.ax = self.plt.subplots(figsize=kwargs.get('figsize',(16,8)))

		self.plt.scatter (x, y, color='orange', label=kwargs.get('dataname', 'Data'))

		self.ax.set_xlabel(kwargs.get('xlabel', 'X'))
		self.ax.set_ylabel(kwargs.get('ylabel', 'Y'))
		self.ax.set_title(kwargs.get('title', 'Scatter Plot'))

		# Best Fit Curve
		c = np.polyfit(x, y[:], 3)
		p = np.poly1d (c)
		l = np.linspace (x[0], x[-1], num=100)
		self.plt.plot (l, p(l), '--', color='black', alpha=0.75, label='Best Fit Curve')

		self.ax.legend(loc='upper left')
		self.plt.grid()
		self.plt.show()

		equation = f'{c[0]:.2e}*x^3 {c[1]:+.2e}*x^2 {c[2]:+.2e}*x {c[3]:+.2e}'
		print (f'Approximate Equation for line of best fit is: {equation}')
		print (f'Precise coefficient values: {c[0]} | {c[1]} | {c[2]} | {c[3]}')

		return (self, c)

	"""
	Logarithmic Scatter Plot with Trendline
	"""
	def ScatterLog (self, x, y, **kwargs):
		self.fig, self.ax = self.plt.subplots(figsize=kwargs.get('figsize',(16,8)))

		# Zero checks & converts to log
		# Removes any data point where y[mean] is zero as it breaks this function
		if (np.min(y[:]) == 0):
			zflag = True
			zidx = (y[:] != 0)
			xlog = np.log10(x[zidx])
			ylog = np.log10(y[zidx,0])
		else:
			zflag = False
			xlog = np.log10(x)
			ylog = np.log10(y[:,0])

		self.plt.scatter (xlog, ylog, color='orange', label=kwargs.get('dataname', 'Data'))

		self.ax.set_xlabel(kwargs.get('xlabel', 'X'))
		self.ax.set_ylabel(kwargs.get('ylabel', 'Y'))
		self.ax.set_title(kwargs.get('title', 'Logarithmic Scatter Plot'))

		# Best Fit Curve
		c = np.polyfit(xlog, ylog, 1)
		p = np.poly1d (c)
		l = np.linspace (xlog[0], xlog[-1], num=100)
		self.plt.plot (l, p(l), '--', color='black', alpha=0.75, label='Best Fit Line')

		self.ax.legend(loc='upper left')
		self.plt.grid()
		self.plt.show()

		equation = '{0[0]:.2f}*x {0[1]:+.2f}'.format(c)
		print (f'Approximate Equation for line of best fit is: {equation}')
		print ('Precise coefficient values: {0[0]} | {0[1]}'.format(c))
		if zflag:
			print ('Warning: Data contains zero-values, data partially truncated to removed zero-valued points')

		return (self, c)

	"""
	Linear scatter plot function for comparing multiple datasets

	x = single column vector of x data points
	y = matrix of y data points, each column belonging to one dataset
	datalabel = list of strings, containing the label for each dataset, in same order
	colorlist = list of strings, containing the named color for each dataset, in same order
	"""
	def ScatterLinearCompare (self, x, y, datalabel, colorlist, **kwargs):
		self.fig, self.ax = self.plt.subplots(figsize=kwargs.get('figsize',(16,8)))

		# data set count
		dsc = y.shape[1]

		# y_data[i,:] = [ds1[i], ds2[i], ds3[i], ...]
		for i in range (dsc):
			self.plt.scatter (x, y[:,i], label=datalabel[i], alpha=0.5, color=colorlist[i])

			if kwargs.get('typeflag','None') == 'linear':
				# Best Fit Line
				c = np.polyfit(x, y[:,i], 1)
				p = np.poly1d (c)
				l = np.linspace (x[0], x[-1], num=100)
				self.plt.plot (l, p(l), '--', alpha=0.75, label=datalabel[i], color=colorlist[i])
				equation = f'{c[0]:+.2e}*x {c[1]:+.2e}'
				print (f'Approximate Equation for line of best fit is: {equation}')
				print (f'Precise coefficient values: {c[0]} | {c[1]}')
			else:
				# Best Fit Curve
				c = np.polyfit(x, y[:,i], 3)
				p = np.poly1d (c)
				l = np.linspace (x[0], x[-1], num=100)
				self.plt.plot (l, p(l), '--', alpha=0.75, label=datalabel[i], color=colorlist[i])
				equation = f'{c[0]:+.2e}*x^3 {c[1]:+.2e}*x^2 {c[2]:+.2e}*x {c[3]:+.2e}'
				print (f'Approximate Equation for line of best fit is: {equation}')
				print (f'Precise coefficient values: {c[0]} | {c[1]} | {c[2]} | {c[3]}')

		self.ax.set_xlabel(kwargs.get('xlabel', 'Document Size'))
		self.ax.set_ylabel(kwargs.get('ylabel', 'Average Runtime (ms)'))
		self.ax.set_title (kwargs.get('title', 'Performance Comparison Scatter Plot'))

		self.ax.legend(loc='upper left')
		self.plt.grid()
		self.plt.show()

		return (self, c)

	"""
	Logarithmic scatter plot function for comparing multiple datasets

	x = single column vector of x data points
	y = matrix of y data points, each column belonging to one dataset
	datalabel = list of strings, containing the label for each dataset, in same order
	colorlist = list of strings, containing the named color for each dataset, in same order
	"""
	def ScatterLogCompare (self, x, y, datalabel, colorlist, **kwargs):
		self.fig, self.ax = self.plt.subplots(figsize=kwargs.get('figsize',(16,8)))

		# Zero checks & converts to log
		# Removes any data point where y[mean] is zero as it breaks this function
		if (np.min(y) == 0):
			zflag = True
			y[y == 0] = 0.1
			xlog = np.log10(x)
			ylog = np.log10(y)
		else:
			zflag = False
			xlog = np.log10(x)
			ylog = np.log10(y)

		# data set count
		dsc = y.shape[1]

		# List of equations to output later
		equations = []

		# y_data[i,:] = [ds1[i], ds2[i], ds3[i], ...]
		for i in range (dsc):
			self.plt.scatter (xlog, ylog[:,i], label=datalabel[i], alpha=0.5, color=colorlist[i])

			# Best Fit Curve
			c = np.polyfit(xlog, ylog[:,i], 1)
			p = np.poly1d (c)
			l = np.linspace (xlog[0], xlog[-1], num=100)
			self.plt.plot (l, p(l), '--', alpha=0.75, label=datalabel[i], color=colorlist[i])
			equations.append ('{0[0]:.3f}*x {0[1]:+.3f}'.format(c))

		self.ax.set_xlabel(kwargs.get('xlabel', 'Log10 of Document Size'))
		self.ax.set_ylabel(kwargs.get('ylabel', 'Log10 of Average Runtime (ms)'))
		self.ax.set_title(kwargs.get('title', 'Comparison Logarithmic Plot'))

		self.ax.legend(loc='upper left')
		self.plt.grid()
		self.plt.show()

		if zflag:
			print ('Warning: Data contains zero-values, data partially truncated to removed zero-valued points\n')

		print (f'Equations for {dsc} best fit lines:')
		for i, e in enumerate(equations):
			print (f'\t {datalabel[i]} \t= {e}')

		return (self, c)

	"""
	3D plotter for mapper-reducer processes

	Data: x,y,z are all vectors of the same length.
	This function handles reshaping them automatically.

	xstep and ystep is simply xtick and ytick values.
	"""
	def LandscapePlot (self, x, y, z, xstep, ystep, **kwargs):
		self.fig = kwargs.get( 'fig', self.plt.figure(figsize=kwargs.get('figsize',(12,12))) )
		self.ax = kwargs.get( 'ax', self.fig.add_subplot(kwargs.get('subplot',111) , projection='3d') )

		X = np.reshape(x, (xstep.size,ystep.size))
		Y = np.reshape(y, (xstep.size,ystep.size))
		Z = np.reshape(z, (xstep.size,ystep.size))

		# Plot Surface
		cmap = kwargs.get('cmap',cm.jet)
		surf = self.ax.plot_surface(X, Y, Z,
							   rstride=1, cstride=1, cmap=cmap, zorder=1, alpha=kwargs.get('alpha',1),
							   linewidth=1, antialiased=True, shade=True)
		cticks = np.arange(np.floor(np.min(z)), np.ceil(np.max(z)), kwargs.get('ctickstep',10) )
		cbar = self.fig.colorbar(surf, shrink=kwargs.get('colorbar_shrink',1), aspect=12, ticks=cticks, extend='both')
		cbar.set_label('Avg time (ms)\n', rotation=270, fontsize=12, labelpad=20)

		self.ax.set_xlabel( kwargs.get('xlabel','Map Processes (X)'), fontsize=kwargs.get('xlabel_fontsize',10) )
		self.ax.set_ylabel( kwargs.get('ylabel','Reduce Processes (Y)'), fontsize=kwargs.get('ylabel_fontsize',10) )
		self.ax.set_zlabel( kwargs.get('zlabel','Avg Time (ms)'), fontsize=kwargs.get('zlabel_fontsize',10) )
		self.ax.set_title( kwargs.get('title','Map Reduce'), fontsize=kwargs.get('title_fontsize',10))
		self.ax.view_init( kwargs.get('view_v',60), 180+kwargs.get('view_h',60))

		self.plt.xticks(np.arange(1,xstep.size+1))
		self.plt.yticks(np.arange(1,ystep.size+1))

		# Find Minimum
		xmin = x[np.argmin(z)]
		ymin = y[np.argmin(z)]
		zmin = z[np.argmin(z)]
		minLabel = "Minimum: ({0:d}, {1:d}, {2:.2f})".format(xmin, ymin, zmin)
		minPoint = self.ax.scatter(xmin,ymin,zmin, label=minLabel, color='green', s=50, zorder=10)

		# Find Maximum
		xmax = x[np.argmax(z)]
		ymax = y[np.argmax(z)]
		zmax = z[np.argmax(z)]
		maxLabel = "Maximum: ({0:d}, {1:d}, {2:.2f})".format(xmax, ymax, zmax)
		maxPoint = self.ax.scatter(xmax,ymax,zmax, label=maxLabel, color='fuchsia', s=50, zorder=10)

		self.ax.legend(loc='lower left')

		return (self)

	"""
	fig, ax = violin_multiplot (int[c][y][x], int, int, int[y][x])

	Plots multiple violin plots.
	Data array for this function is in the format: [PLOT_NUMBER][Y-AXIS][X-AXIS]
	PLOT_NUMBER is which plot to draw in; raveled like: [[1,2],[3,4]] -> [1,2,3,4]
	Y-AXIS is the sample size; all the provided datapoints are used to draw one "violin"
	X-AXIS is the "cohort"; len(X-AXIS) > 1 would draw multiple "violins" on a single plot
	Made this way, due to how self.plt.violinplot() works...

	-  input : int[][][] : 3D array of data for the violin plot
	-  input : int : Number of rows in plot | works with value > 1
	-  input : int : Number of columns in plot | works with value > 1
	-  input : int[][] : Data color for each plot, can only select the entire plot
	- output : matplolib figure
	- output : matplotlib axes
	"""
	def ViolinPlot (self, data, nrows=2, ncols=2, cmap=None):
		fig_height = 4 * nrows
		fig_width = 16

		fig, ax = self.plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width,fig_height))
		self.plt.rcParams.update({'font.size': 10})

		# Color map code: 0, 1, 2, 3
		color_list = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']
		if (cmap == None): cmap = np.zeros((nrows,ncols))

		count = 0
		for i in range(nrows):
			for j in range(ncols):
				# Plots a violin plot for each category
				parts = ax[i,j].violinplot (data[count], showmeans=True, showextrema=True)
				count += 1

				# Sets different color for each plot
				for pc in parts['bodies']:
					pc.set_facecolor(color_list[cmap[i][j]])
					pc.set_edgecolor(color_list[cmap[i][j]])

				ax[i,j].grid()
				ax[i,j].set_ylim(0.5,0.9)

		return (self)
