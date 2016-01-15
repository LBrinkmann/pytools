import matplotlib.colors as colors
import numpy as np
import pandas as pd

from bokeh.plotting import output_notebook, figure, show
#from bokeh.charts import Bar,BoxPlot, output_file, show
from bokeh.models import Range1d

import matplotlib.pyplot as plt


All = slice(None)


class MyLinePlot():
    #nice colors
    tableau20 = [(31, 119, 180), (255, 127, 14),
         (44, 160, 44), (214, 39, 40),
         (148, 103, 189), (140, 86, 75),
         (227, 119, 194), (127, 127, 127),
         (188, 189, 34), (23, 190, 207)]
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    tableau20 = np.array(tableau20)/255.
    tableau20hex = list(map(lambda c : colors.rgb2hex(c),tableau20))


    def __init__(self,title='None',x_label='',y_label='', width = 900,x_inverse=False):
        self.p = figure(title=title,width = 900, x_axis_label=x_label, y_axis_label=y_label)
        self.x_inverse = x_inverse
    def plot(self,x,y,yerr=None):
        s = 1.05
        s2 = 1/s
        if np.isnan(yerr).all():
            self.y_range = np.min(y) * s2, np.max(y) * s
        else:
            self.y_range = np.min(y-yerr) * s2, np.max(y+yerr) * s
        if len(x.shape) == 1:
            self.x_range = np.min(x), np.max(x)
        else:
            self.x_range = np.max(np.min(x,axis=1)), np.min(np.max(x,axis=1))
        self.update_plot(self.x_range,self.y_range)
        self.create_plot(x,y,yerr)

    def add_points(self,x,y):
        if len(x.shape) == 1:
            n = 1
            x = x.reshape([1,x.shape[0]])
            y = y.reshape([1,y.shape[0]])
        else: n = x.shape[0]
        for i in range(n):
            self.p.circle(x[i,:],y[i,:],size=7,color=self.tableau20hex[i], alpha=1)

    def update_plot(self,x_range,y_range):
        if self.x_inverse:
            self.p.x_range = Range1d(x_range[1],x_range[0])
        else:
            self.p.x_range = Range1d(*x_range)
        self.p.y_range = Range1d(*y_range)

    def create_plot(self,x,y,yerr):
        if len(x.shape) == 1:
            n = 1
            x = x.reshape([1,x.shape[0]])
            y = y.reshape([1,y.shape[0]])
            yerr = yerr.reshape([1,yerr.shape[0]])
        else: n = x.shape[0]
        for i in range(n):
            self.p.line(x[i,:],y[i,:],line_color=self.tableau20hex[i], alpha=1, line_width=2)
            if not np.isnan(yerr).all():
                ypluserr = y[i,:] + yerr[i,:]
                yminerr = y[i,:] - yerr[i,:]
                self.p.patches([np.append(x[i,:],list(reversed(x[i,:])))],[np.append(ypluserr,list(reversed(yminerr)))],color=self.tableau20hex[i], alpha=0.2, line_width=0)
    def show(self):
        show(self.p)



class MyPlot():
    #nice colors
    tableau20 = [(31, 119, 180), (255, 127, 14),
         (44, 160, 44), (214, 39, 40),
         (148, 103, 189), (140, 86, 75),
         (227, 119, 194), (127, 127, 127),
         (188, 189, 34), (23, 190, 207)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    tableau20 = np.array(tableau20)/255.
    def __init__(self,dfs,group='None',stack='None',label='None',value='None',title='None',x_label='',y_label='',legend=True):
        #define groups and order data
        self.level = 1
        self.group = group
        self.stack = stack
        self.label = label
        if isinstance(value, str):
            self.value = value
        else:
            self.value = value[0]
            self.values = value
        self.blegend = legend
        if y_label == '':
            y_label = title
        if x_label == '':
            x_label = self.value
        self.title,self.x_label,self.y_label = title,x_label,y_label
        if not group == 'None':
            self.bgroup = True
            self.groups = dfs[group].sort_values().unique()
            self.ngroups = len(self.groups)
            self.level += 1
        else:
            self.ngroups = 1
            self.bgroup = False
        if not stack == 'None':
            self.bstack = True
            self.stacks = dfs[stack].sort_values().unique()
            self.nstacks = len(self.stacks)
            self.level += 1
        else:
            self.nstacks = 1
            self.bstack = False
        self.labels = dfs[label].sort_values().unique()
        self.nlabels = len(self.labels)

        self.mapping = [self.label]
        self.mappings = [self.labels]
        shape = [self.nlabels]
        if self.bgroup:
            shape.insert(0,self.ngroups)
            self.mapping.insert(0,self.group)
            self.mappings.insert(0,self.groups)
        if self.bstack:
            shape.insert(0,self.nstacks)
            self.mapping.insert(0,self.stack)
            self.mappings.insert(0,self.stacks)
        self.shape = tuple(shape)
        self.builder(dfs)

    def builder(self,dfs):
        dfm = self.map_reduce(dfs)
        self.prepare_plot(dfm)
        self.color_plot()
        self.create_plot(self.x_label,self.y_label,self.title)
        self.add_labels(dfs,dfm)
        if self.blegend:
            self.create_legend(dfs)
        self.show()

    def legend_label(self,n,l,v):
        self.legendnames = list(map(lambda x : '%s : %s'%(v,x),l))

    def add_labels(self,dfs,dfm):
        pass

    def map_reduce(self,dfs):
        #accumulate data
        return dfs.fillna(0).groupby(self.mapping).mean()

    def prepare_plot(self,dfm):
    # plot design
        bdensity = 0.8
        bgroupgap = 0.8
        if self.bstack:
            self.maxheight = dfm.reset_index().groupby(self.mapping[1:]).sum().max().loc[self.value]
        else:
            self.maxheight = dfm.max().loc[self.value]
        if self.bgroup:
            bwidth = 1./self.ngroups*bgroupgap*bdensity
        else:
            bwidth = bdensity
        mrange = np.arange(1,self.nlabels+1)
        if self.bgroup:
            grange = (np.arange(0,self.ngroups)-(self.ngroups-1)*0.5)/self.ngroups*bgroupgap
            mrange = np.add.outer(mrange,grange)
        self.left = mrange-bwidth/2
        self.right = mrange+bwidth/2
        if self.bstack:
            self.left = np.outer(self.left,[1]*self.nstacks).reshape(list(reversed(self.shape)))
            self.right = np.outer(self.right,[1]*self.nstacks).reshape(list(reversed(self.shape)))
        if self.level == 1:
            self.index = self.mappings[0]
        else:
            if self.level == 2:
                nindex = [(m0,m1) for m0 in self.mappings[0] for m1 in self.mappings[1]]
            else:
                nindex = [(m0,m1,m2) for m0 in self.mappings[0] for m1 in self.mappings[1] for m2 in self.mappings[2]]
            self.index = pd.MultiIndex.from_tuples(nindex, names=self.mapping)

        level = list(range(0,self.level))
        top = pd.DataFrame(0, index=self.index, columns = [self.value])
        if self.level == 1:
            top.index.name = self.mapping[0]

        if self.bstack:
            top[self.value] = dfm[self.value].groupby(level=level[1:]).cumsum()
            top = top.fillna(0)
            self.top = top.reset_index().set_index(list(reversed(self.mapping))).sort_index()
            bottom = self.top.groupby(level=level[:-1]).shift(1).fillna(0)
        else:
            top[self.value] = dfm.loc[All,self.value]
            top = top.fillna(0)
            self.top = top.reset_index().set_index(list(reversed(self.mapping))).sort_index()
            bottom = pd.DataFrame(0, index=self.index, columns = [self.value])
            if self.level == 1:
                bottom.index.name = self.mapping[0]
        self.bottom = bottom.reset_index().set_index(list(reversed(self.mapping))).sort_index()


    def color_plot(self):
        if self.bstack:
            if self.bgroup:
                self.wlegend = (self.labels[0],All,self.stacks[0])
                self.legend_label(self.ngroups,self.groups,self.group)
                shade_column_name = self.stack
                shade_column_values = self.stacks
                color_column_name = self.group
                color_column_values = self.groups
            else:
                self.wlegend = (self.labels[0],All)
                self.legend_label(self.nstacks,self.stacks,self.stack)
                shade_column_name = 'None'
                color_column_name = self.stack
                color_column_values = self.stacks
        else:
            shade_column_name = 'None'
            if self.bgroup:
                self.wlegend = (self.labels[0],All)
                self.legend_label(self.ngroups,self.groups,self.group)
                color_column_name = self.group
                color_column_values = self.groups
            else:
                self.wlegend = (All)
                self.legend_label(self.nlabels,self.labels,self.label)
                color_column_name = self.label
                color_column_values = self.labels

        color = pd.DataFrame('red', index=self.index, columns = ['color'])
        if self.level == 1:
            color.index.name = self.mapping[0]
        color = color.reset_index()
        for co,ccv in zip(self.tableau20[0:len(color_column_values),:],color_column_values):
            if shade_column_name == 'None':
                color.loc[(color[color_column_name] == ccv),'color'] = colors.rgb2hex(co)
            else:
                for i,scv in enumerate(shade_column_values):
                    shade = (0.8*(len(shade_column_values)-i)/len(shade_column_values)+0.2)
                    color.loc[(color[shade_column_name] == scv) & (color[color_column_name] == ccv),'color'] = \
                    colors.rgb2hex(1-(1-co)*shade)
        color = color.set_index(self.mapping)
        self.color = color.reset_index().set_index(list(reversed(self.mapping))).sort_index()

    def create_plot(self,x_axis_label,y_axis_label,title):
        s = 1.1
        y_range = 0, self.maxheight * s
        self.p = figure(title=title, x_range = list(map(lambda x : '%1.0f'%(x+1),self.labels)), y_range = y_range, width = 900, x_axis_label=x_axis_label, y_axis_label=y_axis_label )
        self.p.quad(top = self.top[self.value].tolist(),
        bottom = self.bottom[self.value].tolist(),
        right = self.right.flatten(),
        left = self.left.flatten(),
        color =  self.color['color'].tolist(),
        line_color="gray")

    def create_legend(self,dfm):
        for c,n in zip(self.color.loc[self.wlegend,'color'].tolist(),self.legendnames):
            self.p.quad(top = 0, bottom = 0, right = 0, left = 0 , color =  c, legend = n)

    def show(self):
        show(self.p)

def plot_churn_samples(X,Y,Y_samples,maxx,majors,title,maxy=0.65,vline_sep=0.2,y_unit='%'):
    tableau20 = [(31, 119, 180), (255, 127, 14),
         (44, 160, 44), (214, 39, 40),
         (148, 103, 189), (140, 86, 75),
         (227, 119, 194), (127, 127, 127),
         (188, 189, 34), (23, 190, 207)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    plt.figure(figsize=(12, 9))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, maxy)
    plt.xlim(0, maxx)

    if y_unit == '%':
        x_multi = 100
    else:
        x_multi = 1
    plt.yticks(np.arange(0, maxy, vline_sep), [str(x*x_multi) + y_unit for x in np.arange(0, maxy, vline_sep)], fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('passed days',fontsize=14)

    for y in np.arange(0, maxy, vline_sep):
        plt.plot(range(0, maxx), [y] * len(range(0, maxx)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

    y_pos = np.array(Y)[:,maxx]-0.03*maxy
    sort = np.argsort(y_pos)
    sort2 = np.argsort(sort)
#    import ipdb; ipdb.set_trace()
    y_pos2 = np.linspace(y_pos[sort][0],y_pos[sort][-1],num=y_pos.shape[0])
    y_pos_m = np.mean(np.array([y_pos,y_pos2[sort2]]),axis=0)

    for rank, column in enumerate(majors):
        for y_s in Y_samples[rank]:
            plt.plot(X[rank],y_s,lw=1, color=tableau20[rank], alpha=0.3)
        plt.plot(X[rank],Y[rank],lw=2.5, color=tableau20[rank])

        plt.text(maxx*1.01, y_pos_m[rank], column, fontsize=14, color=tableau20[rank],horizontalalignment='left')

    plt.text(maxx/2, maxy*0.9, title, fontsize=17, ha="center")

#    plt.text(1966, -8, "", fontsize=10)
    plt.show()
