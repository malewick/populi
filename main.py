

try:
    from functools import lru_cache
except ImportError:
    # Python 2 does stdlib does not have lru_cache so let's just
    # create a dummy decorator to avoid crashing
    print ("WARNING: Cache for this example is available on Python 3 only.")
    def lru_cache():
        def dec(f):
            def _(*args, **kws):
                return f(*args, **kws)
            return _
        return dec





from os.path import dirname, join
from functools import partial

import pandas as pd
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Div
from bokeh.models import ColumnDataSource
from bokeh.models import CustomJS
from bokeh.models import Range1d
from bokeh.models import FuncTickFormatter
from bokeh.models import HoverTool
from bokeh.models import Text
from bokeh.models import LabelSet
from bokeh.models import Label
from bokeh.models.widgets import PreText, Select
from bokeh.plotting import figure

from datetime import date
import datetime






class AgeDistribution :


	def __init__(self,source,dataframe,which_q='',which_sex='both',which_job='both'):
		"""
		Positional arguments:
			1 - source (ColumnDataSource) - neccessary for choosing ranges
			2 - Pandas dataframe
		Key arguments:
			which_sex - 'both' (default), 'female' or 'male'
			which_job - 'both' (default), 'working' or 'unemployed'
		"""

		self.df = dataframe
		if which_sex != 'both':
			self.df = self.df.loc[ self.df['sex'] == which_sex ]
		if which_job != 'both':
			self.df = self.df.loc[ self.df['working'] == which_job ]


		self.age_hist, self.age_edges = np.histogram(self.df["age"],
						   bins=(self.df["age"].max()-self.df['age'].min()),
						   range=(self.df["age"].min(),self.df['age'].max()))

		self.age_zeros = np.zeros(len(self.age_edges)-1)
		self.age_hmax = max(self.age_hist)*1.1
		self.age_xmax = max(self.age_edges)*1.1

		hover = HoverTool(tooltips=[
			("Wiek", "@left lat"),
			("Odpowiedzi", "@top"),
			])

		self.age_selector = figure( plot_width=950, plot_height=430, tools=['xbox_select',hover])

		self.set_plot_properties()

		self.phq = self.age_selector.quad( bottom=0,
						   top   = self.age_hist,
						   right = self.age_edges[1:],
						   left  = self.age_edges[:-1],
						   color="white", line_color="#3A5785",
						   nonselection_color="white", nonselection_line_color="#3A5785",
						   nonselection_line_alpha=0.5,
						   selection_alpha=0.7,
						   selection_line_alpha=1.0,
						   selection_color=TEST_COLOR)

		self.asel = self.age_selector.circle('age', 'zeros', size=0, source=source)

	def set_plot_properties(self):
		"""Sets properties of the age distriubtion plot -- purely graphical piece of code."""
		self.age_selector.xgrid.grid_line_color = None
		self.age_selector.ygrid.grid_line_color = None
		self.age_selector.axis.minor_tick_in = 0
		self.age_selector.axis.minor_tick_out = 2
		self.age_selector.toolbar.logo = None
		self.age_selector.toolbar_location = None

		self.age_selector.xaxis.axis_label = "Wiek"
		self.age_selector.xaxis.axis_line_width = 2
		self.age_selector.xaxis.axis_line_color = "gray"
		self.age_selector.xaxis.axis_label_text_font_size = "16pt"
		self.age_selector.xaxis.axis_label_text_color = "black"
		self.age_selector.xaxis.major_label_text_font_size = "16pt"
		self.age_selector.xaxis.major_label_text_color = "black"
		self.age_selector.x_range = Range1d( self.df["age"].min()-7, 1.1*self.df['age'].max() )
		self.age_selector.xaxis.bounds = ( self.df["age"].min()-1, self.df['age'].max()+1 )

		self.age_selector.yaxis.axis_label = "Liczba uczestników ankiety"
		self.age_selector.yaxis.axis_label_text_font_size = "14pt"
		self.age_selector.yaxis.axis_label_standoff = 25
		self.age_selector.yaxis.axis_line_width = 2
		self.age_selector.yaxis.axis_line_color = "gray"
		self.age_selector.yaxis.axis_label_text_color = "black"
		self.age_selector.yaxis.major_label_text_font_size = "14pt"
		self.age_selector.yaxis.major_label_text_color = "black"
		self.age_selector.y_range = Range1d( -0.01, 1.05*self.age_hmax )
		self.age_selector.yaxis.bounds = ( 0.01, self.age_hmax/1.1 )

	def get_figure(self):
		"""Returns the bokeh figure produced by the class."""
		return self.age_selector





class AnswersDistribution :

	def __init__(self,dataframe,which_q='zeros',which_sex='both',which_job='both'):
		"""
		Positional arguments:
			1 - Pandas dataframe
			2 - string with question tag (column title)
			3 - list of strings with answers to show on plots (instead of num values)
		Key arguments:
			which_sex - 'both' (default), 'female' or 'male'
			which_job - 'both' (default), 'working' or 'unemployed'
		"""

		self.q_keys = Q_KEYS_DICT[which_q]

		self.df = dataframe
		if which_sex != 'both':
			self.df = self.df.loc[ self.df['sex'] == which_sex ]
		if which_job != 'both':
			self.df = self.df.loc[ self.df['working'] == which_job ]

		self.q_hist, self.q_edges = np.histogram(self.df[which_q],
							 bins=self.df[which_q].max()+1,
							 range=(0,self.df[which_q].max()+1),
							 density=True)

		self.q_hist *= 100.
		self.q_zeros = np.zeros(len(self.q_edges)-1)
		self.q_hmax = max(self.q_hist)*1.1
		self.q_xmax = max(self.q_edges)*1.1

		self.q_fig = figure(plot_width=950,plot_height=430,tools="pan,wheel_zoom,box_select,reset",
				y_range=(0, self.q_hmax))

		self.q_fig.quad( bottom=0,
				 top=self.q_hist,
				 right=self.q_edges[1:],
				 left=self.q_edges[:-1], 
				 color="white", line_color="#3A5785" )

		LINE_ARGS = dict(color=TEST_COLOR, line_color=None)
		self.hh1 = self.q_fig.quad(bottom=0,
					   left=self.q_edges[:-1],
					   right=self.q_edges[1:],
					   top=self.q_zeros,
					   alpha=0.7, **LINE_ARGS)

		self.set_plot_properties()
		self.id = datetime.datetime.now()

		formatted_hist = [(' %04.2f%% ' % item) for item in self.q_hist]

		new_labels=[]
		x=[ edge+0.3 for edge in self.q_edges ]
		for i in range(len(self.q_hist)) :
			new_labels.append( Label( x=x[i], y=self.q_hmax*1.0,
					 text=formatted_hist[i],
					 text_color='black', text_alpha=1.0,
					 border_line_color='#3A5785', border_line_alpha=1.0,
					 background_fill_color='white', background_fill_alpha=1.0) )
		for lab in new_labels:
			self.q_fig.add_layout(lab)




	def set_plot_properties(self):
		"""Set graphical properties of the answers distribution plot (bokeh figure)"""
		ticks = [ edge+0.5 for edge in self.q_edges[:-1] ]
		self.q_fig.xaxis.ticker = ticks
		label_dict = {}
		for t, k in zip(ticks,self.q_keys):
			label_dict[t] = k
		self.q_fig.xaxis.formatter = FuncTickFormatter(code="""
					var labels = %s;
					return labels[tick];
					""" % label_dict)


		self.q_fig.xaxis.axis_line_width = 2
		self.q_fig.xaxis.axis_line_color = "gray"
		self.q_fig.xaxis.major_label_text_font_size = "16pt"
		self.q_fig.xaxis.major_label_text_color = "black"
		self.q_fig.x_range = Range1d( ticks[0]-1.0, ticks[-1]+1.0 )
		self.q_fig.xaxis.bounds = ( ticks[0]-0.6, ticks[-1]+0.6 )

		self.q_fig.yaxis.axis_label = "Procent odpowiedzi (%)"
		self.q_fig.yaxis.axis_label_text_font_size = "14pt"
		self.q_fig.yaxis.axis_label_standoff = 25
		self.q_fig.yaxis.axis_line_width = 2
		self.q_fig.yaxis.axis_line_color = "gray"
		self.q_fig.yaxis.axis_label_text_color = "black"
		self.q_fig.yaxis.major_label_text_font_size = "14pt"
		self.q_fig.yaxis.major_label_text_color = "black"
		self.q_fig.y_range = Range1d( -0.01, 1.1*self.q_hmax )
		self.q_fig.yaxis.bounds = ( 0, self.q_hmax/1.1 )


	def get_figure(self):
		"""Returns the bokeh figure produced by the class."""
		return self.q_fig






def get_figure(plot_instance):
	fig = plot_instance.get_figure()
	return fig




class PlotInitAndLayout:

	def update(self, attr, old, new):
		which_q = Q_DICT[self.ticker1.value]
		which_sex = SEX_DICT[self.ticker2.value]
		inds = np.array(new['1d']['indices'])
		if len(inds) == 0:
			hhist1 = self.q1.q_zeros
		else:
			arr = self.q1.df[which_q][inds]
			arr = arr[np.isfinite(arr)]
			hhist1, _ = np.histogram(arr, bins=self.q1.q_edges, density=True)
			hhist1 *= 100

		self.q1.hh1.data_source.data["top"]   =  hhist1

		formatted_hist = [(' %05.2f%% ' % item) for item in hhist1]

		new_labels = []

		x=[ edge+0.3 for edge in self.q1.q_edges ]
		for i in range(len(hhist1)) :
			new_labels.append( Label( x=x[i], y=self.q1.q_hmax*0.94,
					 text=formatted_hist[i],
					 text_color='white', text_alpha=1.0,
					 background_fill_color=TEST_COLOR, background_fill_alpha=1.0) )
		for lab in new_labels:
			self.q1.q_fig.add_layout(lab)

	def __init__(self, pandas_dataframe) :

		# --- tickers business --- 
		def generic_ticker_change(source, dataframe, **kwargs):
			self.q1 = AnswersDistribution(dataframe, **kwargs)
			layout.children[1] = get_figure(self.q1)
			self.ad = AgeDistribution(source,dataframe, **kwargs)
			layout.children[2] = get_figure(self.ad)

		# --- declaring widgets --- 
		which_q="pyt_1"
		self.ticker1 = Select(value='Pytanie 1', options=Q_TICKERS, title='Pytanie:')
		def ticker1_change(attrname, old, new):
			wq = Q_DICT[self.ticker1.value]
			generic_ticker_change(source,pandas_dataframe,which_q=wq)
		self.ticker1.on_change('value', ticker1_change)

		self.ticker2 = Select(value='both', options=SEX_TICKERS, title='Płeć')
		def ticker2_change(attrname, old, new):
			wq = Q_DICT[self.ticker1.value]
			ws = SEX_DICT[self.ticker2.value]
			wj = JOB_DICT[self.ticker3.value]
			generic_ticker_change(source,pandas_dataframe,which_q=wq,which_sex=ws,which_job=wj)
		self.ticker2.on_change('value', ticker2_change)

		self.ticker3 = Select(value='both', options=JOB_TICKERS, title='Zatrudnienie')
		def ticker3_change(attrname, old, new):
			wq = Q_DICT[self.ticker1.value]
			ws = SEX_DICT[self.ticker2.value]
			wj = JOB_DICT[self.ticker3.value]
			generic_ticker_change(source,pandas_dataframe,which_q=wq,which_sex=ws,which_job=wj)
		self.ticker3.on_change('value', ticker3_change)



		source = ColumnDataSource()

		source.data = source.from_df(pandas_dataframe[['date_of_birth', 'pyt_1', 'pyt_2', 'sex', 'age', 'zeros']])


		# --- age distribution and selection ---
		self.ad = AgeDistribution(source,pandas_dataframe)

		# --- connect selection signal --- 
		self.ad.asel.data_source.on_change('selected', self.update)

		# --- question nr 1 answers distribution ---  
		self.q1 = AnswersDistribution(pandas_dataframe,which_q)

		# --- set up layout --- 
		layout = column(row(self.ticker1,self.ticker2,self.ticker3),
				get_figure(self.q1),
				get_figure(self.ad))

		# --- bokeh server ---    <- for standalone use only
		curdoc().add_root(layout)
		curdoc().title = "Populi Dataviz"





# --- global functions --- 

def calculate_age(born):
	today = date.today()
	return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

@lru_cache()
def load_data(filename):
	dateparse = lambda x: pd.datetime.strptime(str(x), '%Y-%m-%d') if str(x)!='nan' else None
	data = pd.read_csv(filename, delimiter=';', header=0,
			   na_values=['None'],
			   parse_dates=['date_of_birth'], date_parser=dateparse )
	return data


def process_data(dataframe):
	pd.set_option('display.width', 1000)
	pd.set_option('max_rows',100)
	data=dataframe.dropna(subset=['sex'])
	data['age'] = pd.Series([calculate_age(item) for item in data["date_of_birth"]], index=data.index)
	data['zeros'] = pd.Series([0 for item in data["date_of_birth"]], index=data.index)
	return data





#		###############
#		# --- MAIN ---#
#		###############


# --- Global variables that (probably) won't be changing for different surveys
TEST_COLOR="#004770"
SEX_TICKERS = ['both','female','male']
JOB_TICKERS = ['both','working', 'unemployed']

SEX_TAGS = ['both', 'f', 'm'] 
JOB_TAGS =  ['both', True, False]
# ----------------------------------------------------------------------------



# --- Global variables that require changing for different surveys:

# - tags of the column in the dataframe
Q_TAGS = ['pyt_1','pyt_2','pyt_3']
#Q_TAGS = ['pyt_1','pyt_2']

# - tickers to be shown in the web applet 
Q_TICKERS = ['Pytanie 1','Pytanie 2','Pytanie 3']
#Q_TICKERS = ['Pytanie 1','Pytanie 2']

# - the list of answers for each of the questions present in the survey
ANSWERS_LISTS = [ ["Nie","Trochę","Tak","No kurwa!"],
		  ["key1","key2","key3"],
		  ['a','b','c','d','e','f','g','h','i','j']  ]
#ANSWERS_LISTS = [ ["Nie","Trochę","Tak","No kurwa!"],
#		  ["key1","key2","key3"] ]

# ----------------------------------------------------------------------------


# --- Helper dictionaries to convert between values in tickers and column names in dataframe
Q_KEYS_DICT = dict(zip( Q_TAGS, ANSWERS_LISTS ))
Q_DICT = dict(zip( Q_TICKERS, Q_TAGS )) 
SEX_DICT = dict(zip( SEX_TICKERS, SEX_TAGS )) 
JOB_DICT = dict(zip( JOB_TICKERS, JOB_TAGS )) 

# 	       Orange, Light Sky Blue, Sky Blue, Steel Blue, Steel Blue
POPULI_COLORS=["#E29C04","#84D4F9","#69C0E9","#448BA4","#4F9EC3"]

# --- loading semi-raw data --- 
DATA_FILENAME = "/home/maciej/cern/populi/new/survey_3.csv"
#DATA_FILENAME = "/home/maciej/cern/populi/new/survey_1.csv"
raw_dataframe = load_data(DATA_FILENAME)
print(list(raw_dataframe.columns.values))


processed_dataframe = process_data(raw_dataframe)
print(list(processed_dataframe.columns.values))

plot = PlotInitAndLayout(processed_dataframe)
