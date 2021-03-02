import numpy as np
import pandas as pd
import streamlit as st
import bokeh
from datetime import date, timedelta
from numpy import interp
from PIL import Image
import pydeck as pdk

import networkx as nx

# from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral11, Category20c, viridis, Spectral6
from bokeh.transform import linear_cmap
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges
from networkx.algorithms import community
from bokeh.transform import factor_cmap

st.set_page_config(layout="wide")
st.title('The BARMM Procurement Network')


image = Image.open('barmm_logo.png')
st.sidebar.image(image, caption='', use_column_width=True)


@st.cache(allow_output_mutation=True)
def get_data():
    df = pd.read_csv('final_df.csv', index_col=0)
    return df

add_selectbox = st.sidebar.radio(
    "",
    ("Context", "Data Set", "Methodology",
     "Exploratory Data Analysis",
     "Network Analysis", "References","Author")
)

if add_selectbox == 'Context':
    st.subheader('Problem Statement')
    st.write('-----------------------------------------------------------------------')
    st.write('Processing and analyzing procurement data is one method to characterize \
    government transactions, identify gaps, gatekeep from anomalies and aid in regulations.')
    st.write('The move of the Philippine government to open procurement data is a huge \
    step for transparency and for the public to be involved in citizen data science actively \
    promoting social responsibilities.')
    st.write(' This project will explore PhilGEPS data in conjunction with data proxies \
    through BARMM open datasets and use network analysis to visualize and understand the \
    dynamics of the procurement system in BARMM.')
    st.markdown("This project also aims to **explore if network analysis methods are good techniques\
    to analyize Philippine procurement data.**", unsafe_allow_html=True)

    # st.write('Understanding Procurement Dynamics in BARMM Through Network Analysis')

    # df1 = (
    # "https://raw.githubusercontent.com/uber-common/"
    # "deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv"
    # )


elif add_selectbox == 'Data Set':
    # st.subheader('Data Set')
    st.write('There are two main sources of data.')
    st.write('(1) PhiGEPS Procurement Data')
    image1 = Image.open('philgeps.png').convert('RGB')
    st.image(image1, caption='')
    st.write('Here is a sample snapshot of the dataset from PhilGEPS')
    df = get_data()
    st.table(df.head(5))
    st.write('(2) BARMM Geo Data Set to locate distribution of companies')

elif add_selectbox == 'Methodology':
    st.subheader('Methodology')
    st.write('From the dataset, we explore two actors.')
    st.markdown("(1) Government Agency initiating the project.", unsafe_allow_html=True)
    st.markdown("(2) Private companies taking the bid.", unsafe_allow_html=True)
    st.markdown("We apply network analysis techniques to visualize and \
    characterize these two entities as *nodes* and their relations (frequency of \
    interaction and total fund received) as *edges*.", unsafe_allow_html=True)

    st.subheader('Preprocessing')
    st.markdown("(1) Null and duplicates values were removed.", unsafe_allow_html=True)
    st.markdown("(2) DataFrame was filtered to extract **Organization Name** \
    as **Target**, **Awardee Corporate Title** as **Source** and **Price** as \
    **Weight**.", unsafe_allow_html=True)

    st.subheader('Network Construction')
    st.markdown("Here is how the network was constructed in Python using Networkx:", unsafe_allow_html=True)


    code1 = '''
    G = nx.from_pandas_edgelist(net_3, 'Source', 'Target', ['Weight','Price'])
    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)

    number_to_adjust_by = 5
    adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(G)])
    nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)

    weighted_degrees = dict(nx.degree(G, weight='Weight'))
    nx.set_node_attributes(G, name='weighted_degree', values=weighted_degrees)

    price = dict(nx.degree(G, weight='Price'))
    nx.set_node_attributes(G, name='price', values=price)
    '''
    st.code(code1, language='python')



elif add_selectbox == 'Exploratory Data Analysis':
    st.subheader('Cummulative Propject Costs Across Time')
    df1 = get_data()

    year = st.multiselect('Year:',
                             ('All', 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011,
                              2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002),
                              [2020])
    if len(year) == 0:
        df1 = df1
    elif year[0] == 'All':
        df1 = df1
    else:
        df1 = df1[df1['year'].isin(year)]

    df1 = df1.replace('nan', np.nan)
    df1 = df1.replace(' NULL ', np.nan)
    df1['date'] = pd.to_datetime(df1['Award Date'], format='%m-%d-%Y')

    # df['date'] = df['Award Date'].dt.date#.dt.to_period('b')
    df1[' Contract Amount '] = [str(i) for i in df1[' Contract Amount ']]
    df1[' Contract Amount '] = [i.replace(' ','') for i in df1[' Contract Amount ']]
    df1[' Contract Amount '] = [i.replace(',','') for i in df1[' Contract Amount ']]
    df1[' Contract Amount '] = [float(i) for i in df1[' Contract Amount ']]

    df1 = df1[~(df1.date.isnull())]

    dataframe= ColumnDataSource(df1)
    plot = figure(x_axis_label='Date',
              y_axis_label='Project Cost',
              plot_width=900,
              plot_height=600,
              x_axis_type='datetime')

    plot.circle(
    'date',
    ' Contract Amount ',
    size=10,#'bins',
    fill_color=factor_cmap('Classification', palette=Spectral6, factors=df1['Classification'].unique()),
    source=dataframe,
    legend_field="Classification",
    line_color='black',
    alpha=0.5)

    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.legend.location = "top_left"

    st.bokeh_chart(plot)

    st.subheader('Degree of the Network')
    st.markdown("Who has the most number connections in the network (unweighted)?", unsafe_allow_html=True)

    code2 = '''
    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)
    '''
    st.code(code2, language='python')

    image2 = Image.open('degree.png').convert('RGB')
    st.image(image2, caption='')

    st.markdown("Who has the most number connections in the network (weighted)?", unsafe_allow_html=True)

    code3 = '''
    weighted_degrees = dict(nx.degree(G, weight='Weight'))
    nx.set_node_attributes(G, name='weighted_degree', values=weighted_degrees)
    '''
    st.code(code3, language='python')

    image3 = Image.open('weighted.png').convert('RGB')
    st.image(image3, caption='')

    st.markdown("Who connects others in the network?", unsafe_allow_html=True)

    code4 = '''
    betweenness_centrality = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, name='betweenness', values=betweenness_centrality)
    '''
    st.code(code4, language='python')

    image4 = Image.open('betweenness.png').convert('RGB')
    st.image(image4, caption='')

    st.markdown("What distinct communities formed in the network?", unsafe_allow_html=True)

    code5 = '''
    communities = community.greedy_modularity_communities(G)
    '''
    st.code(code5, language='python')

    image5 = Image.open('community.png').convert('RGB')
    st.image(image5, caption='')

    st.subheader('Distribution of Companies in BARMM')
    df1 = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

    # basilan = [6.4296, 121.9870,1]
    # lanao_del_sur = [7.8232, 124.4198,2]
    # magindanao = [6.9423, 124.4198,3]
    # sulu = [5.9749, 121.0335,4]
    # tawi_tawi = [5.1338, 119.9509,5]


    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-122.4,
        zoom=11,
        pitch=50,
        ),
        layers=[
            pdk.Layer(
            'HexagonLayer',
            data=df1,
            get_position='[lon, lat]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            auto_highlight=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df1,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
                ),
                ],
        ))

    # def map(data, lat, lon, zoom):
    #     st.write(pdk.Deck(
    #         map_style="mapbox://styles/mapbox/light-v9",
    #         initial_view_state={
    #             "latitude": lat,
    #             "longitude": lon,
    #             "zoom": zoom,
    #             "pitch": 50,
    #             },
    #         layers=[
    #             pdk.Layer(
    #                 "HexagonLayer",
    #                 data=data,
    #                 get_position=["lng", "lat"],
    #                 radius=100,
    #                 elevation_scale=4,
    #                 elevation_range=[0, 1000],
    #                 pickable=True,
    #                 extruded=True,
    #                 ),
    #                 ]
    #     ))
    #
    # basilan = [6.4296, 121.9870,1]
    # lanao_del_sur = [7.8232, 124.4198,2]
    # magindanao = [6.9423, 124.4198,3]
    # sulu = [5.9749, 121.0335,4]
    # tawi_tawi = [5.1338, 119.9509,5]
    #
    #
    # multiple_lists = [basilan, lanao_del_sur, magindanao, sulu, tawi_tawi]
    # arrays = [np.array(x) for x in multiple_lists]
    # barmm_map = [np.mean(k) for k in zip(*arrays)]
    #
    # df = pd.read_csv('heatmap-data.csv')
    #
    # # df = pd.DataFrame(multiple_lists, columns=['lat', 'lon', 'date'])
    # map(df1, -1.415,52.2323, 7)


elif add_selectbox == 'Network Analysis':
    df = get_data()

    # my_expander = st.beta_expander("Search and Filters", expanded=False)
    # with my_expander:
    #     col1, col2 = st.beta_columns(2)
    year = st.multiselect('Year:',
                             ('All', 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011,
                              2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002),
                              [2020])
    if len(year) == 0:
        df = df
    elif year[0] == 'All':
        df = df
    else:
        df = df[df['year'].isin(year)]

    net = df[['Organization Name', 'Awardee Corporate Title', ' Contract Amount ']]
    net.columns = ['Target', 'Source', 'Price']
    net = net.replace('nan', np.nan)
    net = net.replace(' NULL ', np.nan)
    net.dropna(inplace=True)
    net['Price'] = [float(i.replace(',','')) for i in net['Price']]

    net_1 = net.groupby(['Target', 'Source']).size().reset_index().\
    rename(columns={0:'Weight'})

    net_2 = net.groupby(['Target', 'Source']).sum().reset_index().\
    rename(columns={0:'Price'})

    net_3 = pd.merge(net_1, net_2)

    G = nx.from_pandas_edgelist(net_3, 'Source', 'Target', ['Weight','Price'])

    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)

    number_to_adjust_by = 5
    adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(G)])
    nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)

    weighted_degrees = dict(nx.degree(G, weight='Weight'))
    nx.set_node_attributes(G, name='weighted_degree', values=weighted_degrees)

    price = dict(nx.degree(G, weight='Price'))
    nx.set_node_attributes(G, name='price', values=price)

    emty = {}
    for i in range(len(net_3)):
        emty[(net_3['Target'][i],net_3['Source'][i])] = (net_3['Price'][i]/sum(net_3['Price']))*1000

    nx.set_edge_attributes(G, name='priced_degree', values=emty)

    betweenness_centrality = nx.betweenness_centrality(G)

    nx.set_node_attributes(G, name='betweenness', values=betweenness_centrality)

    communities = community.greedy_modularity_communities(G)
    communities = communities[:19]

    # Create empty dictionary
    modularity_class = {}
    modularity_color = {}
    #Loop through each community in the network
    for community_number, community in enumerate(communities):
        #For each member of the community, add their community number
        for name in community:
            modularity_class[name] = community_number
            modularity_color[name] = Category20c[20][community_number]

    nx.set_node_attributes(G, modularity_class, 'modularity_class')
    nx.set_node_attributes(G, modularity_color, 'modularity_color')

    #Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'modularity_color'

    node_highlight_color = 'white'
    edge_highlight_color = 'black'
    #Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
    color_palette = Blues8

    #Choose a title!
    # title = 'BARMM Network'

    #Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [
       ("Entity", "@index"),
       ("Degree", "@degree"),
       ("Price", "@price")
       ]

    #Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips = HOVER_TOOLTIPS,
              tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1))

    #Create a network graph object
    network_graph = from_networkx(G, nx.spring_layout, scale=10, center=(0, 0))

    #Set node sizes and colors according to node degree (color as category from attribute)
    network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)
    #Set node highlight colors
    network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
    network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)


    #Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

    #Set edge highlight colors
    network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width="priced_degree")
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width="priced_degree")

    #Highlight nodes and edges
    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()#EdgesAndLinkedNodes()

    plot.renderers.append(network_graph)

    st.bokeh_chart(plot)

elif add_selectbox == 'References':
    st.subheader('References')
    st.markdown("[1] Public Procurement Data in the Philippines and Where to Find It: \
    https://schoolofdata.org/2019/03/06/public-procurement-data-in-the-philippines-and-where-to-find-it/", unsafe_allow_html=True)
    st.markdown("[2] Networks on Maps (with Python): \
    http://www.sociology-hacks.org/?p=67", unsafe_allow_html=True)
    st.markdown("[3] PhilGEPS Open Data: \
    https://www.philgeps.gov.ph/", unsafe_allow_html=True)
    st.markdown("[4] Exploring and Analyzing Network Data with Python: \
    https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python", unsafe_allow_html=True)
    st.markdown("[5] Introduction to Cultural Analytics and Python: \
    https://melaniewalsh.github.io/Intro-Cultural-Analytics", unsafe_allow_html=True)
    st.markdown("[6] Social network analysis as a method in the data journalistic toolkit: \
    http://arno.uvt.nl/show.cgi?fid=133754", unsafe_allow_html=True)
    st.markdown("[7] Characterization of the Chilean Public Procurement Ecosystem Using Social Network Analysis: \
    https://www.researchgate.net/publication/343245057_Characterization_of_the_Chilean_Public_Procurement_Ecosystem_Using_Social_Network_Analysis", unsafe_allow_html=True)
    st.markdown("[8] ANALYZING THE CORRUPTION ONA PROCUREMENT NETWORK USING GRAPH THEORY: \
    http://www.oecd.org/corruption/integrity-forum/academic-papers/Fountoukidis-Dafli-graph-theory.pdf", unsafe_allow_html=True)
    st.markdown("[9] Social network analysis of project procurement in Iranian construction mega projects: \
    https://link.springer.com/article/10.1007/s42107-021-00348-1", unsafe_allow_html=True)

else:
    st.subheader('Author')
    image = Image.open('test.gif').convert('RGB')
    st.image(image, caption='')#, width=300, height=150)
