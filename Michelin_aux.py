import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
import folium
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
import matplotlib.cm as cm
import matplotlib.colors as colors_map
from branca.element import Template, MacroElement

def count_restau(data):
    """For a dataframe, select the city with most of stared restaurants"""
    city = pd.DataFrame.from_dict(Counter(data["city"]), orient='index').reset_index()
    city.columns=["city", "count"]
    return city[city["count"]>1].reset_index(drop = True)

def city_by_star(data, nb_restau = 1):
    """Return dataframe of city with the most of stared restaurants"""
    city = pd.DataFrame.from_dict(Counter(data["city"]), orient='index').reset_index()
    city.columns=["city", "count"]
    return city[city["count"]>nb_restau].reset_index(drop = True)

def fig_best_cities(best_city_one, best_city_two, best_city_three):
    """Create a barplot representing the count of stared restaurants by city."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=best_city_one["city"],
        y=best_city_one["count"],
        name='One star',
        marker_color='#ff5800'
    ))
    fig.add_trace(go.Bar(
        x=best_city_two["city"],
        y=best_city_two["count"],
        name='Two stars',
        marker_color='#fb0014'
    ))
    fig.add_trace(go.Bar(
        x=best_city_three["city"],
        y=best_city_three["count"],
        name='Three stars',
        marker_color='#d30019'
    ))
    fig.update_layout(barmode='stack',  xaxis={'categoryorder':'total descending'}, xaxis_tickangle=-45)
    return fig

def create_map(data, city, zoom=12):
    """Create a map with folium, representing the restaurants location"""
    address = city
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude

    final_map = folium.Map(location=[latitude, longitude], zoom_start=zoom)

    # add markers to map
    for lat, lng, name, star in zip(data['latitude'], data['longitude'], data['name'], data['Star']):
        label = '{}, {} stars'.format(name, star)
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='blue',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7,
            parse_html=False).add_to(final_map)  

    return final_map


def return_most_common_venues(row, num_top_venues):
    """For a row, return a vector with the num_top_venues top venues number."""
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

def neighborhoods_names(num_top_venues):
    """create columns according to number of top venues"""
    indicators = ['st', 'nd', 'rd']
    columns = ['Neighborhood']
    for ind in np.arange(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind+1))
    return columns

def df_top_venues(sf_grouped, num_top_venues):
    """Return df with neighborhoods venues shorted"""
    columns = neighborhoods_names(num_top_venues)
    # create a new dataframe
    neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
    neighborhoods_venues_sorted['Neighborhood'] = sf_grouped['Neighborhood']

    for ind in np.arange(sf_grouped.shape[0]):
        neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(sf_grouped.iloc[ind, :], num_top_venues)
    return neighborhoods_venues_sorted

def dendo(data):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(data)
    plot_dendrogram(model)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
        # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def silhouette(sf_grouped_clustering):
    """Calculate the silhouette score on data, output a dict with all values"""
    silhouette={}
    method = ["ward", "complete", "average", "single"]
    for met in method:
        dict_values = {}
        for k in range(2,11):
            hc = AgglomerativeClustering(n_clusters = k, affinity = 'euclidean', linkage = met)
            y_hc=hc.fit_predict(sf_grouped_clustering)
            dict_values["k"+ str(k)] =  silhouette_score(sf_grouped_clustering, y_hc)
        silhouette[str(met)] = dict_values
    return silhouette

def HC_predict(data, kclusters):
    """Fit the HC on data"""
    hc = AgglomerativeClustering(n_clusters = kclusters, affinity = 'euclidean', linkage ='ward')
    return hc.fit_predict(data)

def Add_predict_vector(data, neighborhoods_venues, prediction_vector):
    """add clustering labels to data"""
    df_merged = neighborhoods_venues
    try:
        df_merged.insert(0, 'Cluster Labels', prediction_vector)
    except:
        df_merged['Cluster Labels'] = prediction_vector
    return data.join(df_merged.set_index('Neighborhood'), on='Neighborhood')


    # merge sf_grouped with sf_data to add latitude/longitude for each neighborhood
    return data.join(neighborhoods_venues.set_index('Neighborhood'), on='Neighborhood')

def create_template(kclusters, rainbow):
    """Generate HTML code for the map legend"""
    template_part1 = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>jQuery UI Draggable - Default functionality</title>
      <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

      <script>
      $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

      </script>
    </head>
    <body>


    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
         border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

    <div class='legend-title'>Legend (draggable!)</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>"""
    template_part2 = """

      </ul>
    </div>
    </div>

    </body>
    </html>

    <style type='text/css'>
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""
        
    for i in range(kclusters):
        template_part1 = template_part1 + """<li><span style='background:"""+ str(rainbow[i]) + """;opacity:0.7;'></span>Cluster """ + str(i) +"""</li>"""
    return template_part1 + template_part2
    

def create_map_clusters(data, city, kclusters, zoom=12, rainbow = None):
    """Create a map with folium, representing the restaurants location and cluster"""
    if rainbow is None:
        x = np.arange(kclusters)
        ys = [i + x + (i*x)**2 for i in range(kclusters)]
        colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
        rainbow = [colors_map.rgb2hex(i) for i in colors_array]
        
    template = create_template(kclusters, rainbow)
    address = city
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude

    map_clusters = folium.Map(location=[latitude, longitude], zoom_start=zoom)

    # set color scheme for the clusters


    # add markers to the map
    markers_colors = []
    for lat, lon, name, poi, cluster in zip(data['latitude'], data['longitude'], data["name"], data['Neighborhood'], data['Cluster Labels']):
        label = folium.Popup(str(name) + " at " + str(poi), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            popup=label,
            color=rainbow[cluster],
            fill=True,
            fill_color=rainbow[cluster],
            fill_opacity=0.7).add_to(map_clusters)  

    macro = MacroElement()
    macro._template = Template(template)

    map_clusters.get_root().add_child(macro)

    return map_clusters

def silhouette_out(met, data, k):    
    silhouette=[]
    hc_NY = AgglomerativeClustering(n_clusters = k, affinity = 'euclidean', linkage = met)
    y_hc=hc_NY.fit_predict(data)
    silhouette.append(silhouette_score(data, y_hc))
    return silhouette[0]    

def silhouette_graph(data, kvals):
    method = ["ward", "complete", "average", "single"]
    for met in method:
        silhouette_scores = []
        for k in kvals:
            silhouette_scores.append(silhouette_out(met, data, k))   
        plt.bar(kvals, silhouette_scores) 
        plt.xlabel('Number of clusters ' + met) 
        plt.ylabel('S(i)') 
        plt.show() 
        
def best_cities(data):
    """Find all cities with the most of stared restaurants"""
    return [city_by_star(data[data['Star'] == 1], 12),
    city_by_star(data[data['Star'] == 2]),
    city_by_star(data[data['Star'] == 3])]