# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
# calculate cardinality of columns in data frame and categorise each variable

def cardinalidad(df_in, umbral_cat, umbral_conti):
    # initial caluclations to keep code clean and imporve readability
    num_uniques = df_in.nunique()
    percentages = (num_uniques / len(df_in)) * 100

    # initialize dataframe
    df_cardinalidad = pd.DataFrame({
        'Card': num_uniques,
        '%_Card': percentages
    })

    # make 'Clasification' column and set all to 'Categorica'
    df_cardinalidad['Clasification'] = 'Categorica'

    # conditions to classify variables
    df_cardinalidad.loc[df_cardinalidad['Card'] == 2, 'Clasification'] = 'Binario'
    df_cardinalidad.loc[(df_cardinalidad['Card'] >= umbral_cat) & (df_cardinalidad['%_Card'] >= umbral_conti), 'Clasification'] = 'Numerica Continua'
    df_cardinalidad.loc[(df_cardinalidad['Card'] >= umbral_cat) & (df_cardinalidad['%_Card'] < umbral_conti), 'Clasification'] = 'Numerica Discreta'

    return df_cardinalidad

# %%
def card_tipo(df,umbral_categoria = 10, umbral_continua = 30):
    # Part 1: Prepare dataset with cardinalities, % variation of cardinality and data types
    df_temp = pd.DataFrame([df.nunique(), df.nunique()/len(df) * 100, df.dtypes]) # Cardinality and percentage variation of cardinality
    df_temp = df_temp.T # Use transpose to convert the the columns into rows
    df_temp = df_temp.rename(columns = {0: "Card", 1: "%_Card", 2: "dType"}) # rename the transposed columns

    # correction for when there is only 1 value
    df_temp.loc[df_temp.Card == 1, "%_Card"] = 0.00

    # create a column to suggest the variable types. Start by considering all variables as categorical and adapt the filters to the dataset at hand
    df_temp["tipo_sugerido"] = "Categorica"
    df_temp.loc[df_temp["Card"] == 2, "tipo_sugerido"] = "Binaria"
    df_temp.loc[df_temp["Card"] >= umbral_categoria, "tipo_sugerido"] = "Numerica discreta"
    df_temp.loc[df_temp["%_Card"] >= umbral_continua, "tipo_sugerido"] = "Numerica continua"

    return df_temp

# %%
def count_grapes(grape_vars):
    return len(grape_vars.split(', '))

# %%
regions_dict = {
    'Andalucia':['Málaga', 'Cádiz', 'Sierras de Málaga','Montilla-Moriles','Ribera del Guadiana','Sierra Norte de Sevilla','Los Palacios','Andalucía','Sanlúcar de Barrameda','Jerez-Xérès-Sherry'],
    'Aragon':['Somontano','Valdejalón','Aragón','Cariñena','Calatayud','Ribera del Gállego-Cinco Villas','Campo de Borja','Aylés','Huesca'],
    'Asturias':['Asturias','Cangas'],
    'Baleares':['Mallorca','Binissalem-Mallorca','Plà i Llevant','Isla de Menorca','Formentera'],
    'Canarias':['Gran Canaria','Valle de la Orotava','Lanzarote','Valle de Güímar','Tacoronte-Acentejo','Ycoden-Daute-Isora','El Hierro','La Palma','Islas Canarias','Tenerife'],
    'Cantabria':['Liébana'],
    'Castilla y Leon':['Ribera del Duero','Castilla y León','Abadía Retuerta','Bierzo','Rueda','Toro','Tierra de León','Tierra del Vino de Zamora','Sierra de Salamanca','Arribes','Arlanza','Cigales','Valladolid','Zamora','Segovia','Sierra de Gredos','Salamanca','Burgos','Peñaranda'],
    'Castilla La Mancha':['Jumilla','La Mancha','Finca Élez','Dehesa del Carrizal','Méntrida','Valdepeñas','Uclés','Manchuela','Pago Calzadilla','Ribera del Júcar','Dominio de Valdepusa','Pago Florentino','Toledo','Ciudad Real','Guadalajara','Albacete','Cuenca'],
    'Catalunya':['Priorat','Empordà','Catalunya','Costers del Segre','Penedès','Terra Alta','Alella','Pla de Bages','Conca de Barberà','Montsant','Tarragona','Clàssic Penedès','Barcelona'],
    'Valencia':['Valencia','Alicante','Utiel-Requena','El Terrerazo','Villena'],
    'Extremadura':['Extremadura','Cáceres'],
    'Galicia':['Valdeorras','Rías Baixas','Ribeiro','Galicia','Ribeira Sacra','Monterrei','Barbanza e Iria','Ourense','Pontevedra','La Coruña','Val do Salnés'],
    'Madrid':['Madrid'],
    'Murcia':['Jumilla','Murcia','Almansa','Yecla','Bullas'],
    'Navarra':['Otazu','Navarra','Arínzano','Ribera del Queiles'],
    'Pais Vasco':['País Vasco','Bizkaiko Txakolina','Getariako Txakolina','Rioja Alavesa','Arabako Txakolina','Álava'],
    'La Rioja':['Rioja','Rioja Alta','Valles de Sadacia','La Rioja']
}

def get_community(region):
    for community, regions in regions_dict.items():
        if region in regions:
            return community
    return 'Unknown'

# %%
def print_val_counts(df, cols, relativa=False):
    for variable in cols:
        print(f"para {variable}")
        if relativa:
            print(df[variable].value_counts()/len(df)*100)
        else:
            print(df[variable].value_counts())
        print('\n'*2)

# %%
def print_counts(df, cols):
    for variable in cols:
        print(f"For {variable}")
        df_temp = pd.DataFrame()
        df_temp['freq count'] = df[variable].value_counts()
        df_temp['rel freq count'] = (df[variable].value_counts()/len(df)*100).round(2)
        
        print(df_temp)
        print('\n'*2)

# %%
def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: (x / total)*100)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# %%
def pinta_distribucion_categoricas_alone(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas + 1) // 2

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: (x / total)*100)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# %%
def get_central_tends(df, cols=None):
    
    if cols is not None:
        df = df[cols]

    df_described = df.describe().T
    df_described.reset_index(names='variable', inplace=True)

    df_described.loc[:,'IQR'] = df_described['75%'] - df_described['25%']
    df_described.loc[:,'range'] = df_described['max'] - df_described['min']
    df_described.loc[:,'CV'] = df_described['std'] / df_described['mean']
    df_described = df_described.round(2)
    
    return df_described

# %%
def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histogram & KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

# %%
def plot_categorical_relationship_fin(df, cat_col1, cat_col2, relative_freq=False, show_values=False, size_group = 5):
    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()

# %%
def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=data_subset.index, y=data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de 5 categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()

# %%
def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

# %%
def plot_categorical_numerical_relationship_together(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    
    # Crea el gráfico
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

    # Añade títulos y etiquetas
    plt.title(f'Relación entre {categorical_col} y {numerical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
    plt.xticks(rotation=45)

    # Mostrar valores en el gráfico
    if show_values:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

    # Muestra el gráfico
    plt.show()

# %%
def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

# %%
def plot_grouped_boxplots(df, cat_col, num_col):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=subset_df)
        plt.title(f'Boxplots of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xticks(rotation=45)
        plt.show()

# %%
def plot_grouped_boxplots_together(df, cat_col, num_col, colour_scheme='Blues'):
    plt.figure(figsize=(12, 8))

    medianprops = {
    'color':'red',
    'linestyle':'--',
    'linewidth':2
    }

    boxprops = {
    'alpha':0.8
    }

    sns.boxplot(x=cat_col,
                y=num_col,
                data=df,
                hue=cat_col,
                legend=False,
                palette=colour_scheme,
                patch_artist=True,
                notch=True,
                medianprops=medianprops,
                boxprops=boxprops)
    

    
    plt.title(f'Boxplots of {num_col} for {cat_col}')
    plt.xticks(rotation=45)
    plt.show()

# %%
def plot_winetype_boxplots(df, cat_col, num_col, whisker_vals=False, offset = 0):
    plt.figure(figsize=(12, 8))
    
    # Create boxplot with specific colors for the first two categories
    unique_categories = df[cat_col].unique()
    boxplot = sns.boxplot(x=cat_col, y=num_col, data=df, hue=cat_col, palette=["palegoldenrod", "red"], legend=False)
    
    # Add median and whisker values
    for i, category in enumerate(unique_categories):
        median = df[df[cat_col] == category][num_col].median()
        boxplot.text(i, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')
        
        if whisker_vals:
            q1 = df[df[cat_col] == category][num_col].quantile(0.25)
            q3 = df[df[cat_col] == category][num_col].quantile(0.75)
            whisker_low = df[df[cat_col] == category][num_col][df[df[cat_col] == category][num_col] >= (q1 - 1.5 * (q3 - q1))].min()
            whisker_high = df[df[cat_col] == category][num_col][df[df[cat_col] == category][num_col] <= (q3 + 1.5 * (q3 - q1))].max()
            
            boxplot.text(i + offset, whisker_low, f'{whisker_low:.2f}', ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')
            boxplot.text(i + offset, whisker_high, f'{whisker_high:.2f}', ha='center', va='top', color='black', fontsize=10, fontweight='bold')
    
    plt.title(f'Boxplots of {num_col} for {cat_col}')
    plt.xticks(rotation=45)
    plt.show()

# %%
def get_whisker_vals(df, cat_col, num_col):
    whisker_data = []
    
    for category in df[cat_col].unique():
        subset = df[df[cat_col] == category][num_col]
        Q1 = subset.quantile(0.25)
        Q3 = subset.quantile(0.75)
        IQR = Q3 - Q1
        lower_whisker = subset[subset >= (Q1 - 1.5 * IQR)].min()
        upper_whisker = subset[subset <= (Q3 + 1.5 * IQR)].max()

        whisker_data.append({
            cat_col: category,
            'Lower Whisker': lower_whisker,
            'Upper Whisker': upper_whisker
        })

    whisker_df = pd.DataFrame(whisker_data)
    return whisker_df

# %%
def plot_grouped_histograms(df, cat_col, num_col, group_size, bins = "auto"):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat), bins = bins)
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

# %%
def grafico_dispersion_con_correlacion(df, columna_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    columna_x (str): Nombre de la columna para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos)

    if mostrar_correlacion:
        correlacion = df[[columna_x, columna_y]].dropna().corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlacion:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.grid(True)
    plt.show()

# %%
def bubble_plot(df, col_x, col_y, col_size, scale = 1000):
    """
    Crea un scatter plot usando dos columnas para los ejes X e Y,
    y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    col_x (str): Nombre de la columna para el eje X.
    col_y (str): Nombre de la columna para el eje Y.
    col_size (str): Nombre de la columna para determinar el tamaño de los puntos.
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[col_size] - df[col_size].min() + 1)/scale

    plt.scatter(df[col_x], df[col_y], s=sizes)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Burbujas de {col_x} vs {col_y} con Tamaño basado en {col_size}')
    plt.show()

# %%
def my_mannwhitney(group1, group2, var_name=None, group_difference='group difference'):
    u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    n1 = len(group1)
    n2 = len(group2)
    
    N = n1 + n2
    
    print(f'P-Value: {p_val:.2e}')
    if p_val < 0.05:
        print("The p-value is less than 0.05, so we reject the null hypothesis.")
        print(f"This indicates that there is a statistically significant difference between {var_name} of the two groups.")
    else:
        print("The p-value is greater than or equal to 0.05, so we fail to reject the null hypothesis.")
        print(f"This indicates that there is no statistically significant difference between {var_name} of the two groups.")
    
    z_score = stats.norm.ppf(1 - p_val / 2)
    
    effect_size_r = z_score / np.sqrt(N)
    
    effect_size_eta_squared = u_stat / (n1 * n2)
    
    print(f'\nZ-Score: {z_score:.2f}')
    print(f'Effect Size r: {effect_size_r:.2f}')
    print(f'Effect Size n²: {effect_size_eta_squared:.2f}')
    
    if effect_size_r < 0.1:
        effect_strength = "small"
    elif effect_size_r < 0.3:
        effect_strength = "medium"
    else:
        effect_strength = "large"
    
    print(f"\nThese results suggest a {effect_strength} effect size and that about {effect_size_eta_squared * 100:.1f}% of the variance is explained by the {group_difference}.")


# %%
def my_anova_testing(df, categoric_vars=None, numeric_vars=None, print_boxplots=False):
    
    significant_vars_dictionary = {key:[] for key in categoric_vars}
    for cat_var in categoric_vars:   
        for num_var in numeric_vars:
            if print_boxplots:
                plot_grouped_boxplots_together(df, cat_var, num_var)

            grouped = df.groupby(cat_var)[num_var]
            grouped_data = []
            for group, vals in grouped:
                group_cleaned = vals.dropna().values
                if len(group_cleaned) > 0:
                    grouped_data.append(group_cleaned)

            if len(grouped_data) < 2:
                raise ValueError(f'Not enough groups for {num_var} across {cat_var}. Consider use my_mannwhitney function instead')

            f_stat, p_val = stats.f_oneway(*grouped_data)
            print(f'ANOVA results for {num_var} across {cat_var}:')
            print('F-statistic:', f_stat)
            print('P-value:', p_val)
            
            if p_val < 0.05:
                print(f"There is a statistically significant difference in {num_var} across different {cat_var}.")
                print('\n')
                significant_vars_dictionary[cat_var].append(num_var)
            else:
                print(f"There is no statistically significant difference in {num_var} across different {cat_var}.")
                print('\n')


    sig_vars_DF = pd.DataFrame({'significant variables': significant_vars_dictionary})
    return sig_vars_DF
    

# %%
def my_dispersion_correlation(df, colx, coly, tamano_puntos=50, mostrar_correlacion=False, show_graph=True):
    
    df_cleaned = df[[colx,coly]].dropna()

    if show_graph:
        grafico_dispersion_con_correlacion(df, colx, coly, tamano_puntos=tamano_puntos, mostrar_correlacion=mostrar_correlacion)
    
    stat, p_val = stats.pearsonr(df_cleaned[colx], df_cleaned[coly])
    
    print(f"Pearson correlation stat: {stat:.4f}")
    print(f"P_value: {p_val:.4f}")

    if np.abs(stat) < 0.1:
        correlation_strength = 'very weak'
    elif np.abs(stat) < 0.3:
        correlation_strength = 'weak'
    elif np.abs(stat) < 0.5:
        correlation_strength = 'moderate'
    elif np.abs(stat) < 0.7:
        correlation_strength = 'strong'
    else:
        correlation_strength = 'very strong'

    if stat < 0:
        direction = 'negative'
    else:
        direction = 'positive'

    if p_val < 0.05 and stat:
        print(f"There is a statistically significant correlation between {colx} and {coly} result")
        print(f"With a correlation stat of <{stat:.4f}>, there is a {correlation_strength} {direction} correlation")
    else:
        print(f"There is NO statistically significant correlation between {colx} and {coly} result")

    print()

# %%
def correlations_by_category(df, colx, coly_list, groupings=None, tamano_puntos=50, mostrar_correlacion=False, show_graph=True):
    
    groups = df.groupby(groupings)

    for var in coly_list:
        for group, vals in groups:
            print(f"Analysing {colx} vs {var} for {groupings}: {group}")

            df_cleaned = vals[[colx, var]].dropna()

            if len(df_cleaned) < 2:
                print(f"VALUE-ERROR: Not enough data points for correlation analysis in group {group}. Skipping this group.")
                print('\n' + '-'*50 + '\n')
                continue

            my_dispersion_correlation(df_cleaned, colx=colx, coly=var, tamano_puntos=tamano_puntos, mostrar_correlacion=mostrar_correlacion, show_graph=show_graph)
            print('\n' + '-'*50 + '\n')

# %%
def correlation_matices_category(df, numeric_vars, groupings, min_samples=2, colour_scheme='YlGn'):
    groups = df.groupby(groupings)

    for group, vals in groups:
        print(f"Heatmap for group: {group}")

        df_cleaned = vals[numeric_vars].dropna()
        print('Sample size:', len(df_cleaned))

        if len(df_cleaned) < min_samples:
            print('Not enough data for correlation analysis. Skipping this group')
            print('\n' + '-'*50 + '\n')
            continue
        
        corr_matrix = df_cleaned.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=colour_scheme, 
                    cbar=True, square=True, linewidths=.5)
        
        plt.title(f'Correlation Matrix for {groupings}: {group}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        plt.show()
        print('\n' + '-'*50 + '\n')


