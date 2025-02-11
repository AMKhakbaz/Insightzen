import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, t
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import plotly.figure_factory as ff
from huggingface_hub import InferenceClient
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import math
from transformers import pipeline

classifier = pipeline('zero-shot-classification', model='MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33')

def sorting(df):
    df.index = list(map(float, df.index))
    df = df.sort_index

    return df

def edit_strings(string_list):
  edited_list = []
  for string in string_list:
    if "_" in string:
      last_underscore_index = string.rfind("_")
      edited_string = string[:last_underscore_index]
      edited_list.append(edited_string)
    else:
      edited_list.append(string)
  return edited_list

def equalize_list_lengths(input_dict):
  max_len = 0
  for key in input_dict:
    max_len = max(max_len, len(input_dict[key]))
  
  for key in input_dict:
    while len(input_dict[key]) < max_len:
      input_dict[key].append(None)
      
  return pd.DataFrame(input_dict)

def figo(plot_type, df, title, xlabel=None, ylabel=None, legend_title=None, colorscale='Plotly3', width=800, height=600):
    if plot_type == "Scatter":
        fig = go.Figure()

        for column in df.columns[0:]:
            df.index = list(map(float, list(df.index)))
            sorted_data = df.sort_index()
            fig.add_trace(go.Scatter(
                x=sorted_data[column],
                y=sorted_data.index,
                mode='lines+markers+text',
                name=column,
                text=sorted_data[column].round(2),
                textposition="middle right"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Percentage",
            yaxis_title="Category",
            yaxis={'categoryorder': 'array', 'categoryarray': sorted_data.index},
            width=width,
            height=height
        )
    
    elif plot_type == "Heatmap":
        df = df.apply(pd.to_numeric, errors='coerce')

        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            hoverongaps=False,
            colorscale=colorscale
        ))

        fig.update_layout(
            title={
                'text': title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend_title=legend_title,
            template="plotly_white",
            width=width,
            height=height
        )

    elif plot_type == "Bar":
        fig = go.Figure()
        col = df.name
        fig.add_trace(go.Bar(
            x=df.index,
            y=df,
            name=col
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend_title=legend_title,
            template="plotly_white",
            barmode='group',
            width=width,
            height=height
        )

    else:
        raise ValueError("Invalid plot_type. Supported types are 'Scatter', 'Heatmap', and 'Bar'.")

    return fig

def is_matching_pattern(column, prefix):
    if not column.startswith(prefix + '_'):
        return False
    suffix = column[len(prefix) + 1:]
    if 1 <= len(suffix) <= 3 and suffix.isdigit():
        return True
    return False

def multi_answer(df):
    friquency = {}
    for i in df.columns:
        try:
            unique_values = list(set(df[i].dropna()))[0]
            friquency[str(unique_values)] = df[i].value_counts().get(unique_values, 0)
        except Exception as e:
            #st.error(f"Warning: One of the data columns has no value.: {e}")
            friquency[i] = 0
            

    friquency_dataframe = pd.DataFrame({"Value": friquency.keys(), 'Frequency': friquency.values(), "Percentage": np.array(list(friquency.values()))/len(df.dropna(how='all'))}).sort_values(by='Value')
    friquency_dataframe.loc[len(friquency_dataframe)] = ['Sample_size', len(df.dropna(how='all')), 1]
    return friquency_dataframe
    
def single_answer(df):
    counter = df.value_counts()
    friquency_dataframe = pd.DataFrame({
        'Value': counter.index, 
        'Frequency': counter.values, 
        'Percentage': (counter.values / counter.sum())}).sort_values(by='Value')
    friquency_dataframe.loc[len(friquency_dataframe)] = ['Sample_size', len(df.dropna()), 1]
    return friquency_dataframe

def score_answer(df):
    counter = df.value_counts().sort_index()

    friquency_dataframe = pd.DataFrame({
        'Value': list(counter.index)+["Meen", "Variance"],
        'Frequency': list(counter.values)+[df.mean(), df.var()], 
        'Percentage': list((counter.values / counter.sum()))+["", ""]})
    
    return friquency_dataframe

def two_variable_ss(df, var1, var2):

    counter = df.groupby(var1)[var2].value_counts()
    friquency_dataframe = counter.unstack(fill_value=0)

    #friquency_dataframe = sorting(friquency_dataframe)

    column_sums = friquency_dataframe.sum(axis=0)
    percentage_dataframe = friquency_dataframe.div(column_sums, axis=1)

    friquency_dataframe['Total'] = list(single_answer(df[var1]).iloc[:,1])[:-1]
    friquency_dataframe.loc['Sample_size'] = list(single_answer(df[var2]).iloc[:,1])
    percentage_dataframe['Total'] = list(single_answer(df[var1]).iloc[:,2])[:-1]
    percentage_dataframe.loc['Sample_size'] = list(single_answer(df[var2]).iloc[:,1])
    
    return percentage_dataframe, friquency_dataframe

def two_variable_sm(df, var1, var2):
    unique_values = list(set(df[var1].dropna()))
    value = multi_answer(df[var2]).iloc[:-1,0]
    friquency_dataframe, percentage_dataframe = {}, {}

    for i in unique_values:
        dataframe = multi_answer(df[df[var1] == i][var2]).iloc[:-1,:]
        friquency_dataframe[i], percentage_dataframe[i] = dataframe['Frequency'], dataframe['Percentage']

    friquency_dataframe = pd.DataFrame(friquency_dataframe)
    percentage_dataframe = pd.DataFrame(percentage_dataframe)

    friquency_dataframe.index, percentage_dataframe.index = value, value

    #friquency_dataframe = sorting(friquency_dataframe)
    #percentage_dataframe = sorting(percentage_dataframe)

    friquency_dataframe['Total'] = list(multi_answer(df[var2]).iloc[:,1])[:-1]
    friquency_dataframe.loc['Sample_size'] = list(single_answer(df[var1]).iloc[:,1])
    percentage_dataframe['Total'] = list(multi_answer(df[var2]).iloc[:,2])[:-1]
    percentage_dataframe.loc['Sample_size'] = list(single_answer(df[var1]).iloc[:,1])
    

    return percentage_dataframe, friquency_dataframe

def two_variable_mm(df, var1, var2):
    friquency_dataframe, percentage_dataframe = {}, {}
    value = multi_answer(df[var2]).iloc[:-1,0]

    for i in var1:
        unique_values = list(set(df[i].dropna()))[0]
        dataframe = multi_answer(df[df[i] == unique_values][var2]).iloc[:-1,:]
        friquency_dataframe[i], percentage_dataframe[i] = dataframe['Frequency'], dataframe['Percentage']

    friquency_dataframe = pd.DataFrame(friquency_dataframe)
    percentage_dataframe = pd.DataFrame(percentage_dataframe)

    friquency_dataframe.index, percentage_dataframe.index = value, value

    #friquency_dataframe = sorting(friquency_dataframe)
    #percentage_dataframe = sorting(percentage_dataframe)

    friquency_dataframe['Total'] = list(multi_answer(df[var2]).iloc[:,1])[:-1]
    friquency_dataframe.loc['Sample_size'] = list(multi_answer(df[var1]).iloc[:,1])
    percentage_dataframe['Total'] = list(multi_answer(df[var2]).iloc[:,2])[:-1]
    percentage_dataframe.loc['Sample_size'] = list(multi_answer(df[var1]).iloc[:,1])

    return percentage_dataframe, friquency_dataframe

def two_variable_ssc(df, var1, var2):
    unique_values = list(set(df[var1].dropna()))
    mean_dataframe = {'Mean': [], 'Variation': [], 'Frequency': []}
    for i in unique_values:
        d = df[df[var1] == i][var2]
        mean_dataframe['Mean'] += [d.mean()]
        mean_dataframe['Variation'] += [d.var()]
        mean_dataframe['Frequency'] += [len(d)]

    mean_dataframe = pd.DataFrame(mean_dataframe)
    mean_dataframe.index = unique_values

    mean_dataframe.loc['Total'] = [df[var2].mean(), df[var2].var(), len(df[var2])]

    return mean_dataframe

def two_variable_msc(df, var1, var2):
    mean_dataframe, unique_values = {'Mean': [], 'Variation': [], 'Frequency': []}, []
    for i in var1:
        d = df[i].dropna()
        j = list(set(df[i].dropna()))[0]
        d = df[df[i]==j][var2]
        unique_values += [j]
        mean_dataframe['Mean'] += [d.mean()]
        mean_dataframe['Variation'] += [d.var()]
        mean_dataframe['Frequency'] += [len(d)]

    mean_dataframe = pd.DataFrame(mean_dataframe)
    mean_dataframe.index = unique_values

    mean_dataframe.loc['Total'] = [df[var2].mean(), df[var2].var(), len(df[var2])]

    return mean_dataframe

def funnel(df, dictionary):
    friquency = {}
    for i in dictionary.keys():
        if dictionary[i] == "Single":
            friquency[i] = list(single_answer(df[i])['Frequency'])[:-1]

        elif dictionary[i] == "Multi":
            matching_cols = [col for col in df.columns if is_matching_pattern(col, i)]
            friquency[i] = list(multi_answer(df[matching_cols])['Frequency'])[:-1]

        elif dictionary[i] == "Score":
            friquency[i] = list(score_answer(df[i])['Frequency'])[:-1]

    try:
        friquency = pd.DataFrame(friquency)
    except:
        friquency = equalize_list_lengths(friquency)
    
    first = None
    for key, value in dictionary.items():
        if value == "Single":
            first = key
            break
    
    percentage = friquency/len(df[first])

    return friquency, percentage

def t_test(m1, m2, n1, n2, v1, v2):
    te = (m1 - m2) / ((v1/n1 + v2/n2)**0.5)
    p_value = 2 * (1 - t.cdf(abs(te), n1+n2-2))
    return p_value

def z_testes(n1, n2, p1, p2):
    p_hat = ((n1*p1) + (n2*p2)) / (n1 + n2)
    z = (p1 - p2) / ((p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)) ** 0.5)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_value

def z_test_data(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    num_rows, num_cols = df.shape
    
    for i in range(num_rows -1):
        for j in range(num_cols -1):
            n1 = df.iloc[-1, -1]
            n2 = df.iloc[-1, j]
            p1 = df.iloc[i, -1]
            p2 = df.iloc[i, j]
            p_value = z_testes(n1, n2, p1, p2)
            if pd.notnull(p_value) and p_value <= 0.05:
                styles.iloc[i, j] = 'background-color: lightgreen'
                
    return df.style.apply(lambda _: styles, axis=None)

def t_test_data(df):
    rows, cols = df.shape
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    for i in range(rows-1):
        p_value = t_test(df['Mean'].iloc[-1,], df['Mean'].iloc[i], df['Frequency'].iloc[-1], df['Frequency'].iloc[i], df['Variation'].iloc[-1], df['Variation'].iloc[i])
        if p_value <= 0.05:
            styles.iloc[i, :] = 'background-color: lightgreen'

    return df.style.apply(lambda _: styles, axis=None)

def Z_test_dataframes(sheets_data):
    """Processes each sheet's DataFrame and computes new DataFrames with Z-test results."""
    result_dataframes = {}
    for sheet_name, df in sheets_data.items():
        if df.empty:
            st.warning(f"⚠️ Sheet '{sheet_name}' is empty and has been skipped.")
            continue
        df = df.set_index(df.columns[0])  # Use the first column as index
        rows, cols = df.shape
        if cols < 2:
            st.warning(f"⚠️ Sheet '{sheet_name}' does not have enough columns for analysis and has been skipped.")
            continue
        new_df = pd.DataFrame(index=df.index[:-1], columns=df.columns[1:])
        for i, row_name in enumerate(df.index[:-1]):
            for j, col_name in enumerate(df.columns[1:]):
                try:
                    n1 = df.iloc[-1, 0]  # x_I1
                    n2 = df.iloc[-1, j+1]  # x_Ij
                    p1 = df.iloc[i, 0]  # x_1J
                    p2 = df.iloc[i, j+1]  # x_ij
                    p_value = z_testes(n1, n2, p1, p2)
                    new_df.iloc[i, j] = p_value
                except Exception as e:
                    st.error(f"❌ Error processing sheet '{sheet_name}', row '{row_name}', column '{col_name}': {e}")
                    new_df.iloc[i, j] = np.nan

        result_dataframes[sheet_name] = new_df

    return result_dataframes

def analyze_z_test(file):
    """
    Performs Z-Test analysis on the uploaded Excel file.

    Parameters:
    - file: Uploaded Excel file

    Returns:
    - result_dataframes: Dictionary of DataFrames with p-values
    """
    sheets_data = read_excel_sheets(file)
    if sheets_data is None:
        return None

    result_dataframes = Z_test_dataframes(sheets_data)

    if not result_dataframes:
        st.error("❌ No valid sheets found for Z-Test analysis.")
        return None

    st.write("### 📈 Processed Tables with Z-Test Results")
    for sheet_name, df in result_dataframes.items():
        st.write(f"#### Sheet: {sheet_name}")
        
        # Apply color coding based on p-value
        def color_p_value(val):
            try:
                if pd.isna(val):
                    return 'background-color: lightgray'
                elif val < 0.05:
                    return 'background-color: lightgreen'
                else:
                    return 'background-color: lightcoral'
            except:
                return 'background-color: lightgray'
        
        styled_df = df.style.applymap(color_p_value)
        
        # Display the styled DataFrame
        st.dataframe(styled_df, use_container_width=True)
    
    return result_dataframes

def join_dataframes(mlist, dataframes):
    max_rows = max(df.shape[0] for df in dataframes)
    result = pd.DataFrame()
    col_counts = {}  # Dictionary to track column name occurrences

    for i, df in enumerate(dataframes):
        rows_to_add = max_rows - df.shape[0]
        if rows_to_add > 0:
            empty_rows = pd.DataFrame(index=range(rows_to_add), columns=df.columns)
            empty_rows = empty_rows.fillna(np.nan)
            df = pd.concat([df, empty_rows], ignore_index=True)

        for col in df.columns:
            original_col = col
            count = 1
            while col in result.columns:
                col = f"{original_col}_{count}"
                count += 1
            col_counts[original_col] = col_counts.get(original_col, 0) + 1
            df = df.rename(columns={original_col: col})
        
        result = pd.concat([result, df.reset_index(drop=True)], axis=1)

    return result

def all_tabulation(df, main_dict, follow_dict):

    for j in main_dict["single"]:

        dataframe_list, name_list = (single_answer(df[j]),), [j]

        for i in follow_dict["single"]:
            dataframe_list = dataframe_list + (two_variable_ss(df[[j, i]], j, i)[0], )
            name_list.append(i)

        for i in follow_dict["multi"]:
            matching_cols1 = [col for col in df.columns if is_matching_pattern(col, i)]
            dataframe_list = dataframe_list + (two_variable_sm(df[[j] + matching_cols1], j, matching_cols1)[0].T, )
            name_list.append(i)

        for i in follow_dict["score"]:
            dataframe_list = dataframe_list + (two_variable_ssc(df[[j, i]], j, i), )
            name_list.append(i)

        st.subheader(j)
        st.markdown(j + ": " + ", ".join(follow_dict['single'] + follow_dict['multi'] + follow_dict['score']))
        st.dataframe(join_dataframes(name_list, dataframe_list))

    for j in main_dict["multi"]:

        matching_cols0 = [col for col in df.columns if is_matching_pattern(col, j)]
        dataframe_list, name_list = (multi_answer(df[matching_cols0]), ), [j]

        #for i in follow_dict["single"]:
            #dataframe_list.append(two_variable_ms(df[matching_cols0, i], matching_cols0, i).drop(columns=['Value'])[0])

        for i in follow_dict["multi"]:
            matching_cols1 = [col for col in df.columns if is_matching_pattern(col, i)]
            dataframe_list = dataframe_list + (two_variable_mm(df[matching_cols0 + matching_cols1], matching_cols0, matching_cols1)[0], )
            name_list.append(i)

        for i in follow_dict["score"]:
            dataframe_list = dataframe_list + (two_variable_msc(df[matching_cols0 + [i]], matching_cols0, i), )
            name_list.append(i)

        st.subheader(j)
        st.markdown(j + ": " + ", ".join(follow_dict['multi'] + follow_dict['score']))
        st.dataframe(join_dataframes(name_list, dataframe_list))

    for j in main_dict["score"]:

        dataframe_list, name_list = [single_answer(df[j])], [j]

        st.subheader(j)
        st.dataframe(join_dataframes(name_list, dataframe_list))

def process_dataframe(df):
  df = df.fillna(0)
  for col in df.columns:
    df[col] = pd.Categorical(df[col])
  return df

import numpy as np
from sklearn.decomposition import PCA

def pca_with_variance_threshold(data, threshold=0.80):
    pca = PCA()
    pca.fit(data)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    
    return transformed_data

def hierarchical_clustering_with_plotly(df, linkage_method):

    df_encoded = df.apply(lambda x: pd.factorize(x)[0])

    Z = linkage(df_encoded, method=linkage_method)

    fig = ff.create_dendrogram(df_encoded, linkagefun=lambda x: Z, orientation='bottom')
    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig)

    num_clusters = int(st.text_input("Enter the desired number of clusters"))

    clusters = fcluster(Z, num_clusters, criterion='maxclust')

    df['Cluster'] = clusters

    return df

def kmeans_clustering(df, k):

  numeric_df = df.select_dtypes(include=['number'])

  if numeric_df.empty:
    raise ValueError("DataFrame does not contain any numeric columns for clustering.")
    
  kmeans = KMeans(n_clusters=k, random_state=0) # You can modify random_state
  df['cluster'] = kmeans.fit_predict(numeric_df)
  return df

def sample_size_calculator(confidence_level, p, E):
    Z = norm.ppf(1 - (1 - confidence_level) / 2)
    
    n = (Z**2 * p * (1 - p)) / (E**2)
    
    n = math.ceil(n)
    
    return n

import pandas as pd

def categorize_sentences(prompt, df, Text_name):
    texts = df[Text_name].tolist()

    labels = []
    for text in texts:
        result = classifier(text, candidate_labels=[prompt])
        labels.append(result['labels'][0])

    df['labels'] = labels
    return df

def upload_and_select_dataframe():
    st.sidebar.title("File Upload")
    uploaded_files = st.sidebar.file_uploader("Choose CSV or Excel files", type=["csv", "xlsx", "xls", "xlsb"], accept_multiple_files=True)
      
    dataframes = {}
    dataframes["Sample Dataset"] = pd.read_excel("Sample dataset.xlsx")

    try:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith(('.csv')):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xls', '.xlsx', '.xlsb')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.sidebar.error(f"Unsupported file type: {uploaded_file.name}")
                    continue
                dataframes[uploaded_file.name] = df
            except Exception as e:
                  st.sidebar.error(f"Error reading {uploaded_file.name}: {e}")
      
        if len(uploaded_files) > 7:
            st.sidebar.error('Maximum 7 files can be uploaded.')
            return None
    
        if dataframes:
            selected_file = st.sidebar.selectbox("Select a DataFrame", ["Select a dataset"] + list(dataframes.keys()))
            return dataframes[selected_file]
        else:
            st.sidebar.info("Please upload some files.")
            return None
    except:
        pass

#st.markdown('[Click to register a suggestion or comment](https://docs.google.com/forms/d/e/1FAIpQLScLyP7bBbqMfGdspjL7Ij64UZ6v2KjqjKNbm8gwEsgWsFs_Qg/viewform?usp=header)')

st.image("Insightzen.png", width=600)

df = upload_and_select_dataframe()

try:
    try:
        d = df.head()
        st.subheader("Data preview")
        
        st.data_editor(df)
        #st.dataframe(df.head())

        cols = edit_strings(df.columns)
        cols = sorted(list(set(cols)))
    except:
        pass
    
    main_option = st.selectbox("Please select an option:", ["Select a Task","Tabulation", "Funnel Analysis", "Segmentation Analysis", "Hypothesis test", "Machine Learning", "Sample Size Calculator" ,"Coding", "AI Chat"])
    
    if main_option == "Tabulation":
        st.header("Tabulation Analysis")
    
        tabulation_option = st.selectbox("Please select the type of analysis:", ["Univariate", "Multivariate", "All"])
    
        if tabulation_option == "All":
            
            st.sidebar.header("Settings")
    
            main_dict = {"single": [], "multi": [], "score": []}
            
            st.sidebar.subheader("Main")
            main_dict["single"] = st.sidebar.multiselect(
                'Main: Single answer questions', 
                cols,
                default=[]
            )
            
            main_dict["multi"] = st.sidebar.multiselect(
                'Main: Multi answer questions',
                cols,
                default=[]
            )
    
            main_dict["score"] = st.sidebar.multiselect(
                'Main: Score answer questions',
                cols,
                default=[]
            )
    
            follow_dict = {"single": [], "multi": [], "score": []}
            
            st.sidebar.subheader("Follow")
            follow_dict["single"] = st.sidebar.multiselect(
                'Follow: Single answer questions', 
                cols,
                default=[]
            )
            
            follow_dict["multi"] = st.sidebar.multiselect(
                'Follow: Multi answer questions',
                cols,
                default=[]
            )
    
            follow_dict["score"] = st.sidebar.multiselect(
                'Follow: Score answer questions',
                cols,
                default=[]
            )
    
            all_tabulation(df, main_dict, follow_dict)
            
        elif tabulation_option == "Univariate":
            uni_option = st.selectbox("Select the type of univariate analysis:", ["Multiple answer", "Single answer", "Score answer"])
    
            if uni_option == "Single answer":
                var = st.text_input("Please enter the name of the desired column:")
                if var:
                    if var in df.columns:
                        result_df = single_answer(df[var])
                        st.subheader("Univariate Analysis Results")
                        st.dataframe(result_df)
    
                        fig = figo('Bar', result_df["Percentage"][:-1, ], title='Percentage Histogram', xlabel=var, ylabel='Percentage', colorscale='Plotly3')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("The entered column was not found.")
            elif uni_option == "Multiple answer":
                var = st.text_input("Please enter the name of the desired column:")
                if var:
                    matching_cols = [col for col in df.columns if is_matching_pattern(col, var)]
                    if matching_cols:
                        subset_df = df[matching_cols]
                        result_df = multi_answer(subset_df)
                            
                        st.subheader("Multiple Answer Analysis Results")
                        st.dataframe(result_df)
                        
                        fig = figo('Bar', result_df["Percentage"][:-1], title='Percentage Histogram', xlabel=var, ylabel='Percentage', colorscale='Plotly3')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No columns matching the entered pattern were found.")
    
            elif uni_option == "Score answer":
                var = st.text_input("Please enter the name of the desired column:")
                if var:
                    subset_df = df[var]
                    result_df = score_answer(subset_df)
    
                    st.subheader("Score Answer Analysis Results")
                    st.dataframe(result_df)
                    
                    fig = figo('Bar', result_df["Percentage"][:-2], title='Percentage Histogram', xlabel=var, ylabel='Percentage', colorscale='Plotly3')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No columns matching the entered pattern were found.")
                    
        elif tabulation_option == "Multivariate":
            st.subheader("Multivariate Analysis")
            var1 = st.text_input("Please enter the name of the first column:")
            var2 = st.text_input("Please enter the name of the second column:")
    
            if var1 and var2:
                type1 = st.selectbox("Select the type of analysis for the first column:", ["Multiple answer", "Single answer"], key='type1')
                type2 = st.selectbox("Select the type of analysis for the second column:", ["Multiple answer", "Single answer", "Score answer"], key='type2')
    
                if type1 == "Single answer" and type2 == "Single answer":
                    percentile_df, frequency_df = two_variable_ss(df[[var1, var2]], var1, var2)
                    st.subheader("Percentage Table")
                    st.write(z_test_data(percentile_df))
    
                    st.subheader("Frequency Table")
                    st.dataframe(frequency_df)
    
                    row, col = df.shape
                    fig = figo('Scatter', percentile_df.iloc[:-1,:], title='Percentage Scatter plot', width=(col*5)+5, height=(row*25) + 10)
                    st.plotly_chart(fig, use_container_width=True)
    
                elif type1 == "Single answer" and type2 == "Multiple answer":
                    matching_cols = [col for col in df.columns if is_matching_pattern(col, var2)]
                    if matching_cols:
                        percentile_df, frequency_df = two_variable_sm(df[[var1] + matching_cols], var1, matching_cols)
                        st.subheader("Percentage Table")
                        st.write(z_test_data(percentile_df))
    
                        st.subheader("Frequency Table")
                        st.dataframe(frequency_df)
    
                        row, col = df.shape
                        fig = figo('Scatter', percentile_df.iloc[:-1,:], title='Percentage Scatter plot', width=(col*5)+5, height=(row*25) + 10)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("No columns matching the entered pattern were found.")
    
                elif type1 == "Multiple answer" and type2 == "Multiple answer":
                    matching_cols1 = [col for col in df.columns if is_matching_pattern(col, var1)]
                    matching_cols2 = [col for col in df.columns if is_matching_pattern(col, var2)]
                    if matching_cols1 and matching_cols2:
                        percentile_df, frequency_df = two_variable_mm(df[matching_cols1 + matching_cols2], matching_cols1, matching_cols2)
                        st.subheader("Percentage Table")
                        st.write(z_test_data(percentile_df))
    
                        st.subheader("Frequency Table")
                        st.dataframe(frequency_df)
    
                        row, col = df.shape
                        fig = figo('Scatter', percentile_df.iloc[:-1,:], title='Percentage Scatter plot', width=(col*5)+5, height=(row*25) + 10)
                        st.plotly_chart(fig, use_container_width=True)
    
                elif type1 == "Single answer" and type2 == "Score answer":
    
                    mean_df = two_variable_ssc(df[[var1, var2]], var1, var2)
                    st.subheader("Mean Table")
                    st.write(t_test_data(mean_df))
    
                    row, col = df.shape
                    fig = figo('Bar', mean_df["Mean"][:-1], title='Mean Histogram', xlabel=var1, ylabel='Mean', colorscale='Plotly3')
                    st.plotly_chart(fig, use_container_width=True)                            
    
    
                elif type1 == "Multiple answer" and type2 == "Score answer":
                    matching_cols1 = [col for col in df.columns if is_matching_pattern(col, var1)]
                    if matching_cols1:
                        mean_df = two_variable_msc(df[matching_cols1 + [var2]], matching_cols1, var2)
                        st.subheader("Mean Table")
                        st.write(t_test_data(mean_df))
    
                        row, col = df.shape
                        fig = figo('Bar', mean_df["Mean"][:-1], title='Mean Histogram', xlabel=var1, ylabel='Mean', colorscale='Plotly3')
                        st.plotly_chart(fig, use_container_width=True)                    
                else:
                    st.info("This section of the program is under development.")
    
    elif main_option == "Funnel Analysis":
        st.header("Funnel")
        
        st.sidebar.header("Funnel Settings")
        single_list = st.sidebar.multiselect(
            'Single answer questions', 
            cols,
            default=[]
        )
        
        multi_list = st.sidebar.multiselect(
            'Multi answer questions',
            cols,
            default=[]
        )
        selected_dict = {}
        
        for option in single_list:
            selected_dict[option] = "Single"
        for option in multi_list:
            selected_dict[option] = "Multi"
        
        funnel_frequency, funnel_percentage = funnel(df, selected_dict)
        st.subheader("Percentage Table")
        st.dataframe(funnel_percentage)
        
        st.subheader("Frequency Table")
        st.dataframe(funnel_frequency)
        
        st.sidebar.header("Chart Settings")
        
        bar_columns = st.sidebar.multiselect('Which columns should be displayed as bar charts?', sorted(funnel_percentage.columns))
        line_columns = st.sidebar.multiselect('Which columns should be displayed as line charts?', sorted(funnel_percentage.columns))
        
        funnel_percentage_cleaned = funnel_percentage.dropna(axis=0, how='all')
        
        columns = st.sidebar.multiselect('Sort by which questions?', sorted(funnel_percentage_cleaned.columns))
        sort_order = st.sidebar.radio('Sort Order', ['Ascending', 'Descending'])
        
        ascending = True if sort_order == 'Ascending' else False
        funnel_percentage_cleaned = funnel_percentage_cleaned.sort_values(by=columns, ascending=ascending)
        
        # Update index to string to ensure proper sorting in Plotly
        funnel_percentage_cleaned.index = funnel_percentage_cleaned.index.astype(str)
        
        fig = go.Figure()
        
        # Define modern and diverse color palette
        modern_colors = [
            "#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1", 
            "#955251", "#B565A7", "#009B77", "#DD4124", "#45B8AC"
        ]
        
        # Add Bar traces with transparency and custom colors
        for idx, col in enumerate(bar_columns):
            funnel_percentage_col = funnel_percentage_cleaned[col]
            fig.add_trace(
                go.Bar(
                    x=funnel_percentage_cleaned.index, 
                    y=funnel_percentage_col, 
                    name=col,
                    marker_color=modern_colors[idx % len(modern_colors)],  # Cycle through colors
                    opacity=0.8  # Set transparency
                )
            )
        
        # Add Line traces with transparency and custom colors
        for idx, col in enumerate(line_columns):
            funnel_percentage_col = funnel_percentage_cleaned[col]
            fig.add_trace(
                go.Scatter(
                    x=funnel_percentage_cleaned.index, 
                    y=funnel_percentage_col, 
                    mode='lines', 
                    name=col,
                    line=dict(color=modern_colors[(idx + len(bar_columns)) % len(modern_colors)]),  # Cycle through colors
                    opacity=0.8  # Set transparency
                )
            )
        
        fig.update_layout(
            title="Combined Bar and Line Chart",
            xaxis_title="Brands",
            yaxis_title="Percentage",
            template="plotly_dark",
            barmode="group",
            xaxis=dict(
                tickmode='array', 
                categoryorder='array', 
                categoryarray=funnel_percentage_cleaned.index.tolist()  # Ensure sorting of x-axis matches the DataFrame index
            )
        )
        
        st.plotly_chart(fig)

    elif main_option == "Segmentation Analysis":
        st.header("Segmentation Analysis")
        
        st.sidebar.header("Selection of questions")
        single_list = st.sidebar.multiselect(
            'Single answer questions', 
            cols,
            default=[]
        )
    
        multi_list = st.sidebar.multiselect(
            'Multi answer questions', 
            cols,
            default=[]
        )
    
        score_list = st.sidebar.multiselect(
            'Score answer questions', 
            cols,
            default=[]
        )
    
        matching_cols1 = []
        for i in multi_list:
            matching_cols1 += [col for col in df.columns if is_matching_pattern(col, i)]
    
        df_clean = process_dataframe(df[single_list + matching_cols1])
        st.subheader("Selected Table")
        st.dataframe(df_clean)

        PCA_radio = st.sidebar.radio('PCA', ['Yes', 'No'])

        if PCA_radio == 'Yes':
            dfc = pd.DataFrame(pca_with_variance_threshold(df_clean, threshold=0.80))
        else:
            dfc = df_clean

        
        selected_method = st.sidebar.selectbox("Select the Linkage Method of Segmentation Analysis:", ['Hierarchical Clustering', 'K-means Clustering'])
        if selected_method == 'Hierarchical Clustering':
            linkage_method = st.sidebar.selectbox("Select the Linkage Method of Segmentation Analysis:", ['average', 'single', 'complete', 'weighted', 'centroid', 'median', 'ward'])
            df_cluster = hierarchical_clustering_with_plotly(dfc, linkage_method)
        if selected_method == 'K-means Clustering':
            k = int(st.text_input("Enter the desired number of clusters"))
            df_clean = kmeans_clustering(dfc, k)

        st.subheader("Cluster Table")
        st.dataframe(df_clean)
    
    elif main_option == "Hypothesis test":
        st.header("Hypothesis Testing")
        hypothesis_option = st.selectbox("Please select the type of hypothesis test:", ["Z test", "T test", "Chi-Square test", "ANOVA test"])
    
        if hypothesis_option != "Z test":
            st.info("This section of the program is under development.")
        else:
            uploaded_file = st.file_uploader("Please upload your Excel file for Z-Test", type=["xlsx", "xls"])
            if uploaded_file:
                result = analyze_z_test(uploaded_file)
                if result:
                    st.success("Z-Test analysis completed successfully.")

    elif main_option == "Coding":
        selected_list = st.sidebar.multiselect(
            'Select the desired "Open Question" column.', 
            cols,
            default=[]
        )
        df["id"] = df.index
        prompt_user = st.text_input("Write a brief description of the selected column question.")
        if st.button("Submit"):

            df2 = categorize_sentences(prompt_user, df, selected_list)
            
            st.subheader("Categorized data")
            st.dataframe(df2)
        
    elif main_option == "Machine Learning":
        st.info("This section of the program is under development.")

    elif main_option == "AI Chat":

        client = InferenceClient(
        	provider="together",
        	api_key="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
        )
        
        messages = [
        	{
        		"role": "user",
        		"content": "What is the capital of France?"
        	}
        ]
        
        stream = client.chat.completions.create(
        	model="deepseek-ai/DeepSeek-R1", 
        	messages=messages, 
        	max_tokens=500,
        	stream=True
        )
        
        for chunk in stream:
            st.warning(chunk.choices[0].delta.content, end="")

    elif main_option == "Sample Size Calculator":
        st.header("Sample Size Calculator")
        confidence_level = st.text_input("Confidence levels: (In percentage terms)")
        p = st.text_input("Estimated probability of success: (In percentage terms)")
        E = st.text_input("Margin of error: (In percentage terms)")
        try:
            confidence_level, p, E = float(confidence_level)/100, float(p)/100, float(E)/100
            n = sample_size_calculator(confidence_level, p, E)
        except:
            pass

        st.write(f"Sample size: {n}")

except Exception as e:
    pass
    #st.error(f"❌ Error: {e}")
