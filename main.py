import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
import plotly.graph_objects as go
import plotly.express as px  # Import plotly express

app = FastAPI()

# Global variable to store the uploaded CSV data
uploaded_csv_data = None

# Function to read CSV data
def read_csv_data(content: bytes):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV data: {str(e)}")

# Function to clean the dataset
def clean_dataset(df):
    try:
        # Get number of rows and columns in the dataset
        num_rows, num_columns = df.shape
        print(f"Number of rows in the dataset: {num_rows}")
        print(f"Number of columns in the dataset: {num_columns}")

        # Check if number of rows exceeds the limit
        if num_rows > 3000:
            print(f"WARNING: Application Crash Imminent - Number of rows found = {num_rows}, exceeds the limit (3000 rows). Data will be truncated to 3000 rows only.")
            df = df.iloc[:3000, :]

        # Check if number of columns exceeds the limit
        if num_columns > 30:
            print(f"WARNING: Application Crash Imminent - Number of columns found = {num_columns}, exceeds the limit (30 columns). Data will be truncated to 30 columns only.")
            df = df.iloc[:, :30]

        # Drop columns with string values
        df = df.select_dtypes(exclude=['object'])

        # Replace NaN values with mean values representation
        df = df.fillna(df.mean())

        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning dataset: {str(e)}")

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_csv_data
    if file.filename.endswith('.csv'):
        content = await file.read()
        try:
            # Read CSV file
            df = read_csv_data(content)
            
            # Clean dataset
            cleaned_df = clean_dataset(df)
            
            # Update uploaded_csv_data
            uploaded_csv_data = cleaned_df
            
            # Get rows and columns count
            rows, columns = uploaded_csv_data.shape
            
            # Get summary
            summary = uploaded_csv_data.describe().to_dict()
            
            # Count empty cells in each column
            empty_cells_count = uploaded_csv_data.isnull().sum().to_dict()
            for column, empty_count in empty_cells_count.items():
                summary[column]['empty_cells'] = empty_count
            
            # Retrieve data of each column
            column_data = {}
            for column in uploaded_csv_data.columns:
                column_data[column] = uploaded_csv_data[column].tolist()

            return {"rows": rows, "columns": columns, "summary": summary, "column_data": column_data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

# Scatter Plot
@app.post("/plot/scatter/")
async def plot_scatter_chart(x_column: str, y_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")

    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")

    fig = go.Figure(data=go.Scatter(x=uploaded_csv_data[x_column], y=uploaded_csv_data[y_column], mode='markers'))
    fig.update_layout(title=f"Scatter Plot ({x_column} vs {y_column})")
    fig.update_xaxes(title_text=x_column)
    fig.update_yaxes(title_text=y_column)
    
    return fig.to_json()

# Bar chart
@app.post("/plot/bar/")
async def plot_bar_chart(x_column: str, y_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = go.Figure(data=go.Bar(x=uploaded_csv_data[x_column], y=uploaded_csv_data[y_column]))
    fig.update_layout(title=f"Bar Chart ({y_column} vs {x_column})", xaxis_title=x_column, yaxis_title=y_column)
    
    return fig.to_json()

# Histogram
@app.post("/plot/histogram/")
async def plot_histogram(column_name: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if column_name not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{column_name} column not found in CSV file. Please check the column names and try again.")
    
    fig = go.Figure(data=go.Histogram(x=uploaded_csv_data[column_name]))
    fig.update_layout(title=f"Histogram for {column_name}", xaxis_title=column_name, yaxis_title="Count")
    
    return fig.to_json()

# Heatmap
@app.post("/plot/heatmap/")
async def plot_heatmap(x_column: str, y_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")
    
    pivot_table = uploaded_csv_data.pivot_table(values=y_column, index=x_column, aggfunc='mean')
    fig = go.Figure(data=go.Heatmap(z=pivot_table.values, x=pivot_table.columns, y=pivot_table.index))
    fig.update_layout(title=f"Heatmap ({y_column} vs {x_column})", xaxis_title=x_column, yaxis_title=y_column)
    
    return fig.to_json()

# 3D Scatter Plot
@app.post("/plot/scatter3d/")
async def plot_scatter3d_chart(x_column: str, y_column: str, z_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")

    if z_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{z_column} column not found in CSV file. Please check the column names and try again.")

    fig = go.Figure(data=go.Scatter3d(x=uploaded_csv_data[x_column], y=uploaded_csv_data[y_column], z=uploaded_csv_data[z_column], mode='markers'))
    fig.update_layout(title=f"3D Scatter Plot ({x_column} vs {y_column} vs {z_column})")
    fig.update_scenes(xaxis_title=x_column, yaxis_title=y_column, zaxis_title=z_column)
    
    return fig.to_json()

# Density Mapbox
@app.post("/plot/density_mapbox/")
async def plot_density_mapbox(column_name: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if column_name not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{column_name} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.density_mapbox(uploaded_csv_data, lat=uploaded_csv_data.index, lon=uploaded_csv_data.index, z=uploaded_csv_data[column_name], radius=10)
    fig.update_layout(title=f"Density Mapbox for {column_name}")
    
    return fig.to_json()

# Pie Chart
@app.post("/plot/pie/")
async def plot_pie_chart(values_column: str, names_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if values_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{values_column} column not found in CSV file. Please check the column names and try again.")
    
    if names_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{names_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.pie(uploaded_csv_data, values=values_column, names=names_column)
    fig.update_layout(title=f"Pie Chart ({values_column} vs {names_column})")
    
    return fig.to_json()

# Violin Plot
@app.post("/plot/violin/")
async def plot_violin_chart(x_column: str, y_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.violin(uploaded_csv_data, x=x_column, y=y_column)
    fig.update_layout(title=f"Violin Plot ({x_column} vs {y_column})")
    
    return fig.to_json()

# Pie Chart
@app.post("/plot/pie/")
async def plot_pie_chart(values_column: str, names_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if values_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{values_column} column not found in CSV file. Please check the column names and try again.")
    
    if names_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{names_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.pie(uploaded_csv_data, values=values_column, names=names_column)
    fig.update_layout(title=f"Pie Chart ({values_column} vs {names_column})")
    
    return fig.to_json()

# Violin Plot
@app.post("/plot/violin/")
async def plot_violin_chart(x_column: str, y_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.violin(uploaded_csv_data, x=x_column, y=y_column)
    fig.update_layout(title=f"Violin Plot ({x_column} vs {y_column})")
    
    return fig.to_json()

# Strip Plot
@app.post("/plot/strip/")
async def plot_strip_chart(x_column: str, y_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.strip(uploaded_csv_data, x=x_column, y=y_column)
    fig.update_layout(title=f"Strip Plot ({x_column} vs {y_column})")
    
    return fig.to_json()

# ECDF Plot
@app.post("/plot/ecdf/")
async def plot_ecdf_chart(x_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.ecdf(uploaded_csv_data, x=x_column)
    fig.update_layout(title=f"ECDF Plot ({x_column})")
    
    return fig.to_json()

# Density Contour Plot
@app.post("/plot/density_contour/")
async def plot_density_contour_chart(x_column: str, y_column: str):
    global uploaded_csv_data
    if uploaded_csv_data is None:
        raise HTTPException(status_code=400, detail="CSV file not uploaded")
    
    if x_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{x_column} column not found in CSV file. Please check the column names and try again.")
    
    if y_column not in uploaded_csv_data.columns:
        raise HTTPException(status_code=400, detail=f"{y_column} column not found in CSV file. Please check the column names and try again.")
    
    fig = px.density_contour(uploaded_csv_data, x=x_column, y=y_column)
    fig.update_layout(title=f"Density Contour Plot ({x_column} vs {y_column})")
    
    return fig.to_json()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
