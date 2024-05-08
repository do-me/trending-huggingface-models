# %% [markdown]
# # Mine jsons from HF and paginate. At the moment (may 2024) it's only 6 pages, causing neglectable traffic

# %%
import requests
import pandas as pd 
from tqdm import tqdm 
import panel as pn
tqdm.pandas()

# Base URL for the API
base_url = "https://huggingface.co/models-json?other=feature-extraction&library=transformers.js&sort=trending&numItemsPerPage=50" # attention: https://huggingface.co/posts/do-me/362814004058611

# List to store all models
all_models = []

# Total number of pages to fetch
total_pages = 12 # adding more pages here for the future, should be raised once there are more than 300 models

# Use tqdm to show progress
for page_number in tqdm(range(0, total_pages + 1), desc="Fetching Pages"):
    # Construct the full URL for the current page
    url = f"{base_url}&p={page_number}"
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data from the response
        json_data = response.json()
        
        # Extract the models from the current page's data
        page_models = json_data['models']
        
        # Append the models to the list
        all_models.extend(page_models)
    else:
        print(f"Failed to fetch data from page {page_number}. Status code: {response.status_code}")

print(all_models.__len__(), "Models mined")

df = pd.DataFrame(all_models)


# %% [markdown]
# # For each of the models have a look at the onnx file sizes. Must request each page once unfortunately as it's not in the model's json
# Takes not more than 1.5 mins

# %%
from bs4 import BeautifulSoup
import requests

def extract_size_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all 'a' tags with the specified title attribute
            a_tags = soup.find_all('a', title="Download file")
            model_sizes = []  # Initialize a list to store sizes of model files

            for a_tag in a_tags:
                file_name_tag = a_tag.find_previous_sibling('div').find('span')
                if not file_name_tag:  # Skip if there's no 'span' tag
                    continue
                file_name = file_name_tag.text.strip()
                if file_name.endswith(".onnx"):
                    if file_name.startswith("model"):
                        size = a_tag.text.strip().split("\n")[0]
                        model_sizes.append(size)
                    else: # only if there is no normal model
                        if file_name.startswith(("decoder", "encoder")):
                            size = a_tag.text.strip().split("\n")[0]
                            model_sizes.append(size)
                
            if model_sizes:
                return model_sizes
            else:
                return ""
        else:
            return ""#f"HTTP Status Code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    
# extract_size_from_url("https://huggingface.co/Xenova/instructor-large/tree/main/onnx") # test

def scrape_sizes(model):
    sizes = extract_size_from_url(f"https://huggingface.co/{model}/tree/main/onnx")
    sizes = [i.replace(" ","") for i in sizes]
    sizes = " | ".join(sizes)
    return sizes

# scrape_sizes("mixedbread-ai/mxbai-embed-large-v1")

# %%
df["sizes"] = df["id"].progress_apply(scrape_sizes) # 1 min 19s

# %% [markdown]
# ## Remove models that are not currently working (but have the transformers.js and feature-extraction tag)

# %%
print("Removing following models from dataset: \n",df[df["sizes"] == ""].id)
df = df[df["sizes"] != ""]

# %% [markdown]
# # Add min and max onnx file size for sorting. Must be converted from different units

# %%
import re

# Conversion dictionary
size_conversion = {'Byt': 1, 'Bytes': 1, 'kB': 1024, 'MB': 1024**2, 'GB': 1024**3}

# Conversion function
def size_to_bytes(size_str):
    # Use regex to find the number and unit
    match = re.search(r'(\d+(\.\d+)?)\s*(Byt|Bytes|kB|MB|GB)', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    size_value = float(match.group(1))
    size_unit = match.group(3)
    return size_value * size_conversion[size_unit]

# Parsing and conversion function
def parse_and_find_min_max(sizes_str):
    sizes_list = sizes_str.split(' | ')
    sizes_bytes = [size_to_bytes(s) for s in sizes_list]
    return min(sizes_bytes), max(sizes_bytes)

# Apply the function and create new columns
# Assuming 'df' is a pandas DataFrame and 'sizes' is a column in that DataFrame
df['min_size'], df['max_size'] = zip(*df['sizes'].apply(parse_and_find_min_max))

# %%
#df.sort_values("min_size", ascending=True).head(20) # sort as you please here

# %% [markdown]
# # Removing all sizes below 50kB for the moment to filter 

# %%
df = df[df["min_size"] > 50000].reset_index(drop=True)
df = df.reset_index(drop=True)
df["trending"] = df.index +1 # adding the trending column

# %%
from datetime import datetime

# Get today's date
today = datetime.today().strftime("%d-%m-%Y")

df["mined_date"] = today # append to df so that one could easily concat dfs of different dates and do a groupby or similar, convenience

# %%
df.head(10)

# %% [markdown]
# # Save files 

# %%
df.to_excel(f"data/feature-extraction/transformersjs_{today}.xlsx")
df.to_parquet(f"data/feature-extraction/transformersjs_{today}.parquet")
df.to_csv(f"data/feature-extraction/transformersjs_{today}.csv")
df.to_json(f"data/feature-extraction/transformersjs_{today}.json")

# %% [markdown]
# # To html options (ready to be pasted for SemanticFinder) 

# %%
# Assuming df is your DataFrame
html_options = []

for index, row in df.iterrows():
    # Extracting relevant information from each row
    author = row['author']
    downloads = row['downloads']
    likes = row['likes']
    sizes = row['sizes']
    id = row['id']

    # Creating the option string
    option_str = f'<option value="{id}">{id} | 💾{sizes} 📥{downloads} ❤️{likes}</option>'
    
    # Adding the option to the list
    html_options.append(option_str)

# Joining all options into a single string
html_options_str = '\n'.join(html_options)

with open(f"data/feature-extraction/transformersjs_html_options_{today}.html", 'w') as file:
    file.write(html_options_str)
    

# %% [markdown]
# # To html table with filters/sorting

# %%
# Define the editors for your columns
tabulator_editors = {
    'float': {'type': 'number', 'max': 10, 'step': 0.1},
    'bool': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'str': {'type': 'list', 'valuesLookup': True},
}

# Create the Tabulator widget with header filters
header_filter_table = pn.widgets.Tabulator(
    df, layout='fit_columns',
    editors=tabulator_editors, header_filters=True
)

# Save the widget to HTML with header filters
header_filter_table.save(f"data/feature-extraction/transformersjs_{today}.html")
#df.to_html(index=False) # pandas has not sorting/filtering option


# %% [markdown]
# # Send ntfy notifications

# %%
import requests
import pandas as pd
import json  # Import the json module

# Assume df is your pandas DataFrame

# Format the DataFrame into a Markdown list
list_message = f"Trending HuggingFace Embedding Models - {today}\n"
list_message += f"{df.__len__()} available for feature-extraction in transformers.js:\n\n"

for index, row in df.head(10).iterrows():
    list_message += f"{index + 1}. {row['id']}, Likes: {row['likes']}, Downloads: {row['downloads']}\n Sizes: {row['sizes']}\n\n"

list_message += f"Meta data about all {df.__len__()} models can be downloaded on GitHub as csv, xlsx, json, parquet, html. Models can be downloaded from HuggingFace. Originally designed for SemanticFinder, a web app for in-browser semantic search where you can test all models without installing anything."

# Define the channel name
channel_name = "feature_extraction_transformers_js_models"

# Construct the URL for the ntfy.sh server
url = f"https://ntfy.sh/"

# Prepare the actions as a list of dictionaries
actions = [
    {"action": "view", "label": "GitHub", "url": "https://github.com/do-me/trending-huggingface-models"},
    {"action": "view", "label": "HuggingFace", "url": "https://huggingface.co/models?library=transformers.js&other=feature-extraction&sort=trending"},
        {"action": "view", "label": "SemanticFinder", "url": "https://do-me.github.io/SemanticFinder/"}
]

# Create the payload as a dictionary
payload = {
    "topic": channel_name,
    "message": list_message,
    "actions": actions
}

# Send the notification
response = requests.post(url, json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    print("Notification sent successfully!")
else:
    print(f"Failed to send notification. Status code: {response.status_code}")


# %%



