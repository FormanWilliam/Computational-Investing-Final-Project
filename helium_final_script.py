## Computational Investing: Helium Blockchain Data Analysis
## Will Forman, Cam Rogers, Grace Rego, Greg Toudouze, Matt Garesche

# Importing Packages

import requests
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Functions to scrape Helium API Data

# Only call requests.get() on a particular link once and then wait a while, can't handle too many API requests in a short period of time

headers = {
    'authority': 'api.helium.io',
    'cache-control': 'max-age=0',
    'sec-ch-ua': '"Chromium";v="94", "Google Chrome";v="94", ";Not A Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'dnt': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'none',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'sec-fetch-dest': 'document',
    'accept-language': 'en-US,en;q=0.9',
    }

# Scrape general hotspot information (maximum of 1000 hotspots)

def scrape_helium_hotspots():
  response = requests.get('https://api.helium.io/v1/hotspots', headers=headers, params=params)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    hotspot_data = data['data'][i]
    rows.append(hotspot_data)
    
  df = pd.DataFrame(rows)

  df = pd.concat([df.drop(['status', 'geocode'], axis=1), pd.json_normalize(df['status']), pd.json_normalize(df['geocode'])], axis=1)

  return df

# Scrape general block descriptions like height, hash, transaction count (maximum of 1000 blocks)

def scrape_helium_blocks():

  response = requests.get('https://api.helium.io/v1/blocks', headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    block_data = data['data'][i]
    rows.append(block_data)
    
  df = pd.DataFrame(rows)

  return df

# Scrape city information like city id, state, country, hotspot count

def scrape_cities():

  response = requests.get('https://api.helium.io/v1/cities', headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    city_data = data['data'][i]
    rows.append(city_data)
    
  df = pd.DataFrame(rows)

  df = df[["city_id", "long_city", "long_state", "long_country", "online_count", "offline_count", "hotspot_count"]]

  return df

# Scrape hotspot information from all hotspots in a given city

def scrape_hotspots_by_city(city_id):
  response = requests.get('https://api.helium.io/v1/cities/' + city_id + '/hotspots', headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    city_data = data['data'][i]
    rows.append(city_data)

  df = pd.DataFrame(rows)
  
  df = pd.concat([df.drop(['status', 'geocode'], axis=1), pd.json_normalize(df['status']), pd.json_normalize(df['geocode'])], axis=1)
    
  return df

# For scraping network rewards, you can specify how far back in time you'd like to go, and how you'd like the stats to be bucketed (time_frame)
# General format (anything in parentheses can be modified by user): ?min_time=-(NUMBER)%20(GROUPING)&bucket=(GROUPING)
# The grouping can be day, week, month, year, etc.
# E.g. if I wanted the network rewards from the past 3 weeks grouped by day: ?min_time=-3%20week&bucket=day
# Pass the time_frame argument in as a string

def scrape_network_rewards(time_frame):

  response = requests.get('https://api.helium.io/v1/rewards/sum/'+ 
                          str(time_frame), headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    reward_data = data['data'][i]
    rows.append(reward_data)
    
  df = pd.DataFrame(rows)

  return df

# Get list of accounts with addresses, data credit balance, nonce, etc.

def scrape_accounts():

  response = requests.get('https://api.helium.io/v1/accounts', headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    acct_data = data['data'][i]
    rows.append(acct_data)
    
  df = pd.DataFrame(rows)

  return df

# Get hotspot information for all hotspots under a specific account

def scrape_hotspots_by_account(address):

  response = requests.get('https://api.helium.io/v1/accounts/' + address + '/hotspots', headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    acct_data = data['data'][i]
    rows.append(acct_data)
    
  df = pd.DataFrame(rows)

  return df

# Fetch challenges that a given account's hotspots are involved in (challenger, challengee, witness)

def scrape_account_challenges(address):

  response = requests.get('https://api.helium.io/v1/accounts/' + address + '/challenges', headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    acct_data = data['data'][i]
    rows.append(acct_data)
    
  df = pd.DataFrame(rows)

  return df

# Scrape hotspot information of those in a given location range
# Format of loc: "?lat=___&lon=___&distance=___"

def scrape_hotspots_by_loc(loc):

  response = requests.get('https://api.helium.io/v1/hotspots/location/distance/' + loc, headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    hotspot_loc_data = data['data'][i]
    rows.append(hotspot_loc_data)
    
  df = pd.DataFrame(rows)

  df = pd.concat([df.drop(['status', 'geocode'], axis=1), pd.json_normalize(df['status']), pd.json_normalize(df['geocode'])], axis=1)

  return df

# Get a hotspot's total rewards in HNT over a given time frame (same time frame format as before)

def scrape_hotspot_rewards(address, time_frame):

  response = requests.get('https://api.helium.io/v1/hotspots/' + address + '/rewards/sum/' + time_frame, headers=headers)
  data = response.json()

  df = pd.DataFrame([data['data']])

  return df["sum"].iloc[0]/100000000

# Return the total number of data credits burned for the entire network in a given time frame

def scrape_dc_burns(time_frame):
  response = requests.get('https://api.helium.io/v1/dc_burns/sum/' + time_frame, headers=headers)
  data = response.json()

  rows = []
  for i in range(len(data['data'])):
    hotspot_loc_data = data['data'][i]
    rows.append(hotspot_loc_data)
    
  df = pd.DataFrame(rows)

  return df




### EDA ###


# Hotspots by country (learning experience)

#the first 2 lines just print the entire df that was pulled
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Get city information

cities = scrape_cities()

city_names = hotspot_df[["city_id", "long_city", "short_city"]].drop_duplicates()

# Number of hotspots by state/country

state_hotspot_counts = pd.DataFrame(cities.groupby(["long_state", "long_country"])["hotspot_count"].agg("sum")).sort_values("hotspot_count", ascending=False).reset_index()

plt.rcdefaults()
fig, ax = plt.subplots()

# Filtering for states/countries with 1 or more hotspots

y_pos = state_hotspot_counts[state_hotspot_counts["hotspot_count"] > 1]["hotspot_count"]
print (state_hotspot_counts["hotspot_count"])
hotspots = state_hotspot_counts [state_hotspot_counts["hotspot_count"] > 1] ["long_country"]

ax.barh(hotspots, y_pos, align='center')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(' Number of Hotspots')
ax.set_title('Hotspots per Country')


#1. DATA COLLECTION -- WITNESS

import time as t
import itertools as ite

# Grab hotspots and return their number of witnesses
# Given our hotspot_df dataframe, the start and end parameters determine what hotspots (indices)
# to query for

def hotspot_collector(start, end):
    i = 0
    while i != 11:
        print("HERE")
        #1. new col. for number of witnesses
        for i in range(start, end +1):
            address = hotspot_df.loc[i, "address"]
            witness_number = len(list_witnesses_for_hotspot(hotspot_address = address))
            hotspot_df.loc[i, "witness"] = witness_number
        
        start = start + 10
        end = end + 10
        if end > 1000:
            break
        else:
            print("sleep now ... zzz")
            t.sleep(93)
            i += 1
    return(hotspot_df["witness"])

#hotspot_collector(129,132)
#hotspot_df.to_excel('Witness2.xlsx')

# Scrape witnesses for a hotspot

def list_witnesses_for_hotspot(hotspot_address: str):
    """lists a hotspot's witnesses over the last 5 days"""
    url = 'https://api.helium.io/v1/hotspots/' + hotspot_address + '/witnesses'
    r = requests.get(url)
    witnesses = r.json()['data']
    return witnesses


#2. LOOK AT THE DATA

def classify(df):
    sorted_hotspot = hotspot_df.sort_values(by =["witness"])
    sorted_hotspot["class"] = ""
    
    list1 = list(ite.repeat("low", 500))
    list2 = list(ite.repeat("high", 500))
    final_l = list1 + list2
    
    sorted_hotspot["class"]=final_l
    hsdf = sorted_hotspot
    
    return(hsdf)
    
hsdf = classify(hotspot_df)


#3. PLOT THE DATA
#Witness Histogram

hsdf["witness"] = pd.to_numeric(hsdf["witness"])
hsdf.hist(column = "witness")
plt.xlabel("Witness Total")
plt.ylabel("Frequency")




# Witness & Reward Scale Scatterplot

import seaborn as sns

sns.lmplot(x='witness',y='reward_scale',data=hsdf,fit_reg=True, truncate = False)
plt.xlim(-10, 80)
r = hsdf["witness"].corr(hsdf["reward_scale"])
print(r)

sns.lmplot(x='witness',y='reward_scale',data=hsdf, hue = "class", fit_reg=True)
plt.xlim(-10,80)

#Pos vs. neg witness hist. 
posdf = hsdf.iloc[766:, :]
zerodf = hsdf.iloc[0:766, :]

# Reward scale distribution for hotspots with 1 or more witnesses

fig, axes = plt.subplots(1, 2)
posdf.hist('reward_scale', bins=20, ax=axes[0], range = [0,1], sharex = True, sharey = True)
axes[0].set_title("Positive Witness Numbers")

# Reward scale distribution for hotspots with zero witnesses

plt.ylim(0.50)
zerodf.hist('reward_scale', bins=20, ax=axes[1], range = [0,1], sharex = True, sharey = True)
axes[0].set_ylim(0,60)
axes[1].set_title("Zero Witness")

plt.ylim(0,60)

plt.xlabel("Reward Scale")
plt.ylabel("Frequency")

# Function to encode whether or not a hotspot has witnesses

def classify_pos(df):
    hsdf["pos_0"] = ""
    list1 = list(ite.repeat("zero", 33))
    list2 = list(ite.repeat("pos", 86))
    final_l = list1 + list2
    
    hsdf["pos_0"]=final_l
    
    return(hsdf)
    
hsdf1 = classify_pos(hsdf)

# Plot of number of witnesses vs. reward scale for hotspots with 1 or more witnesses

sns.lmplot(x='reward_scale',y='witness',data=posdf, fit_reg=True)
rpos = posdf["witness"].corr(posdf["reward_scale"])
plt.title("Positive Witness Data")
plt.show()
print("r: ", rpos)



# Search for hotspots in a particular area and get their rewards
# ~ 5 mile radius if possible

def get_reward_summary_by_location(loc):
  hotspot_df = scrape_hotspots_by_loc(loc)
  reward_totals = []
  # Take random sample of 10 hotspots in the area if there are too many to loop through
  if hotspot_df.shape[0] > 10:
    rand_idx = [random.randint(0, hotspot_df.shape[0]-1) for _ in range(10)]
    hotspot_df = hotspot_df.iloc[rand_idx, :]
  for i in hotspot_df["address"]:
    reward_totals.append(scrape_hotspot_rewards(i, "?min_time=-32%20day&max_time=-3%20day"))
  reward_totals = np.array(reward_totals)
  print(reward_totals)
  return reward_totals


# First Example: Goodland, KS (sparse, but several hotspots near each other)
# Latitude: 39.356602, Longitude: -101.711082, 5 mi radius = 8046.72 meters

goodland_rewards = get_reward_summary_by_location("?lat=39.356602&lon=-101.711082&distance=8000")

goodland_rewards # Average of 0.74724 total per hotspot over past month (will change if run again)

# Next Example: West Hollywood, CA (very dense)
# Latitude: 34.090698, Longitude: -118.386002, 5 mi radius = 8046.72 meters

# Had to split this up into multiple runs in order to avoid 429 errors

westholly_rewards = get_reward_summary_by_location("?lat=34.090698&lon=-118.386002&distance=8000")
westholly_rewards2 = get_reward_summary_by_location("?lat=34.090698&lon=-118.386002&distance=8000")
westholly_rewards3 = get_reward_summary_by_location("?lat=34.090698&lon=-118.386002&distance=8000")

np.mean(np.concatenate([westholly_rewards, westholly_rewards2, westholly_rewards3])) 
# Average of 2.1782 total per hotspot over past month (will change if run again)


# Data Credit Burns for the entire network (over a 20 week span)

burns = scrape_dc_burns("?min_time=-20%20week&bucket=week")

burns = burns.iloc[::-1] # Arranging by date ascending

burns["time"] = pd.date_range(start="2021-07-24", end="2021-12-04", freq="W")

# Converting data credit totals to per-billion units

burns["total_billions"] = burns["total"]/1000000000

# Plotting time series of data credit burns

sns.set_style("whitegrid")
sns.lineplot(x="time", y="total", data=burns, color="green")
plt.xticks(rotation=60)
plt.xlabel("Date")
plt.ylabel("DC Burned (in billions)")
plt.title("Data Credits Burned Over Past 20 Weeks")
plt.show()




# Scrape hotspots by city


## Tuscon, AZ

tucson = scrape_hotspots_by_city("dHVjc29uYXJpem9uYXVuaXRlZCBzdGF0ZXM")

tucson["reward_scale"].mean() # 0.7217456237662272, pretty good density (around optimal amount of hotspots)

# Filtering for hotspots that were added at least a month ago

tucson = tucson.query("timestamp_added <= '2021-11-30T23:00:00.000000Z'").reset_index()

tucson["reward_sum"] = None

# Computing one-day reward totals for hotspots
# Since there were so many hotspots, we ran into a lot of 429 errors
# So, our solution was to brute-force it and re-start the loop each time we got an error

for i in range(tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 20

for i in range(20, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 56

for i in range(56, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 79

for i in range(79, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 165

for i in range(165, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 193

for i in range(193, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 216

for i in range(216, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 249

for i in range(249, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 256

for i in range(256, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 265

for i in range(265, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 289

for i in range(289, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 357

for i in range(357, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 373

for i in range(373, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 379

for i in range(379, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 479

for i in range(479, tucson.shape[0]):
  print(i)
  tucson["reward_sum"][i] = scrape_hotspot_rewards(tucson["address"][i], "?min_time=-1%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)


tucson.to_csv("tucson.csv")

# Filtering out null reward totals

tucson = tucson[tucson.reward_sum.notnull()]


## Elevation vs. Rewards

tucson = pd.read_csv("tucson.csv")

np.corrcoef(tucson["elevation"], tucson["reward_sum"]) # 0.33003279

# Filtering out large outliers for elevation

tucson_small = tucson[tucson["elevation"] <= 15]

tucson_small = tucson_small[tucson_small.reward_scale.notnull()]

# Reward Scale Groups

conditions = [(tucson_small["reward_scale"].le(0.3)),
              (tucson_small["reward_scale"].gt(0.3) & tucson_small["reward_scale"].le(0.6)),
              (tucson_small["reward_scale"].gt(0.6))]
choices = ["Low", "Medium", "High"]
tucson_small["scale_group"] = np.select(conditions, choices)


np.corrcoef(tucson_small["elevation"], tucson_small["reward_sum"]) # 0.3806523 (not extremely strong but positive)

np.corrcoef(tucson_small[tucson_small["scale_group"]=="Low"]["elevation"],
            tucson_small[tucson_small["scale_group"]=="Low"]["reward_sum"]) # 0.30261618
np.corrcoef(tucson_small[tucson_small["scale_group"]=="Medium"]["elevation"],
            tucson_small[tucson_small["scale_group"]=="Medium"]["reward_sum"]) # 0.41553096
np.corrcoef(tucson_small[tucson_small["scale_group"]=="High"]["elevation"],
            tucson_small[tucson_small["scale_group"]=="High"]["reward_sum"]) # 0.37366763


# Plot with outliers

sns.set_style("whitegrid")
sns.lmplot(x="elevation", y="reward_sum", data=tucson)
plt.xlabel("Elevation")
plt.ylabel("Reward Total (12/3/21)")
plt.title("Tucson Hotspot Elevation Levels vs. Past Day Rewards")

# Plot without outliers

sns.set_style("whitegrid")
sns.lmplot(x="elevation", y="reward_sum", data=tucson_small)
plt.xlabel("Elevation")
plt.ylabel("Reward Total (12/3/21)")
plt.title("Tucson Hotspot Elevation Levels vs. Past Day Rewards")

# Plot grouped by reward scale

sns.set_style("whitegrid")
sns.lmplot(x="elevation", y="reward_sum", hue="scale_group", data=tucson_small)
plt.xlabel("Elevation")
plt.ylabel("Reward Total (12/3/21)")
plt.legend(title = "Reward Scale Group")
plt.title("Tucson Hotspot Elevation Levels vs. Past Day Rewards")

# Regression of Reward total vs. Elevation, controlling for (log) reward scale

# Tucson

import statsmodels.api as sm

X = sm.add_constant(tucson_small[["elevation", "reward_scale"]])
Y = tucson_small["reward_sum"]
est = sm.OLS(Y, X)
est2 = est.fit()
print(est2.summary())


## Toronto (same process as above)

toronto = scrape_hotspots_by_city("dG9yb250b29udGFyaW9jYW5hZGE")

toronto["reward_scale"].mean() # 0.18699078117449258, poor density (too many hotspots)

toronto = toronto.query("timestamp_added <= '2021-11-30T23:00:00.000000Z'").reset_index()

toronto["reward_sum"] = None

# Don't be alarmed by the amount of for-loops: we had to brute-force this aspect of the code
# because there were so many hotspots to iterate through, and the API did not allow us
# to make too many requests in a limited amount of time

for i in range(toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 9

for i in range(9, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Restart at index 80

for i in range(80, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at index 93

for i in range(93, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 103

for i in range(103, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 115

for i in range(115, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 121

for i in range(121, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 158

for i in range(158, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 191

for i in range(191, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 202

for i in range(202, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 215

for i in range(215, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 290

for i in range(290, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 306

for i in range(306, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 339

for i in range(339, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 350

for i in range(350, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 351

for i in range(351, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 352

for i in range(352, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 374

for i in range(374, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 403

for i in range(403, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 413

for i in range(413, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 431

for i in range(431, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 466

for i in range(466, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 489

for i in range(489, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 500

for i in range(500, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 507

for i in range(507, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 512

for i in range(512, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 523

for i in range(523, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 526

for i in range(526, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 566

for i in range(566, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 579

for i in range(579, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 595

for i in range(595, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 626

for i in range(626, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 664

for i in range(664, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Restart at 700

for i in range(700, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 710

for i in range(710, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 743

for i in range(743, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 768

for i in range(768, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 779

for i in range(779, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 800

for i in range(800, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 809

for i in range(809, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 817

for i in range(817, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 820

for i in range(820, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 845

for i in range(845, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 858

for i in range(858, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# Failed at 863

for i in range(863, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# 880 restart

for i in range(880, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)

# 906 restart

for i in range(906, toronto.shape[0]):
  print(i)
  toronto["reward_sum"][i] = scrape_hotspot_rewards(toronto["address"][i], "?min_time=-4%20day")
  if i != 0 and i%10 == 0:
    time.sleep(93)


toronto.to_csv("toronto.csv")


## Elevation vs. Rewards

toronto = pd.read_csv("toronto.csv")

np.corrcoef(toronto["elevation"], toronto["reward_sum"]) # 0.14866822

toronto_small = toronto[toronto["elevation"] <= 200]

toronto_small = toronto_small[toronto_small.reward_scale.notnull()]

# Reward Scale Groups

conditions = [(toronto_small["reward_scale"].le(0.3)),
              (toronto_small["reward_scale"].gt(0.3) & toronto_small["reward_scale"].le(0.6)),
              (toronto_small["reward_scale"].gt(0.6))]
choices = ["Low", "Medium", "High"]
toronto_small["scale_group"] = np.select(conditions, choices)

toronto_small[["reward_scale", "scale_group"]]

np.corrcoef(toronto_small["elevation"], toronto_small["reward_sum"]) # 0.15390654

np.corrcoef(toronto_small[toronto_small["scale_group"]=="Low"]["elevation"],
            toronto_small[toronto_small["scale_group"]=="Low"]["reward_sum"]) # 0.19718502
np.corrcoef(toronto_small[toronto_small["scale_group"]=="Medium"]["elevation"],
            toronto_small[toronto_small["scale_group"]=="Medium"]["reward_sum"]) # 0.31593388
np.corrcoef(toronto_small[toronto_small["scale_group"]=="High"]["elevation"],
            toronto_small[toronto_small["scale_group"]=="High"]["reward_sum"]) # -0.20376542

# Plot with outliers

sns.set_style("whitegrid")
sns.lmplot(x="elevation", y="reward_sum", data=toronto)
plt.xlabel("Elevation")
plt.ylabel("Reward Total (12/3/21)")
plt.title("Toronto Hotspot Elevation Levels vs. Past Day Rewards") # outlier at 500

# Plot without outliers

sns.set_style("whitegrid")
sns.lmplot(x="elevation", y="reward_sum", data=toronto_small)
plt.xlabel("Elevation")
plt.ylabel("Reward Total (12/3/21)")
plt.title("Toronto Hotspot Elevation Levels vs. Past Day Rewards")

# Plot colored by reward scale groups

sns.set_style("whitegrid")
sns.lmplot(x="elevation", y="reward_sum", hue="scale_group", data=toronto_small)
plt.xlabel("Elevation")
plt.ylabel("Reward Total (12/3/21)")
plt.legend(title = "Reward Scale Group")
plt.title("Toronto Hotspot Elevation Levels vs. Past Day Rewards")

# Regression of Reward total vs. Elevation, controlling for (log) reward scale

# Toronto

import statsmodels.api as sm

toronto_small["log_reward_scale"] = np.log(toronto_small["reward_scale"])

X = sm.add_constant(toronto_small[["elevation", "log_reward_scale"]])
Y = toronto_small["reward_sum"]
est = sm.OLS(Y, X)
est2 = est.fit()
print(est2.summary())

np.corrcoef(toronto_small["elevation"], toronto_small["log_reward_scale"]) # -0.38





# Simple example: looking at a hexagon near BC and observing how those hotspots have been faring

# Specific hotspot addresses

bc_addresses = ["112LtKYN2c3Qy4GH7dbecTfGW8RNFaET6g4iToog3fFayduMJ8qx",
"112oZ6sBqSLStg8vRcXLkQMGkzdWK8WkCqYsqu3BoF4xYkATYYa7",
"112GqcBXtmgDGPZN1rhxLAnmNpgt18VAT4d3dEoJ9du6shBb7yKa",
"11GMnuKam29omnGTRssuBP5nNrSM5iQL4ZyDbyrq8RBLbcKBVFY",
"11rCSYRjGrgEBGUuYPVNhsuaEp2bnderu4SBVJxvHgfYxv7H9sg",
"11sHzBUwyuXL6h3MYrsPYQucanFujcBy32AdHWqW6UAUNtwuPW4",
"11Qu1REHNzE3bU123DiepvhKYczAfK6uLYwzRk148ENBWZuGKQP",
"11sXAVwUaatWJnGMu4HQjPnr7V8Z8ns1upatkgKzwwiNV4obCzh",
"11vqS7YCyeg78WfWCztimDuJ9s1E49zzPgiDB6swWteN8fG3fuE",
"11KtFhfGhAbez99HWp39YU5YbMFaE3RXpHhQt2yNEBmYMkEQJcc",
"118WKcbm76cpUKqefSnSV5gm1D2VZfn8Z157fNENYzJEi2Tw4sZ"]

# Getting reward data for each hotspot per day over the past month

reward_dfs = []

for address in bc_addresses:
  print(address)
  response = requests.get('https://api.helium.io/v1/hotspots/' + address + '/rewards/sum/' + "?min_time=-30%20day&bucket=day", headers=headers)
  data = response.json()
  df = pd.DataFrame(data['data'])
  df["address"] = address
  reward_dfs.append(df)

bc_rewards = pd.concat(reward_dfs, axis=0)

# Get an average per day, scale by 30 to get a monthly rate

np.mean(bc_rewards[bc_rewards["total"] != 0]["total"])
np.mean(bc_rewards[bc_rewards["total"] != 0]["total"])*30

# About 4.922117822784809 HNT earned per month

