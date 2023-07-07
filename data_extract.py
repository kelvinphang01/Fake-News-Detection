import os
import pandas as pd
import urllib.parse
import json as js

# Main function to extract final dataset
def main():
    root_dir = "./code/fakenewsnet_dataset"
    output_path = "./final_dataset.csv"
    
    extractor = Extractor(root_dir)
    
    global final_dataset # For convenience
    final_dataset = extractor.extract_features()
    final_dataset.to_csv(output_path, index=False)

class Extractor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    # Function to call load_folder and load_json over the loop
    def load_dataset(self):
        folder = []
        json = []
        
        # Loop through gossipcop/politifact _ fake/real
        for source in ["gossipcop", "politifact"]:
            for label in ["fake", "real"]:
                folder_path = os.path.join(self.root_dir, source, label)
                for subfolder in os.listdir(folder_path):
                    self.load_folder(folder, subfolder, label)
                    self.load_json(json, folder_path, subfolder)
                    
        return folder, json
    
    # Load folders (news) downloaded from FakeNewsNet
    def load_folder(self, folder, subfolder, label):
        folder.append({
            "id": subfolder,
            "fakenews": 1 if label == "fake" else 0
            })
    
    # Load JSON files (tweets) in the folders
    def load_json(self, json, folder_path, subfolder):
        subfolder_path = os.path.join(folder_path, subfolder)
        tweets_folder_path = os.path.join(subfolder_path, "tweets")
        
        for file_name in os.listdir(tweets_folder_path):
            file_path = os.path.join(tweets_folder_path, file_name)
            with open(file_path, "r") as file:
                json_content = js.load(file)
                json_content['news_id'] = subfolder
                json.append(json_content)
    
    # Load and merge news datasets
    def load_news(self):    
        file_paths = [
            "./dataset/politifact_fake.csv",
            "./dataset/politifact_real.csv",
            "./dataset/gossipcop_fake.csv",
            "./dataset/gossipcop_real.csv"
        ]
        
        news_df = pd.concat([pd.read_csv(file_path) for file_path in file_paths])
        
        # Reset the index of the merged_df DataFrame
        news_df = news_df.reset_index(drop=True)
        
        return news_df

    # Function to call the mini extraction functions
    def extract_features(self):
        folder, json = self.load_dataset()
        news = self.load_news()
        
        folder = pd.DataFrame(folder)
        
        extracted_data = []
        
        # Loop through JSON files
        for row in json:
            user = row['user']
            extracted_row = {}
            self.extract_news_features(extracted_row, row, folder, news)
            self.extract_other_features(extracted_row, row, user)
            self.extract_derived_features(extracted_row, row, user)
            
            extracted_data.append(extracted_row)
        
        final_dataset = pd.DataFrame(extracted_data)
        return final_dataset
    
    def extract_news_features(self, extracted_row, row, folder, news):
        extracted_row['news_id'] = news_id = row['news_id']
        extracted_row['fakenews'] = folder['fakenews'].loc[folder['id'] == news_id].iloc[0]
        extracted_row['title_words'] = len(str(news['title'].loc[news['id'] == news_id].iloc[0]).split())
        
        if row['entities']['urls']:
            url = row['entities']['urls'][0]['expanded_url']
            extracted_row['url_protocol'] = urllib.parse.urlparse(url).scheme
            extracted_row['url_level'] = len(urllib.parse.urlparse(url).path.split('/'))
            extracted_row['url_www'] = 1 if 'www' in urllib.parse.urlparse(url).netloc else 0
        
    def extract_other_features(self, extracted_row, row, user):
        keys_to_extract = ['favorite_count', 'retweet_count', 'followers_count', 'friends_count', 'listed_count',
                             'favourites_count', 'statuses_count', 'verified']
        
        for key in keys_to_extract:
            extracted_row[key] = user[key] if key in user else row[key]    
    
    def extract_derived_features(self, extracted_row, row, user):        
        if user['friends_count'] == 0:
            extracted_row['has_following'] = 0
            extracted_row['follower_following_ratio'] = 0
        else:
            extracted_row['has_following'] = 1
            extracted_row['follower_following_ratio'] = user['followers_count'] / user['friends_count']
        
        extracted_row['description_length'] = len(user['description'])
        extracted_row['tweet_wordcount'] = len(row['text'].split())
        extracted_row['tweet_hashtags'] = len(row['entities']['hashtags'])
        extracted_row['tweet_mentions'] = len(row['entities']['user_mentions'])

        # Retrieve user timeline tweets        
        user_id = user['id_str']
        json2_path = os.path.join(self.root_dir, "user_timeline_tweets", f"{user_id}.json")
        
        if os.path.isfile(json2_path):
            with open(json2_path, "r") as file:
                json2 = js.load(file)
            
            recent_favorite = sum(json2[i]['favorite_count'] for i in range(len(json2)))
            recent_retweet = sum(json2[i]['retweet_count'] for i in range(len(json2)))
            recent_contains_url = sum(1 if len(json2[i]['entities']['urls']) > 0 else 0 for i in range(len(json2)))
        else:
            recent_favorite, recent_retweet, recent_contains_url = 0,0,0
        
        extracted_row['recent_favorite'] = recent_favorite
        extracted_row['recent_retweet'] = recent_retweet
        extracted_row['recent_contains_url'] = recent_contains_url

# Call main function
if __name__ == "__main__":
    main()