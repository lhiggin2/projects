import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import random

connections_data = {
    "Yellow_Group": [],
    "Green_Group": [],
    "Blue_Group": [],
    "Purple_Group": [],
    "Word_List": [],
    "LLM_Response": []
}

for i in range(1, 126):
    url = "https://connections.swellgarfo.com/nyt/"
    url = url + str(i)
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        words_elements = soup.find_all("script")
        for element in words_elements:
            word_junk = element.get_text()
    else:
        print("Failed to retrieve the webpage")

    try:
        data = json.loads(word_junk)
    except json.JSONDecodeError:
        print("Invalid JSON-like string")

    lists_following_words = []

    if 'props' in data and 'pageProps' in data['props']:
        answers = data['props']['pageProps'].get('answers', [])
        for answer in answers:
            words_list = answer.get('words')
            if words_list:
                lists_following_words.append(words_list)

    # Shuffle the combined list
    word_list = [elem for sublist in zip(*lists_following_words) for elem in sublist]
    random.shuffle(word_list)

    # Assign lists to their respective columns in the dictionary
    connections_data["Yellow_Group"].append(lists_following_words[0])
    connections_data["Green_Group"].append(lists_following_words[1])
    connections_data["Blue_Group"].append(lists_following_words[2])
    connections_data["Purple_Group"].append(lists_following_words[3])
    connections_data["Word_List"].append(word_list)

###############LLM ANSWER CREATION#######################

#Gather imports for calling an llm and getting its prompts
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#Set up api key from HuggingFace
os.environ["OPENAI_API_KEY"] = "INSERT API KEY HERE"

#Create prompt template
word_template=PromptTemplate(
    input_variables=['word_list'],
    template="Let's play a game! Here are the rules. You'll be given 16 words.\
        In the list, find groups of four distinct words that share something in common\
        For example: Bass, Flounder, Salmon, Trout are all fish, so\
        Bass, Flounder, Salmon, and Trout belong in a group!\
        Another example: Ant, Drill, Island, Opal all are of the form 'Fire ___' so\
        Ant, Drill, Island, and Opal belong in a group too!\
        Groups will always be more specific than\
        '5-letter words' or 'NAMES' or 'VERBS'.\
        Each word list has exactly one solution.\
        Watch out for words in the list that seem to belong\
        to multiple groups!\
        Play the game with this list of words:{word_list}\
        I want exactly one group of four distinct words from the provided list\
        that you think belongs to a group according to the game.\
        Please write your answer as\
        only the four words you choose. Don't repeat any words in your response!\
        No additional words are needed in your response!"
)

#Initiate Some llm that can handle this simple question...Life is hard
#llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
llm=ChatOpenAI(temperature=0, model="gpt-4")

for i in range (125):
    #Chain llm to prompt template
    prompt_chain=LLMChain(llm=llm,prompt=word_template)
    
    #Get prompt (list of words) to plug into our word_template
    prompt=connections_data['Word_List'][i]
    
    print(prompt_chain.run(word_list=prompt))
    
    connections_data["LLM_Response"].append(prompt_chain.run(word_list=prompt))

#############################################################################

# Convert the dictionary into a DataFrame
connections_df = pd.DataFrame(connections_data)

#Save to local computer to save API usage costs.
connections_df.to_csv("connections_df.csv",index=False)

###################################Test for Accuracy##########################
#Make 'LLM_Response' be a list of four words as opposed to a single string.
connections_df["LLM_Response"]=connections_df["LLM_Response"].str.split(", ")

#Create function that will check to see if GPT's answer is correct.
def correct(k):
    if set(connections_df["LLM_Response"][k]) == set( connections_df["Yellow_Group"][k] ):
        return True
    elif set(connections_df["LLM_Response"][k]) == set( connections_df["Green_Group"][k] ):
        return True
    elif set(connections_df["LLM_Response"][k]) == set( connections_df["Blue_Group"][k] ):
        return True
    elif set(connections_df["LLM_Response"][k]) == set( connections_df["Purple_Group"][k] ):
        return True
    else: return False
    
def remove_apostrophes(word_list):
    result_list = []
    for word in word_list:
        result_list.append(word.replace("'", ''))
    return result_list

connections_df["LLM_Response"]=connections_df["LLM_Response"].apply(remove_apostrophes)

new_column=[]
for index, row in connections_df.iterrows():
    new_boolean=( correct(index) )
    new_column.append(new_boolean)

connections_df["Correct"]=new_column
connections_df.to_csv("connections_df_complete.csv")

connections_df["Correct"].sum()
#Got 61/125 correct.