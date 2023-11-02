import xml.etree.ElementTree as ET
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader
from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences


def get_components(sense_key):
    components = sense_key.split('%')
    lemma = components[0]
    info = components[1].split(':')
    part_of_speech = info[0]
    sense_number = info[1]
    lexicographer_file = info[2]
    return lemma, part_of_speech, sense_number, lexicographer_file
def xml_to_txt():
    # Load the XML content from your file (assuming it's named 'input.xml')
    tree = ET.parse('senseval2.data.xml')
    root = tree.getroot()

    text = []
    for sentence in root.findall(".//sentence"):
        sentence_text = " ".join(sentence.itertext())
        sentence_text = sentence_text.replace('\n', '')
        text.append(sentence_text)

    text2 = []
    for sent in text:
        cleaned_text = ' '.join(sent.split())
        text2.append(cleaned_text)
    final_text = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for sent in text2:
        f = word_tokenize(sent)
        f = [word.lower() for word in f if word.isalnum() and word.lower()]
        final_text.append(f)
    final_text
    return final_text
def senseval_prepros():
    words = []
    index = 0
    with open('senseval2.gold.key.txt', 'r') as infile:
            # Read the lines from the input file
            lines = infile.readlines()

    # Open the output file for writing
    with open('lemmas.txt', 'w') as outfile:
        for line in lines:
            stripped_line = line[14:].strip()
            lemma = get_components(stripped_line)
            outfile.write(str(lemma[0]) + '\n')
def model_upload():
    # Initialize Word2Vec model
    #model = Word2Vec(sentences=final_text, vector_size=100, window=5, min_count=1, workers=4)
    #
    # ##################################################
    glove_vectors = gensim.downloader.load('word2vec-google-news-300')
    # Save the model
    #model.save("word2vec.model")
    return glove_vectors
def create_table(similarity_data):
    # Filter out None values from similarity_data
    similarity_data = [item for item in similarity_data if item[1] is not None]

    # Sort the filtered list by the second element (similarity score) in reverse order
    similarity_data.sort(key=lambda x: x[1], reverse=True)

    # Select the top 10 and least 10 items
    top_10_lemmas = []
    top_10_sim = []
    least_10_lemmas = []
    least_10_sim = []

        # Extract the top 10 unique lemmas with their similarity scores
    for lemma, similarity in similarity_data:
        if len(top_10_lemmas) < 10 and lemma not in top_10_lemmas:
            top_10_lemmas.append(lemma)
            top_10_sim.append(similarity)

    # Extract the least 10 unique lemmas with their similarity scores
    for lemma, similarity in reversed(similarity_data):
        if len(least_10_lemmas) < 10 and lemma not in least_10_lemmas:
            least_10_lemmas.append(lemma)
            least_10_sim.append(similarity)
    plt.plot(top_10_lemmas, top_10_sim, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)

    ######################################################

    # naming the x axis
    plt.xlabel('Words')
    # naming the y axis
    plt.ylabel('Similarity score')

    # giving a title to my graph
    plt.title('Top 10 similarities to word "woman"!')
    ######################################################
    # function to show the plot
    plt.show()

    plt.plot(least_10_lemmas, least_10_sim, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
    plt.xlabel('Words')
    # naming the y axis
    plt.ylabel('Similarity score')

    # giving a title to my graph
    plt.title('Least 10 similarities to word "woman"!')

    # function to show the plot
    plt.show()
def fast_text_vectors(final_text):
    model2 = FastText(vector_size=4, window=3, min_count=1, sentences=final_text, epochs=10)
    word_vectors = model2.wv
    return word_vectors
def convex_comb(word1, word2, glove_vectors, fasttext_vectors):
    glove_weight=0.3
    try:
        glove_vector1 = glove_vectors[word1]
        fasttext_vector1 = fasttext_vectors[word1]

        glove_vector2 = glove_vectors[word2]
        fasttext_vector2 = fasttext_vectors[word2]

        # Normalize the vectors if needed
        glove_vector1 = glove_vector1 / np.linalg.norm(glove_vector1)
        fasttext_vector1 = fasttext_vector1 / np.linalg.norm(fasttext_vector1)
        glove_vector2 = glove_vector2 / np.linalg.norm(glove_vector2)
        fasttext_vector2 = fasttext_vector2 / np.linalg.norm(fasttext_vector2)

        # Handle different dimensions
        # Pad or truncate vectors to the same dimension
        max_dim = max(len(glove_vector1), len(fasttext_vector1), len(glove_vector2), len(fasttext_vector2))

        glove_vector1 = np.pad(glove_vector1, (0, max_dim - len(glove_vector1)))
        fasttext_vector1 = np.pad(fasttext_vector1, (0, max_dim - len(fasttext_vector1)))
        glove_vector2 = np.pad(glove_vector2, (0, max_dim - len(glove_vector2)))
        fasttext_vector2 = np.pad(fasttext_vector2, (0, max_dim - len(fasttext_vector2)))

        # Calculate the convex combination
        combined_vector1 = glove_weight * glove_vector1 + (1 - glove_weight) * fasttext_vector1
        combined_vector2 = glove_weight * glove_vector2 + (1 - glove_weight) * fasttext_vector2

        # Calculate the cosine similarity between the combined vectors
        similarity = cosine_similarity([combined_vector1], [combined_vector2])[0][0]

        #print(f"Cosine Similarity between '{word1}' and '{word2}' after combination: {similarity}")
        return similarity
    except KeyError:
        print("One of the words is not in the pretrained vectors.")
def main():
    word1 = 'man'
    word2 = 'woman'
    final_text = xml_to_txt()
    #senseval_prepros()
    similarity_data = []
    glove_vectors = model_upload()
    fasttext_vectors = fast_text_vectors(final_text)
    #convex_comb(word1, word2, glove_vectors, fasttext_vectors)


    with open('lemmas_original.txt', 'r') as file:
        # Read the file and split it into lines
        lines = file.read().splitlines()

    vector1 = ''
    for j in range(5):
        for i in lines:
        # Get the word vectors for the words
            try:
                similarity = convex_comb(i, word2, glove_vectors, fasttext_vectors)
            except KeyError:
                print("One of the words is not in the pretrained vectors.")
            similarity_data.append((i, similarity))

            # Print the similarity score
            #print(f"Similarity between '{i}' and '{word2}': {similarity}")
    create_table(similarity_data)
    print("***************************")
if __name__ == "__main__":
    main()
