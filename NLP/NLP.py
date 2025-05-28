import json
import csv
from browseai import Browser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pymorphy2 import MorphAnalyzer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

stemmer = SnowballStemmer("russian")
morph = MorphAnalyzer()
ukrainian_stopwords = set(stopwords.words('russian'))  
custom_stopwords = {'це', 'цей', 'який', 'дуже', 'так', 'тільки', 'можна', 'був', 'була'}
ukrainian_stopwords.update(custom_stopwords)

def process_text(text):
    """Функція для обробки тексту відгуку"""

    tokens = word_tokenize(text.lower(), language='russian')
    
    filtered_tokens = [
        token for token in tokens 
        if token.isalpha() and token not in ukrainian_stopwords
    ]
    
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    lemmatized_tokens = [
        morph.parse(token)[0].normal_form for token in filtered_tokens
    ]
    
    return {
        'original_text': text,
        'tokens': filtered_tokens,
        'stems': stemmed_tokens,
        'lemmas': lemmatized_tokens
    }

def scrape_rozetka_reviews(product_url, max_reviews=100):
    """Функція для парсингу відгуків з Rozetka"""
    browser = Browser()
    browser.launch()
    
    browser.open_url(product_url)
    
    browser.wait_for_element('div.product__reviews', timeout=10)
    
    reviews_data = []
    
    while len(reviews_data) < max_reviews:
        review_elements = browser.find_elements('div.product-review')
        
        for review in review_elements:
            try:
                text_element = review.find_element('div.product-review__body')
                review_text = text_element.text.strip()
                
                if review_text:
                    processed_review = process_text(review_text)
                    reviews_data.append(processed_review)
                    
                    if len(reviews_data) >= max_reviews:
                        break
            except:
                continue
        
        try:
            more_button = browser.find_element('a.product-reviews__pager-link')
            browser.click(more_button)
            browser.wait(2)
        except:
            break
    
    browser.quit()
    return reviews_data

def save_to_json(data, filename):
    """Збереження даних у JSON файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_to_csv(data, filename):
    """Збереження даних у CSV файл"""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Original Text', 'Tokens', 'Stems', 'Lemmas'])
        
        for review in data:
            writer.writerow([
                review['original_text'],
                ', '.join(review['tokens']),
                ', '.join(review['stems']),
                ', '.join(review['lemmas'])
            ])

if __name__ == "__main__":
    product_url = "https://rozetka.com.ua/ua/samsung-sm-a346flghsek/p383796909/"
    
    print("Початок збору відгуків...")
    reviews = scrape_rozetka_reviews(product_url, max_reviews=50)
    
    print(f"Зібрано {len(reviews)} відгуків. Початок обробки...")
    
    save_to_json(reviews, 'rozetka_reviews_processed.json')
    save_to_csv(reviews, 'rozetka_reviews_processed.csv')
    
    print("Обробка завершена. Результати збережено у файли:")
    print("- rozetka_reviews_processed.json")
    print("- rozetka_reviews_processed.csv")