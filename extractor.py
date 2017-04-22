from bs4 import BeautifulSoup
import os
import re


data = os.listdir('data/')
file_names = ['book1', 'book2', 'book3', 'book4', 'book5', 'book6', 'book7']


def extract_text(file_path):
    with open(file_path,'r') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
        return soup.pre.string


def write_data(text, file_name):
    with open('final_data/'+file_name+'.txt', 'w') as file:
        file.write(text)


def clean_data(text):
    ctext = re.sub(r'/', ' ', text)
    dtext = re.sub(r'P( )?a( )?g( )?e( )?\|( )?[0-9a-zA-Z]+( )?(\n)*Harry Potter [a-zA-Z ]+( )?-( )?J.K. Rowling', ' ', ctext)
    return dtext


for d,file in zip(data, file_names):
    file_path = 'data/'+d
    text = extract_text(file_path)
    text = clean_data(text)
    write_data(text, file)
