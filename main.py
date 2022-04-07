import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def preprocess_data(inp_data):
    # Remove package name as it's not relevant
    return_data = inp_data.drop('package_name', axis=1)

    # Convert text to lowercase
    return_data['review'] = return_data['review'].str.strip().str.lower()
    return return_data


def transform_predict(predicted):
    predicted_data = predicted[0]
    if predicted_data == 1:
        return "Positive"
    else:
        return "Negative"


# load data set
data = pd.read_csv('google_play_store_apps_reviews_training.csv')

# preproses data digunakan untuk menghapus data package_name, sehingga hanya meninggalkan review dan polarity
# kemudia ubah semua review menjadi huruf kecil
data = preprocess_data(data)
data.head()

# pisahkan antara data review dan polarity
x = data['review']
y = data['polarity']
# bagi dataset menjadi dua, untuk test dan train
x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

# Vectorize text reviews to numbers
# buang kata-kata yang tidak berpengaruh pada proses training dengan stop_words
vec = CountVectorizer(stop_words='english')
# ubah data text menjadi number kemudian lakukan train dataset
x = vec.fit_transform(x).toarray()
# ubah data text menjadi number, tidak perlu dilakukan train karena x_test adalah data untuk melakukan testing
x_test = vec.transform(x_test).toarray()

# load algoritma multinomial naive bayes
model = MultinomialNB()
# lakukan train dataset dengan menggunakan algoritma  multinomial naive bayes
model.fit(x, y)

# check score dari proses train
model.score(x_test, y_test)

positive_review = 'Love this app simply awesome!'
negative_review = 'the apps is suck'

# uji coba positive review
print(positive_review + ",. merupakan review bernilai :")
print(transform_predict(model.predict(vec.transform([positive_review]))))

# uji coba negative review
print(negative_review + ",. merupakan review bernilai :")
print(transform_predict(model.predict(vec.transform([negative_review]))))
