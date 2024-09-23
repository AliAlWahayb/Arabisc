# -*- coding: utf-8 -*-
from nltk.stem.isri import ISRIStemmer
import nltk
import string
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import *
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
import arabic_reshaper
from bidi.algorithm import get_display


print(tensorflow.__version__)    # >=2.3.1

st = ISRIStemmer()

# Do this once
nltk.download('punkt')


print("Loading the spelling the checker model...")
model = load_model("model-dual-input/model.05-0.302-0.287.keras")


print("Building the tokenizer...")
data_file = "data/News-Multi.ar-en.ar.more.clean"
data = open(data_file, encoding="utf8").read()

corpus = data.lower().split("\n")
# print("First sentence in the corpus:", corpus[0])

vocab_size = 100000
max_sequence_len = 15
out_of_vocab = "<unk>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=out_of_vocab)
tokenizer.fit_on_texts(corpus)


# # RWE
# texts_to_correct = ["وتدعو الحريق إلى مواصلة التماس الآراء والتعليقات من جميع الجهات المعنية",
#                     "الفعالية المثلي في هذا الصدد تتعزز بالتعاون الدولي الواسع النطاق",
#                     "لم يبدأ تنفيذ خطط العمل الدولية إلا عدد صفير من البلدان",
#                     "توجه السكر إلى الدول الأعضاء",
#                     "التمييز ضد المرأة فيما يتعلق بحصولها على القرود المصرفية",
#                     "تدعيم تلك المناطق في النصف الجنوبي للكرة العرضية والمناطق المجاورة",
#                     "العمارات العربية المتحدة",
#                     "تؤيد فكره عقد مؤتمر للدول الأطراف",
#                     "قدمت مضر مشروع القرار الموصى به في تقرير اللجنة",
#                     "آثار التغييرات في أشعار الصرف ومعدلات التضخم",
#                     "مرسوم مجلس قيادة الثروة",
#                     "إسرار المجتمع الدولي على وضع حد لإفلات مرتكبي هذه الجرائم من العقاب",
#                     "التدخن في الشؤون التي تكون من صميم سيادة الدول",
#                     "زهور حركات عنصرية تدعو إلى العنف",
#                     "تعيد تأكيد حق الشغب الفلسطيني في تقرير المصير",
#                     "تؤكد أن التثقيف في مجال حقوق الإنسان أمر أثاثي في تغيير الاتجاهات",
#                     "المزيد من الجهود لمعالجة الغرق بين أجور الجنسين",
#                     "يقضي بتخفيض شعر الفائدة",
#                     "منظمة الأمم المتحدة للتربية والألم والثقافة",
#                     "حق الشعوب في تقرير المسير وغيره من حقوق الإنسان"
#                     ]

# # NWE
# texts_to_correct = ["وتدعو الفريق إلى مواصله التماس الآراء والتعليقات من جميع الجهات المعنية",
#                     "الفعالية المثلى في هذا الصدد تتعززز بالتعاون الدولي الواسع النطاق",
#                     "لم يبدأ تفنيذ خطط العمل الدولية إلا عدد صغير من البلدان",
#                     "توجة الشكر إلى الدول الأعضاء",
#                     "التمييز ضد المرأة فيما يتعلق بحصولها على القروض المرصفية",
#                     "تدعيم تلك المناطق في النصف الجنوبي للكرة الأرظية والمناطق المجاورة",
#                     "اشتركت في تقديمه الأردن والإمارات العربيع المتحدة",
#                     "تؤيد فكرة عقد مؤتمر للدول الأتراف",
#                     "قدمت مصر مشروع القرار المصى به في تقرير اللجنة",
#                     "آثار التغييرات في أسعار الصرف ومعدلت التضخم",
#                     "مرسوم مجلس غيادة الثورة",
#                     "إصرار المجتمع الدولي على وضع حد لإفلات مرتكبي هذه الجرايم من العقاب",
#                     "التدخل في الشؤون التي تكون من صميم يادة الدول",
#                     "ضهور حركات عنصرية تدعو إلى العنف",
#                     "تعيد تأكيد حق الشاعب الفلسطيني في تقرير المصير",
#                     "تؤكد أن التثقيف في مجال حقوق الإنسان أمر أساسي في تايير الاتجاهات",
#                     "المذيد من الجهود لمعالجة الفرق بين أجور الجنسين",
#                     "يقضي بتخفيد سعر الفائدة",
#                     "منظمة الأمم المتحده للتربية والعلم والثقافة",
#                     "حق الشعوب في تقرير المصير وغيره من حقوق الإنثان"
#                     ]

# AliAlwahayb code
with open("output.txt", "r") as f:
    texts_to_correct = f.read().splitlines()

if len(texts_to_correct) == 0:
    print("No text to correct")
    exit(0)

output = []
count_NotSure = 0
count_Correct = 0
count_Wrong = 0
count_IDoNotKnow = 0

def rtl_text(text):
    return get_display(arabic_reshaper.reshape(text))

# END AliAlwahayb code


def generate_ngrams(text):
    text = "<s>" + text

    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]

    all_grams = []

    for n in range(2, len(tokens)+1):
        grams = [w for w in nltk.ngrams(tokens, n=n)][0]

        grams_rev = tokens[n:]
        grams_rev.reverse()
        all_grams.append((list(grams), list(grams_rev)))

    return all_grams


for text_to_correct in texts_to_correct:

    print("Currently correcting:", rtl_text(text_to_correct))

    ngrams = generate_ngrams(text_to_correct)
    correct = None
    suggestions = []

    for ngram in ngrams:

        if len(ngram[0]) > 2 and correct != 1 and len(suggestions) != 0:
            seed_text_ltr = " ".join(
                word for word in ngram[0][:-2]) + " " + suggestions[0][2]
        else:
            seed_text_ltr = " ".join(word for word in ngram[0][:-1])

        current_word = ngram[0][-1]
        seed_text_rtl = " ".join(word for word in ngram[1])
        print(rtl_text(seed_text_ltr), "->",
              rtl_text(current_word), "->", rtl_text(seed_text_rtl))

        token_list = tokenizer.texts_to_sequences([seed_text_ltr])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')

        token_list_rev = tokenizer.texts_to_sequences([seed_text_rtl])[0]
        token_list_rev = pad_sequences(
            [token_list_rev], maxlen=max_sequence_len-1, padding='pre')

        predicted_id = np.argmax(model.predict(
            [token_list, token_list_rev]), axis=-1)
        predicted_word = tokenizer.sequences_to_texts([predicted_id])[0]
        print(rtl_text(predicted_word))

        predicted_probs = model.predict([token_list, token_list_rev])
        predicted_best = np.argsort(-predicted_probs, axis=-1)[0][:4500]

        suggestions = []
        correct = None

        for prob in predicted_best:
            output_word = tokenizer.sequences_to_texts([[prob]])[0]
            ed = nltk.edit_distance(current_word, output_word)

            if ed == 0:
                print("I got this one; it seems correct -->",
                      rtl_text(current_word), "=", rtl_text(output_word))
                output.append(str(rtl_text(current_word) + " = " + rtl_text(output_word)))
                correct = 1
                break
            elif len(current_word) <= 3 and ed == 1:
                suggestions.append((ed, current_word, output_word))
            elif len(current_word) > 3 and ed <= 2:
                suggestions.append((ed, current_word, output_word))
            else:
                continue

        if len(suggestions) > 0:
            for suggest in suggestions:
                lemmas_cw = []
                lemmas_cw.append(suggest[1])
                lemmas_cw.append(st.suf1(suggest[1]))
                lemmas_cw.append(st.suf32(suggest[1]))
                lemmas_cw.append(st.pre1(suggest[1]))
                lemmas_cw.append(st.pre32(suggest[1]))

                lemmas_ow = []
                lemmas_ow.append(suggest[2])
                lemmas_ow.append(st.suf1(suggest[2]))
                lemmas_ow.append(st.suf32(suggest[2]))
                lemmas_ow.append(st.pre1(suggest[2]))
                lemmas_ow.append(st.pre32(suggest[2]))

                if correct != 1 and len(suggest[1]) > 7:
                    for l in lemmas_cw:
                        if l in lemmas_ow:
                            correct = 2
                            print("I got the lemma; it seems correct -->",
                                  rtl_text(current_word), "~", rtl_text(suggest[2]))
                            output.append(str(rtl_text(current_word) + " ~ " + rtl_text(suggest[2])))

        print("Suggestions:", " - ".join(
            [rtl_text(suggest[2]) for suggest in suggestions]))

        if correct == 2:
            print("Not sure")
            count_NotSure += 1
        elif correct == 1:
            print("CORRECT")
            count_Correct += 1
        elif correct != 1 and len(suggestions) > 0:
            correct = 0
            print("WRONG")
            count_Wrong += 1
        elif correct != 1 and len(suggestions) == 0:
            print("I do not know!")
            count_IDoNotKnow += 1

        print("-------")

total = count_Correct + count_Wrong + count_NotSure + count_IDoNotKnow
print("Correct:", count_Correct, "-->", (count_Correct/total)*100, "%")
print("Wrong:", count_Wrong, "-->", (count_Wrong/total)*100, "%")
print("Not Sure:", count_NotSure, "-->", (count_NotSure/total)*100, "%")
print("I do not know:", count_IDoNotKnow, "-->", (count_IDoNotKnow/total)*100, "%")
print("\n".join(output))
