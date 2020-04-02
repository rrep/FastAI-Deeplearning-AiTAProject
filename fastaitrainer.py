from fastai.text import * 

#set path to local folder
path = Path('.')

#bring in data
df = pd.read_csv(path/'aita_clean.csv')
df.head()


#clean DF
df.dropna(subset = ["body"], inplace=True)
df.dropna(subset = ["verdict"], inplace=True)

#create databunch for classifier
data = (TextList.from_df(df,path, cols=3)
            .split_subsets(train_size=0.8, valid_size=0.2)  
            #refer to label 'verdict'
            .label_from_df(cols=5) 
            #turn into databunch
            .databunch())

#create data bunch for LM
bs=48
data_lm = (TextList.from_df(df,path, cols=3)
            #split 10% out for validation data
            .split_by_rand_pct(0.1)
           #language model not classifier
            .label_for_lm()           
            .databunch(bs=bs))


#optionally save the model data
#data_lm.save('data_lm_export.pkl')
#data.save('data_clas_export.pkl')

#CREATE BASE LANGUAGE MODEL
#create a learner - uses transfer learning from pre-trained AWD_LSTM
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

#train one cycle to on all the data
learn.fit_one_cycle(1, 1e-2)

#fine tune
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)

#optionally test where we're at with the language model through a predictive exercise
# learn.predict("I had a big fight with my ", n_words=20)

#save the trained LM 
learn.save_encoder('ft_enc')

#CREATE ACTUAL CLASSIFIER
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')

#Train and Tune Dawg
learn.fit_one_cycle(1, 1e-2)

#freeze all but the last 2 weight matrices
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

#do em all for one last pass
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

# optionally see if the computer thinks you're an a-hole
#learn.predict("I accidently stepped on a child when I was walking to the store.")

#release the trained model to judge the masses
learn.export()