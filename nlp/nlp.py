# add proximity
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# stemmed = [stemmer.stem(token) for token in tokenized]

# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# lemmatized = [lemmatizer.lemmatize(token) for token in tokenized]

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation


# import spacy
# nlp = spacy.blank("en")


# from scipy.spatial.distance import cosine
# squid_vec=nlp("desaster").vector
# dist_sponge_star=cosine(sponge_vec,starfish_vec)


### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py


### 2. source data
full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\nlp-getting-started"
import pandas as pd
df_train=pd.read_csv(full_path+"\\train.csv", index_col='id')
df_test=pd.read_csv(full_path+"\\test.csv", index_col='id')

df_stacked_train=df_train.copy()
df_stacked_test=df_test.copy()

# For identification purposes
df_stacked_train["f_train"] = 1
df_stacked_test["f_train"] = 0
df_stacked_test["target"] = 0
df_stacked = pd.concat([df_stacked_train, df_stacked_test])


### 3. exploratory data analysis
# 3.1 create overview of data set:
#getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df_stacked)

num_var_nonan.remove('target')
num_var_nonan.remove('f_train')
all_var.remove('f_train')


### 4 pred data
# 4.1 split dependent and independent
dependent=df_train['target']
independent=df_train.copy(deep=True)
independent.drop(columns='target',axis='columns', inplace=True)

# 4.2 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.2, random_state=41, shuffle=True)



################################
# 4.3 Build a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn_pandas import DataFrameMapper

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import TransformerMixin


# print('airplane%20accident'.replace('%20',' '))


class AddVariablesNotImputed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from stop_words import get_stop_words
        import string 
        import re
        from country_list import countries_for_language
        
       
        #pre word_tokenize
        def clean_numbers(X):
            return re.sub('\d+', ' ', X)
              
        #Hashtags
        def find_hashtags(X):
            return re.findall('#[a-zA-Z]*', X, flags=0)
        
        def replace_typical_misspell(text):
            mispell_dict = {"aren't" : "are not",
                            "can't" : "cannot",
                            "couldn't" : "could not",
                            "couldnt" : "could not",
                            "didn't" : "did not",
                            "doesn't" : "does not",
                            "doesnt" : "does not",
                            "don't" : "do not",
                            "hadn't" : "had not",
                            "hasn't" : "has not",
                            "haven't" : "have not",
                            "havent" : "have not",
                            "he'd" : "he would",
                            "he'll" : "he will",
                            "he's" : "he is",
                            "i'd" : "I would",
                            "i'd" : "I had",
                            "i'll" : "I will",
                            "i'm" : "I am",
                            "isn't" : "is not",
                            "it's" : "it is",
                            "it'll":"it will",
                            "i've" : "I have",
                            "let's" : "let us",
                            "mightn't" : "might not",
                            "mustn't" : "must not",
                            "shan't" : "shall not",
                            "she'd" : "she would",
                            "she'll" : "she will",
                            "she's" : "she is",
                            "shouldn't" : "should not",
                            "shouldnt" : "should not",
                            "that's" : "that is",
                            "thats" : "that is",
                            "there's" : "there is",
                            "theres" : "there is",
                            "they'd" : "they would",
                            "they'll" : "they will",
                            "they're" : "they are",
                            "theyre":  "they are",
                            "they've" : "they have",
                            "we'd" : "we would",
                            "we're" : "we are",
                            "weren't" : "were not",
                            "we've" : "we have",
                            "what'll" : "what will",
                            "what're" : "what are",
                            "what's" : "what is",
                            "what've" : "what have",
                            "where's" : "where is",
                            "who'd" : "who would",
                            "who'll" : "who will",
                            "who're" : "who are",
                            "who's" : "who is",
                            "who've" : "who have",
                            "won't" : "will not",
                            "wouldn't" : "would not",
                            "you'd" : "you would",
                            "you'll" : "you will",
                            "you're" : "you are",
                            "you've" : "you have",
                            "'re": " are",
                            "wasn't": "was not",
                            "we'll":" will",
                            "didn't": "did not"}
            mispellings_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
            def replace(match):
                return mispell_dict[match.group(0)]
            return mispellings_re.sub(replace, text)
        
        #post word_tokenize
        # Stopwords
        lst=['text','keyword','location']
        stop_words1=list(stopwords.words('english'))        
        stop_words2=get_stop_words('english')
        punct=list(string.punctuation)
        stopwords_web=["'s","'ll","'tis","'twas","'ve","10","39","a","a's","able","ableabout","about","above","abroad","abst","accordance","according","accordingly","across","act","actually","ad","added","adj","adopted","ae","af","affected","affecting","affects","after","afterwards","ag","again","against","ago","ah","ahead","ai","ain't","aint","al","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","amoungst","amount","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","ao","apart","apparently","appear","appreciate","appropriate","approximately","aq","ar","are","area","areas","aren","aren't","arent","arise","around","arpa","as","aside","ask","asked","asking","asks","associated","at","au","auth","available","aw","away","awfully","az","b","ba","back","backed","backing","backs","backward","backwards","bb","bd","be","became","because","become","becomes","becoming","been","before","beforehand","began","begin","beginning","beginnings","begins","behind","being","beings","believe","below","beside","besides","best","better","between","beyond","bf","bg","bh","bi","big","bill","billion","biol","bj","bm","bn","bo","both","bottom","br","brief","briefly","bs","bt","but","buy","bv","bw","by","bz","c","c'mon","c's","ca","call","came","can","can't","cannot","cant","caption","case","cases","cause","causes","cc","cd","certain","certainly","cf","cg","ch","changes","ci","ck","cl","clear","clearly","click","cm","cmon","cn","co","co.","com","come","comes","computer","con","concerning","consequently","consider","considering","contain","containing","contains","copy","corresponding","could","could've","couldn","couldn't","couldnt","course","cr","cry","cs","cu","currently","cv","cx","cy","cz","d","dare","daren't","darent","date","de","dear","definitely","describe","described","despite","detail","did","didn","didn't","didnt","differ","different","differently","directly","dj","dk","dm","do","does","doesn","doesn't","doesnt","doing","don","don't","done","dont","doubtful","down","downed","downing","downs","downwards","due","during","dz","e","each","early","ec","ed","edu","ee","effect","eg","eh","eight","eighty","either","eleven","else","elsewhere","empty","enough","entirely","er","es","especially","et","et-al","etc","even","evenly","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","face","faces","fact","facts","fairly","far","farther","felt","few","fewer","ff","fi","fifteen","fifth","fifty","fify","fill","find","finds","fire","first","five","fix","fj","fk","fm","fo","followed","following","follows","for","forever","former","formerly","forth","forty","forward","found","four","fr","free","from","front","full","fully","further","furthered","furthering","furthermore","furthers","fx","g","ga","gave","gb","gd","ge","general","generally","get","gets","getting","gf","gg","gh","gi","give","given","gives","giving","gl","gm","gmt","gn","go","goes","going","gone","good","goods","got","gotten","gov","gp","gq","gr","great","greater","greatest","greetings","group","grouped","grouping","groups","gs","gt","gu","gw","gy","h","had","hadn't","hadnt","half","happens","hardly","has","hasn","hasn't","hasnt","have","haven","haven't","havent","having","he","he'd","he'll","he's","hed","hell","hello","help","hence","her","here","here's","hereafter","hereby","herein","heres","hereupon","hers","herself","herse”","hes","hi","hid","high","higher","highest","him","himself","himse”","his","hither","hk","hm","hn","home","homepage","hopefully","how","how'd","how'll","how's","howbeit","however","hr","ht","htm","html","http","hu","hundred","i","i'd","i'll","i'm","i've","i.e.","id","ie","if","ignored","ii","il","ill","im","immediate","immediately","importance","important","in","inasmuch","inc","inc.","indeed","index","indicate","indicated","indicates","information","inner","inside","insofar","instead","int","interest","interested","interesting","interests","into","invention","inward","io","iq","ir","is","isn","isn't","isnt","it","it'd","it'll","it's","itd","itll","its","itself","itse”","ive","j","je","jm","jo","join","jp","just","k","ke","keep","keeps","kept","keys","kg","kh","ki","kind","km","kn","knew","know","known","knows","kp","kr","kw","ky","kz","l","la","large","largely","last","lately","later","latest","latter","latterly","lb","lc","least","length","less","lest","let","let's","lets","li","like","liked","likely","likewise","line","little","lk","ll","long","longer","longest","look","looking","looks","low","lower","lr","ls","lt","ltd","lu","lv","ly","m","ma","made","mainly","make","makes","making","man","many","may","maybe","mayn't","maynt","mc","md","me","mean","means","meantime","meanwhile","member","members","men","merely","mg","mh","microsoft","might","might've","mightn't","mightnt","mil","mill","million","mine","minus","miss","mk","ml","mm","mn","mo","more","moreover","most","mostly","move","mp","mq","mr","mrs","ms","msie","mt","mu","much","mug","must","must've","mustn't","mustnt","mv","mw","mx","my","myself","myse”","mz","n","na","name","namely","nay","nc","nd","ne","near","nearly","necessarily","necessary","need","needed","needing","needn't","neednt","needs","neither","net","netscape","never","neverf","neverless","nevertheless","new","newer","newest","next","nf","ng","ni","nine","ninety","nl","no","no-one","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","notwithstanding","novel","now","nowhere","np","nr","nu","null","number","numbers","nz","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","older","oldest","om","omitted","on","once","one","one's","ones","only","onto","open","opened","opening","opens","opposite","or","ord","order","ordered","ordering","orders","org","other","others","otherwise","ought","oughtn't","oughtnt","our","ours","ourselves","out","outside","over","overall","owing","own","p","pa","page","pages","part","parted","particular","particularly","parting","parts","past","pe","per","perhaps","pf","pg","ph","pk","pl","place","placed","places","please","plus","pm","pmid","pn","point","pointed","pointing","points","poorly","possible","possibly","potentially","pp","pr","predominantly","present","presented","presenting","presents","presumably","previously","primarily","probably","problem","problems","promptly","proud","provided","provides","pt","put","puts","pw","py","q","qa","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","reasonably","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","reserved","respectively","resulted","resulting","results","right","ring","ro","room","rooms","round","ru","run","rw","s","sa","said","same","saw","say","saying","says","sb","sc","sd","se","sec","second","secondly","seconds","section","see","seeing","seem","seemed","seeming","seems","seen","sees","self","selves","sensible","sent","serious","seriously","seven","seventy","several","sg","sh","shall","shan't","shant","she","she'd","she'll","she's","shed","shell","shes","should","should've","shouldn","shouldn't","shouldnt","show","showed","showing","shown","showns","shows","si","side","sides","significant","significantly","similar","similarly","since","sincere","site","six","sixty","sj","sk","sl","slightly","sm","small","smaller","smallest","sn","so","some","somebody","someday","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","sr","st","state","states","still","stop","strongly","su","sub","substantially","successfully","such","sufficiently","suggest","sup","sure","sv","sy","system","sz","t","t's","take","taken","taking","tc","td","tell","ten","tends","test","text","tf","tg","th","than","thank","thanks","thanx","that","that'll","that's","that've","thatll","thats","thatve","the","their","theirs","them","themselves","then","thence","there","there'd","there'll","there're","there's","there've","thereafter","thereby","thered","therefore","therein","therell","thereof","therere","theres","thereto","thereupon","thereve","these","they","they'd","they'll","they're","they've","theyd","theyll","theyre","theyve","thick","thin","thing","things","think","thinks","third","thirty","this","thorough","thoroughly","those","thou","though","thoughh","thought","thoughts","thousand","three","throug","through","throughout","thru","thus","til","till","tip","tis","tj","tk","tm","tn","to","today","together","too","took","top","toward","towards","tp","tr","tried","tries","trillion","truly","try","trying","ts","tt","turn","turned","turning","turns","tv","tw","twas","twelve","twenty","twice","two","tz","u","ua","ug","uk","um","un","under","underneath","undoing","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","upwards","us","use","used","useful","usefully","usefulness","uses","using","usually","uucp","uy","uz","v","va","value","various","vc","ve","versus","very","vg","vi","via","viz","vn","vol","vols","vs","vu","w","want","wanted","wanting","wants","was","wasn","wasn't","wasnt","way","ways","we","we'd","we'll","we're","we've","web","webpage","website","wed","welcome","well","wells","went","were","weren","weren't","werent","weve","wf","what","what'd","what'll","what's","what've","whatever","whatll","whats","whatve","when","when'd","when'll","when's","whence","whenever","where","where'd","where'll","where's","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","whichever","while","whilst","whim","whither","who","who'd","who'll","who's","whod","whoever","whole","wholl","whom","whomever","whos","whose","why","why'd","why'll","why's","widely","width","will","willing","wish","with","within","without","won","won't","wonder","wont","words","work","worked","working","works","world","would","would've","wouldn","wouldn't","wouldnt","ws","www","x","y","ye","year","years","yes","yet","you","you'd","you'll","you're","you've","youd","youll","young","younger","youngest","your","youre","yours","yourself","yourselves","youve","yt","yu","z","za","zero","zm","zr"]
        full_list=stop_words1+stop_words2+punct+['...','..','....']+stopwords_web
        
        def exclude_stop(X):
            lst=list()
            for i in X:
                if i not in full_list:
                    lst.append(i)
            return lst
        
        #countries
        countries = list(countries_for_language('en'))
             
        us_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
                     "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                     "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                     "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                     "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
         
        # def match_country(X):
         
        for i in lst:
            X[i].fillna('None', inplace=True)
            X[str(i)]=X[i].str.replace('%20',' ') # replace %20 by space
            X[str(i)]=X[i].str.lower()
            X[str(i)+'_raw_tokenized'] = X[i].apply(lambda row: word_tokenize(row))
            X[str(i)]=X[i].apply(lambda row: clean_numbers(row))
            X[str(i)]=X[i].apply(lambda row: replace_typical_misspell(row))
            X[str(i)+'_tokenized'] = X[i].apply(lambda row: word_tokenize(row))
            X[str(i)+'_len'] =  X[i].apply(lambda row: len(row))
            X[str(i)+'_cnt_elem'] =  X[str(i)+'_tokenized'].apply(lambda row: len(row))
            #X[str(i)+'_tokenized_woStop']=X[str(i)+'_tokenized'].apply(lambda row: exclude_stop(row))
        X['text_tokenized_woStop']=X['text_tokenized'].apply(lambda row: exclude_stop(row))
        X['hashtags']=X['text'].apply(lambda row: find_hashtags(row))
        
        return X


class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols

################################
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import make_union
#from sklearn.compose import make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


column_trans = make_column_transformer(
    (CountVectorizer(analyzer=lambda x: x), 'text_raw_tokenized'),
    (CountVectorizer(analyzer=lambda x: x), 'keyword_tokenized'),
    (CountVectorizer(analyzer=lambda x: x), 'location_tokenized'),
    (CountVectorizer(analyzer=lambda x: x), 'text_tokenized_woStop'),
    (CountVectorizer(analyzer=lambda x: x), 'hashtags'),
    remainder='passthrough')


# transformation = DataFrameMapper(
#     [([item],[OneHotEncoder(handle_unknown='ignore')]) for item in nominal]
#     + [([col],OrdinalEncoder(categories = [cat])) for col, cat in column_to_cat.items()]
#     + [([item],PowerTransformer()) for item in skew_var]
#     # + [item for item in remainder]
#     # + [([item],ColumnExtractor(item)) for item in remainder]
#     ,default=None
#     ,df_out=True
# )

################################

# pipeline_pre = make_pipeline(
#     AddVariablesNotImputed()
#     # ,(impute)
#     # ,AddVariablesImputed()
#     # ,(transformation)
#     # ,AddVariablesTransformed()
#     # ,(VarianceThreshold(0.1))
#     # ,(StandardScaler())
# )

pipeline_pre = make_pipeline(
    (AddVariablesNotImputed())
    ,ColumnExtractor(['text_raw_tokenized',
    'text_len','text_cnt_elem', 'keyword_tokenized', 'keyword_len', 'keyword_cnt_elem',
       'location_tokenized', 'location_len', 'location_cnt_elem',
       'text_tokenized_woStop', 'hashtags'
       # ,'respell','text'
       ])
    ,(column_trans)
    )

# independent_trans.columns
independent_trans=pipeline_pre.fit_transform(independent.copy())
#independent_trans=pipeline_pre.fit_transform(independent.copy()).toarray()

best_model=pre_eval_models(
    type_model='classification',
    independent=independent_trans, 
    dependent=dependent,
    scoring='accuracy',
    cv=10)

#RandomForestClassifier()

################################

### 4 pred data
# 4.1 split dependent and independent
dependent=df_train['target']
independent=df_train.copy(deep=True)
independent.drop(columns='target',axis='columns', inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.2, random_state=41, shuffle=True)


from sklearn.ensemble import StackingClassifier as stack


estimators = [('LR',LogisticRegression())
              ,('GBC',GradientBoostingClassifier())
              ,('KC',KNeighborsClassifier())
              ]


pipeline = make_pipeline(
    (AddVariablesNotImputed())
    ,ColumnExtractor(['text_raw_tokenized',
    'text_len','text_cnt_elem', 'keyword_tokenized', 'keyword_len', 'keyword_cnt_elem',
       'location_tokenized', 'location_len', 'location_cnt_elem',
       'text_tokenized_woStop', 'hashtags'])
    ,(column_trans)
    #,RandomForestClassifier()
    #,LogisticRegression()
    # ,GradientBoostingClassifier()
    ,stack(estimators=estimators, final_estimator=RandomForestClassifier()
           ,cv=2, n_jobs=-1, passthrough=True, verbose=1)
)
  

# trans_model=pipeline.fit(X_train.copy(),Y_train.copy())

trans_model=pipeline.fit(X_train.copy(),Y_train.copy())

# 4.4 Check metrics

evaluate_model(model_type='classification'
                ,model=trans_model
                ,X=X_test.copy()
                ,y_true=Y_test.copy()
                )

# 5. Apply model
# apply model to validation
df_test=pd.read_csv(full_path+"\\test.csv",index_col='id')

df_test['target']=trans_model.predict(df_test.copy())

#output
out=df_test['target']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)


