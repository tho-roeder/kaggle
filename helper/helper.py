# -*- coding: utf-8 -*-

# import os
# path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
# os.chdir(path)
# %run helper.py


### nlp
def get_nlp_helper():
    #stopwords
    stopwords_git=["'ll","'tis","'twas","'ve","10","39","a","a's","able","ableabout","about","above","abroad","abst","accordance","according","accordingly","across","act","actually","ad","added","adj","adopted","ae","af","affected","affecting","affects","after","afterwards","ag","again","against","ago","ah","ahead","ai","ain't","aint","al","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","amoungst","amount","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","ao","apart","apparently","appear","appreciate","appropriate","approximately","aq","ar","are","area","areas","aren","aren't","arent","arise","around","arpa","as","aside","ask","asked","asking","asks","associated","at","au","auth","available","aw","away","awfully","az","b","ba","back","backed","backing","backs","backward","backwards","bb","bd","be","became","because","become","becomes","becoming","been","before","beforehand","began","begin","beginning","beginnings","begins","behind","being","beings","believe","below","beside","besides","best","better","between","beyond","bf","bg","bh","bi","big","bill","billion","biol","bj","bm","bn","bo","both","bottom","br","brief","briefly","bs","bt","but","buy","bv","bw","by","bz","c","c'mon","c's","ca","call","came","can","can't","cannot","cant","caption","case","cases","cause","causes","cc","cd","certain","certainly","cf","cg","ch","changes","ci","ck","cl","clear","clearly","click","cm","cmon","cn","co","co.","com","come","comes","computer","con","concerning","consequently","consider","considering","contain","containing","contains","copy","corresponding","could","could've","couldn","couldn't","couldnt","course","cr","cry","cs","cu","currently","cv","cx","cy","cz","d","dare","daren't","darent","date","de","dear","definitely","describe","described","despite","detail","did","didn","didn't","didnt","differ","different","differently","directly","dj","dk","dm","do","does","doesn","doesn't","doesnt","doing","don","don't","done","dont","doubtful","down","downed","downing","downs","downwards","due","during","dz","e","each","early","ec","ed","edu","ee","effect","eg","eh","eight","eighty","either","eleven","else","elsewhere","empty","end","ended","ending","ends","enough","entirely","er","es","especially","et","et-al","etc","even","evenly","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","face","faces","fact","facts","fairly","far","farther","felt","few","fewer","ff","fi","fifteen","fifth","fifty","fify","fill","find","finds","fire","first","five","fix","fj","fk","fm","fo","followed","following","follows","for","forever","former","formerly","forth","forty","forward","found","four","fr","free","from","front","full","fully","further","furthered","furthering","furthermore","furthers","fx","g","ga","gave","gb","gd","ge","general","generally","get","gets","getting","gf","gg","gh","gi","give","given","gives","giving","gl","gm","gmt","gn","go","goes","going","gone","good","goods","got","gotten","gov","gp","gq","gr","great","greater","greatest","greetings","group","grouped","grouping","groups","gs","gt","gu","gw","gy","h","had","hadn't","hadnt","half","happens","hardly","has","hasn","hasn't","hasnt","have","haven","haven't","havent","having","he","he'd","he'll","he's","hed","hell","hello","help","hence","her","here","here's","hereafter","hereby","herein","heres","hereupon","hers","herself","herse”","hes","hi","hid","high","higher","highest","him","himself","himse”","his","hither","hk","hm","hn","home","homepage","hopefully","how","how'd","how'll","how's","howbeit","however","hr","ht","htm","html","http","hu","hundred","i","i'd","i'll","i'm","i've","i.e.","id","ie","if","ignored","ii","il","ill","im","immediate","immediately","importance","important","in","inasmuch","inc","inc.","indeed","index","indicate","indicated","indicates","information","inner","inside","insofar","instead","int","interest","interested","interesting","interests","into","invention","inward","io","iq","ir","is","isn","isn't","isnt","it","it'd","it'll","it's","itd","itll","its","itself","itse”","ive","j","je","jm","jo","join","jp","just","k","ke","keep","keeps","kept","keys","kg","kh","ki","kind","km","kn","knew","know","known","knows","kp","kr","kw","ky","kz","l","la","large","largely","last","lately","later","latest","latter","latterly","lb","lc","least","length","less","lest","let","let's","lets","li","like","liked","likely","likewise","line","little","lk","ll","long","longer","longest","look","looking","looks","low","lower","lr","ls","lt","ltd","lu","lv","ly","m","ma","made","mainly","make","makes","making","man","many","may","maybe","mayn't","maynt","mc","md","me","mean","means","meantime","meanwhile","member","members","men","merely","mg","mh","microsoft","might","might've","mightn't","mightnt","mil","mill","million","mine","minus","miss","mk","ml","mm","mn","mo","more","moreover","most","mostly","move","mp","mq","mr","mrs","ms","msie","mt","mu","much","mug","must","must've","mustn't","mustnt","mv","mw","mx","my","myself","myse”","mz","n","na","name","namely","nay","nc","nd","ne","near","nearly","necessarily","necessary","need","needed","needing","needn't","neednt","needs","neither","net","netscape","never","neverf","neverless","nevertheless","new","newer","newest","next","nf","ng","ni","nine","ninety","nl","no","no-one","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","notwithstanding","novel","now","nowhere","np","nr","nu","null","number","numbers","nz","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","older","oldest","om","omitted","on","once","one","one's","ones","only","onto","open","opened","opening","opens","opposite","or","ord","order","ordered","ordering","orders","org","other","others","otherwise","ought","oughtn't","oughtnt","our","ours","ourselves","out","outside","over","overall","owing","own","p","pa","page","pages","part","parted","particular","particularly","parting","parts","past","pe","per","perhaps","pf","pg","ph","pk","pl","place","placed","places","please","plus","pm","pmid","pn","point","pointed","pointing","points","poorly","possible","possibly","potentially","pp","pr","predominantly","present","presented","presenting","presents","presumably","previously","primarily","probably","problem","problems","promptly","proud","provided","provides","pt","put","puts","pw","py","q","qa","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","reasonably","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","reserved","respectively","resulted","resulting","results","right","ring","ro","room","rooms","round","ru","run","rw","s","sa","said","same","saw","say","saying","says","sb","sc","sd","se","sec","second","secondly","seconds","section","see","seeing","seem","seemed","seeming","seems","seen","sees","self","selves","sensible","sent","serious","seriously","seven","seventy","several","sg","sh","shall","shan't","shant","she","she'd","she'll","she's","shed","shell","shes","should","should've","shouldn","shouldn't","shouldnt","show","showed","showing","shown","showns","shows","si","side","sides","significant","significantly","similar","similarly","since","sincere","site","six","sixty","sj","sk","sl","slightly","sm","small","smaller","smallest","sn","so","some","somebody","someday","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","sr","st","state","states","still","stop","strongly","su","sub","substantially","successfully","such","sufficiently","suggest","sup","sure","sv","sy","system","sz","t","t's","take","taken","taking","tc","td","tell","ten","tends","test","text","tf","tg","th","than","thank","thanks","thanx","that","that'll","that's","that've","thatll","thats","thatve","the","their","theirs","them","themselves","then","thence","there","there'd","there'll","there're","there's","there've","thereafter","thereby","thered","therefore","therein","therell","thereof","therere","theres","thereto","thereupon","thereve","these","they","they'd","they'll","they're","they've","theyd","theyll","theyre","theyve","thick","thin","thing","things","think","thinks","third","thirty","this","thorough","thoroughly","those","thou","though","thoughh","thought","thoughts","thousand","three","throug","through","throughout","thru","thus","til","till","tip","tis","tj","tk","tm","tn","to","today","together","too","took","top","toward","towards","tp","tr","tried","tries","trillion","truly","try","trying","ts","tt","turn","turned","turning","turns","tv","tw","twas","twelve","twenty","twice","two","tz","u","ua","ug","uk","um","un","under","underneath","undoing","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","upwards","us","use","used","useful","usefully","usefulness","uses","using","usually","uucp","uy","uz","v","va","value","various","vc","ve","versus","very","vg","vi","via","viz","vn","vol","vols","vs","vu","w","want","wanted","wanting","wants","was","wasn","wasn't","wasnt","way","ways","we","we'd","we'll","we're","we've","web","webpage","website","wed","welcome","well","wells","went","were","weren","weren't","werent","weve","wf","what","what'd","what'll","what's","what've","whatever","whatll","whats","whatve","when","when'd","when'll","when's","whence","whenever","where","where'd","where'll","where's","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","whichever","while","whilst","whim","whither","who","who'd","who'll","who's","whod","whoever","whole","wholl","whom","whomever","whos","whose","why","why'd","why'll","why's","widely","width","will","willing","wish","with","within","without","won","won't","wonder","wont","words","work","worked","working","works","world","would","would've","wouldn","wouldn't","wouldnt","ws","www","x","y","ye","year","years","yes","yet","you","you'd","you'll","you're","you've","youd","youll","young","younger","youngest","your","youre","yours","yourself","yourselves","youve","yt","yu","z","za","zero","zm","zr"]
    
    #punctuation
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
     '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
     '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
     '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
     '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    import string
    puncts = puncts + list(string.punctuation)
    
    #abbreviations
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
    return stopwords_git, puncts, mispell_dict


def getGeneralInformation(df):
    print(df.info)
    print()
    print(type(df))
    print()
    print(df.describe)
    print()
    print(df.dtypes)
    print('Rows and Columns in dataset:', df.shape)
    for col in df.columns:
        temp_col = df[col].isnull().sum()
        print(f'{col}: {temp_col}')


### general
def find_var(df,pattern):
    import re
    sel=[]
    for i in df.columns:
        if re.findall(pattern,i)!=[]:
            #print(i)
            sel.append(i)
    return sel


def getMissingValues(df):
    print("missing values:")
    for col in df.columns:
        if df[col].isnull().sum()>0:
            temp_col = df[col].isnull().sum()
            print(f'{col}: {temp_col}')


def getNegativeValues(df):
    print("negative values:")
    for col in df.columns:
        if df[col].dtypes != 'object':
            if df[df[col]<0][col].count().sum() >0:
                temp_col = df[df[col]<0][col].count().sum()
                print(f'{col}: {temp_col}')


def pre_work(df):
    num_var_nan=list()
    num_var_nonan=list()
    str_var_nan=list()
    str_var_nonan=list()
    for i in df.columns:
        if df[i].dtypes != 'object':
            if df[i].isna().sum()>0:
                num_var_nan.append(i)
            else:
                num_var_nonan.append(i)
        else:
            if df[i].isna().sum()>0:
                str_var_nan.append(i)
            else:
                str_var_nonan.append(i)
    return list(num_var_nan+ num_var_nonan+ str_var_nan+ str_var_nonan), num_var_nan, num_var_nonan, str_var_nan, str_var_nonan

### plotting
def plot_num_var(df,var):
    import matplotlib.pyplot as plt
    for i in var:
        # print(df[i].value_counts())
        if df[i].dtypes != 'object':
            nbr_nan=df[i].isna().sum()
            perc_nan=nbr_nan/len(df[i])
            plt.hist(df[i])
            plt.title("Numeric var: {}, cnt: {}, perc: {}".format(i,nbr_nan,round(perc_nan,2)))
            plt.show()
            plt.clf()


def plot_str_var(df,var):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for i in var:
        if df[i].dtypes == 'object':
            nbr_nan=df[i].isna().sum()
            perc_nan=nbr_nan/len(df[i])
            sns.countplot(x=df[i].fillna('Missing'),order = df[i].fillna('Missing').value_counts().index)
            plt.title("String var: {}, cnt: {}, perc: {}".format(i,nbr_nan,round(perc_nan,2)))
            plt.show()
            plt.clf()


def get_violinplot_for_target(df,var,target):
    import seaborn as sns
    import matplotlib.pyplot as plt
    for i in var:
        sns.violinplot(x=df[target], y=df[i])
        plt.show()
        plt.clf()


def get_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")
    plt.show()


def get_scatter_for_target(df,var,target):
    import matplotlib.pyplot as plt
    for i in var:
        plt.scatter(x=df[target], y=df[i])
        plt.title("{} vs {}".format(i,target))
        plt.show()
        plt.clf()


def plot_Outlier(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(18,25))
    sns.boxplot(data=df, orient="h")
    plt.show()


def impute_var(df,var,perc_drop,style):
    import numpy as np
    var_drop=[]
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    df[i].fillna(value=df[i].mean(),inplace=True)
                if style == 'median':
                    df[i].fillna(value=df[i].median(),inplace=True)
                if style == 'nan':
                    df[i].fillna(value=np.nan,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    df[i].fillna(value='missing',inplace=True)
            if style == 'mode':
                df[i].fillna(value=df[i].mode(dropna=True).values[0],inplace=True)
    return var_drop


def impute_var_v2(df,var,perc_drop,style):
    import numpy as np
    lst_var_drop=[]
    lst_impute=[]
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            lst_var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    impute_value=df[i].mean()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'median':
                    impute_value=df[i].median()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'nan':
                    impute_value=np.nan
                    df[i].fillna(value=impute_value,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    impute_value='missing'
                    df[i].fillna(value=impute_value,inplace=True)
            if style == 'mode':
                impute_value=df[i].mode(dropna=True).values[0]
                df[i].fillna(value=impute_value,inplace=True)
            lst_impute.append([i,impute_value])
    return lst_var_drop,lst_impute


def impute_var_v3(df,var,perc_drop,style):
    import numpy as np
    lst_var_drop=[]
    lst_impute=dict()
    #add drop na for full df
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            lst_var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    impute_value=df[i].mean()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'median':
                    impute_value=df[i].median()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'nan':
                    impute_value=np.nan
                    df[i].fillna(value=impute_value,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    impute_value='missing'
                    df[i].fillna(value=impute_value,inplace=True)
            if style == 'mode':
                impute_value=df[i].mode(dropna=True).values[0]
                df[i].fillna(value=impute_value,inplace=True)
            lst_impute[i]=impute_value
    return lst_var_drop,lst_impute


def impute_var_v4(df,var,perc_drop,style,value=None):
    import numpy as np
    lst_var_drop=[]
    lst_impute=dict()
    #add drop na for full df
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            lst_var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    impute_value=df[i].mean()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'median':
                    impute_value=df[i].median()
                    df[i].fillna(value=impute_value,inplace=True)
                if style == 'nan':
                    impute_value=np.nan
                    df[i].fillna(value=impute_value,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    impute_value='missing'
                    df[i].fillna(value=impute_value,inplace=True)
            if style == 'mode':
                impute_value=df[i].mode(dropna=True).values[0]
                df[i].fillna(value=impute_value,inplace=True)
            elif style == 'value':
                impute_value=value
                df[i].fillna(value=impute_value,inplace=True)
            lst_impute[i]=impute_value
    return lst_var_drop,lst_impute


def low_corr(df,target,min_cor):
    cor=df.corr()
    drop_list_lowCor=cor[abs(cor[target])<=min_cor]
    return list(drop_list_lowCor.index)


def get_df_high_corr_target(df,target,min_cor):
    cor=df.corr()
    df_highCor=cor[abs(cor[target])>=0.5]
    lst_highCor=list(df_highCor.index)
    if target in lst_highCor:
        lst_highCor.remove(target)
    return df_highCor, lst_highCor


def merge_low_corr(df_ind,df_dep,target,min_cor):
    import pandas as pd
    df=pd.merge(left=df_ind,right=df_dep,how="inner",left_index=True,
    right_index=True)
    cor=df.corr()
    drop_list_lowCor=cor[abs(cor[target])<=min_cor]
    return list(drop_list_lowCor.index)


def same_value(df,var,max_perc_rep):
    drop_list_max_perc_rep=[]
    for i in var:
        if (df[i].value_counts().max()/len(df[i]))>=max_perc_rep:
            drop_list_max_perc_rep.append(i)
    return drop_list_max_perc_rep


def transformer(df,var,transformer,prefix):
    df_copy=df.copy()
    for i in var:
        t=transformer
        df_copy[prefix+"_"+str(i)]=t.fit_transform(df_copy[i].to_numpy().reshape(-1,1))
    return df_copy


# def Standardize_values(df):
#     # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#     # transform to same mean and same standard deviation
#     from sklearn.preprocessing import StandardScaler
#     SC = StandardScaler()
#     SC.fit(df)
#     return SC


# def Normalize_values(df):
#     from sklearn.preprocessing import Normalizer
#     transformer = Normalizer().fit(df)
#     transformer.fit(df)
#     return transformer


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def no_intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value not in lst2] 
    return lst3 


# def treat_str_var(df,var):
#     from sklearn.preprocessing import LabelEncoder # Converts cat data to numeric
#     le=LabelEncoder()
#     for i in var:
#         df[i]=le.fit_transform(df[i])


def tree_to_code(tree, feature_names):
    from sklearn.tree import _tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    

def get_tree_pic(df_X, df_Y, var):
    from sklearn.tree import DecisionTreeClassifier    
    import numpy as np
    from sklearn.tree import export_graphviz
    from six import StringIO
    from IPython.display import Image  
    import pydotplus

    if len(var)==1:
        split=np.array(df_X[var]).reshape(-1, 1)
    else:
        split=df_X[var]
    classifier = DecisionTreeClassifier(random_state = 0, max_depth=2)
    
    classifier.fit(split, df_Y)
    
    dot_data = StringIO()
    export_graphviz(classifier
                    ,out_file=dot_data
                    ,filled=True
                    ,rounded=True
                    ,special_characters=True
                    ,feature_names = var
                    ,proportion=True
                    ,rotate=True
                    ,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # graph.write_png('tree.png')
    Image(graph.create_png())
    return classifier, Image(graph.create_png())


def get_bins(df,var,nbr_bins):
    #pd.qcut(factors, 5).value_counts() #fixed volume
    #pd.cut(factors, 5).value_counts() #fixed intervalls e.g. 80/5=16
    import pandas as pd
    for i in var:
        df[i+'_bin_vol'] = pd.qcut(df[i], nbr_bins)
        df[i+'_bin_int'] = pd.cut(df[i], nbr_bins)
        print(df[i+'_bin_vol'].value_counts())
        print(df[i+'_bin_int'].value_counts())


def get_var_value_counts(df,var):
    lst_var_with_value_counts=[]
    for i in var:
        lst_var_with_value_counts.append([i, len(list(df[i].value_counts(dropna=False).index)),list(df[i].value_counts(dropna=False).index),list(df[i].value_counts(dropna=False))])
    return lst_var_with_value_counts


# def get_var_value_counts(df,var):
#     lst_var_with_value_counts=[]
#     for i in var:
#         lst_var_with_value_counts.append([i, len(list(df[i].value_counts().index)),list(df[i].value_counts().index)])
#     return lst_var_with_value_counts


# def create_flags(x):
#     import re
#     import pandas as pd
#     if pd.isnull(x):
#         return 3
#     elif re.search(r'[A-Za-z]',x) != None:
#         return 1
#     elif re.search(r'[0-9]',x) != None:
#         return 2
#     else:
#         return 4


def create_flags(x):
    import re
    import pandas as pd
    if pd.isnull(x):
        return 'M'
    elif re.search(r'[A-Za-z]',x) != None:
        return 'C'
    elif re.search(r'[0-9]',x) != None:
        return 'N'
    else:
        return 'O'


def create_log_var(df, num_var):
    #do not create from negative variables
    import numpy as np
    #import seaborn as sns
    apply_value_log=[]
    from scipy.stats import skew
    skewed_feats = df[num_var].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    for i in skewed_feats.index:
        if skewed_feats[i]>=0.7:
            #   sns.displot(df[i])
            df['log_'+i]=np.log1p(df[i])
            # sns.displot(df['log_'+i])
            apply_value_log.append(i)
    return apply_value_log


def create_log_var_v2(df, num_var, factor=0.5):
    #do not create from negative variables
    import numpy as np
    #import seaborn as sns
    apply_value_log=[]
    from scipy.stats import skew
    skewed_feats = df[num_var].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    for i in skewed_feats.index:
        if skewed_feats[i]>=factor: #0.5 or 0.7
            #   sns.displot(df[i])
            df['log_'+i]=np.log1p(df[i])
            # sns.displot(df['log_'+i])
            apply_value_log.append(i)
    return apply_value_log


def replace_value(x, value):
    #can be used in lambda
    import numpy as np
    if x == value:
        x=np.nan
    else:
        x=x
    return x


def pre_eval_models(type_model, scoring, independent, dependent, cv):
    from sklearn.model_selection import cross_val_score
    
    out=[]
    if type_model=='regression':
        # from sklearn.ensemble import StackingRegressor
        #from mlxtend.regressor import StackingRegressor
        #from mlxtend.regressor import StackingCVRegressor
        
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from sklearn.svm import SVR
        
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Lasso        
        #from sklearn.linear_model import LassoCV
        from sklearn.linear_model import LassoLars
        from sklearn.linear_model import Ridge
        #from sklearn.linear_model import RidgeCV
        from sklearn.linear_model import BayesianRidge
        #from sklearn.linear_model import TweedieRegressor
        #from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import RANSACRegressor
        from sklearn.linear_model import SGDRegressor
        
        from sklearn.neighbors import KNeighborsRegressor
        
        from catboost import CatBoostRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import BaggingRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import RandomForestRegressor
        # from sklearn.ensemble import VotingRegressor
        # from sklearn.ensemble import HistGradientBoostingRegressor
        
        # from mlxtend.regressor import LinearRegression
        
        lst_models=[XGBRegressor(),LGBMRegressor(),SVR()
                    ,ElasticNet(),LinearRegression(),Lasso(),LassoLars(),Ridge()
                    ,BayesianRidge(),RANSACRegressor(),SGDRegressor(),KNeighborsRegressor()
                    ,CatBoostRegressor(),AdaBoostRegressor(),BaggingRegressor(),ExtraTreesRegressor()
                    ,GradientBoostingRegressor(),RandomForestRegressor() 
                    #, VotingRegressor()# ,HistGradientBoostingRegressor()
                    ]
    if type_model=='classification':
        # from sklearn.ensemble import StackingClassifier
        # from mlxtend.classifier import StackingClassifier
        # from mlxtend.classifier import StackingCVClassifier
        
        from xgboost.sklearn import XGBClassifier
        from lightgbm import LGBMClassifier
        # from sklearn import SVM 
        # from sklearn.svm import SVC
        # from sklearn.svm import LinearSVC
 
        from sklearn.linear_model import LogisticRegression 
        #from sklearn.linear_model import LogisticRegressionCV
        from sklearn.linear_model import Perceptron                           
        from sklearn.tree import DecisionTreeClassifier
        
        from sklearn.neighbors import KNeighborsClassifier
        
        from catboost import CatBoostClassifier
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import BaggingClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        # from sklearn.ensemble import VotingClassifier
        # from sklearn.ensemble import HistGradientBoostingClassifier
        
        #from mlxtend.classifier import Adaline
        #from mlxtend.classifier import EnsembleVoteClassifier
        #from mlxtend.classifier import LogisticRegression
        #from mlxtend.classifier import MultiLayerPerceptron
        #from mlxtend.classifier import OneRClassifier
        #from mlxtend.classifier import Perceptron
        #from mlxtend.classifier import SoftmaxRegression
        
        from sklearn.naive_bayes import GaussianNB 
        from sklearn.neural_network import MLPClassifier
        
        lst_models=[LogisticRegression(), LGBMClassifier(), Perceptron(), KNeighborsClassifier(), DecisionTreeClassifier()
             ,XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', silent=True)#, SVM(), LinearSVC()
             ,CatBoostClassifier(verbose=False),AdaBoostRegressor(),BaggingClassifier(),ExtraTreesClassifier()
             ,GradientBoostingClassifier(),RandomForestClassifier() 
             #,VotingClassifier()# ,HistGradientBoostingClassifier()
             ,GaussianNB(),MLPClassifier()
             ] 
    if type_model=='multiclass':
        from sklearn.naive_bayes import MultinomialNB
        from xgboost.sklearn import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        lst_models=[MultinomialNB(),XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                    ,LGBMClassifier(),CatBoostClassifier()]
                
    for model in lst_models:
        scores = cross_val_score(model, independent, dependent, scoring=scoring, cv=cv)
        out.append([str(model),scores.mean(), scores.std()])
    sort=sorted(out,key=lambda x: x[1],reverse=True)
    return sort


def evaluate_model(model_type, model, X, y_true):
    #OV: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    if model_type=='regression':
        import numpy as np
        from sklearn import metrics
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import cross_validate
        
        # cv_scr = cross_val_score(model, X, y_true, cv=5)
        # print('CV Score:', cv_scr.mean())
        
        cv = cross_validate(model, X, y_true, 
                            scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'], cv=2)
        mse_score = np.sqrt(-1 * cv['test_neg_mean_squared_error'].mean())
        mse_std = np.sqrt(cv['test_neg_mean_squared_error'].std())
        mae_score = -1 * cv['test_neg_mean_absolute_error'].mean()
        mae_std = cv['test_neg_mean_absolute_error'].std()
        r2_score_mean = cv['test_r2'].mean()
        r2_std = cv['test_r2'].std()
        print('CV RMSE: %.4f (%.4f)' % (mse_score, mse_std))
        print('CV MAE: %.4f (%.4f)' % (mae_score, mae_std))
        print('CV R^2: %.4f (%.4f)' % (r2_score_mean, r2_std))
        
        y_pred=model.predict(X)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        r2_square = metrics.r2_score(y_true, y_pred)
        neg_rmsle = -1 * np.sqrt(metrics.mean_squared_log_error(y_true, np.abs(y_pred)))
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('R2 Square', r2_square)
        print('Negative RMSLE', neg_rmsle)        
        
        
    if model_type=='classification':
        from sklearn.metrics import classification_report
        y_pred=model.predict(X)
        cr=classification_report(y_true=y_true,y_pred=y_pred)
        print(cr)
        
        from sklearn.metrics import f1_score
        f1=f1_score(y_true, y_pred)
        print(f1)
        
        from sklearn.metrics import log_loss     
        ll=log_loss(y_true, y_pred)
        print(ll)
        
        from sklearn.metrics import  confusion_matrix
        cm=confusion_matrix(y_true, y_pred)
        print(cm)
        
        from sklearn.metrics import accuracy_score
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        #Percentage of correct classification
        acc=accuracy_score(y_true, y_pred)
        print(acc)
        from sklearn.metrics import roc_curve, auc
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
        # fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
        # auc=auc(fpr, tpr)
        # print(auc)
        
    # if 'multilabel':
        
    # if 'clustering':


def determine_skewed_var(df, num_var, factor=0.5):
    out=[]
    from scipy.stats import skew
    skewed_feats = df[num_var].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    for i in skewed_feats.index:
        if skewed_feats[i]>=factor: #0.5 or 0.7
            out.append(i)
    return out


def create_normalized_skew_var(df,num_var):
    #https://www.youtube.com/watch?v=ev7wkRL8OUk
    #https://docs.scipy.org/doc/scipy/reference/stats.html
    #Unlike boxcox, yeojohnson does not require the input data to be positive.
    #Use case: especially for linear models, tree model should not get improved
    #Do not use: stats.yeojohnson (no inverse_transform)
    #in case of single var: pt.fit_transform(df['SalePrice'].to_numpy().reshape(-1,1))
    
    from  sklearn.preprocessing import PowerTransformer
    pt=PowerTransformer()
    pt.fit_transform(df[num_var])
    return pt


def sel_reg_model_features_v2(model,X_train_trans,Y_train,X_test_trans,Y_test,step,min_features_to_select):
    from sklearn.feature_selection import RFECV
    print('step: ', step)
    print('min_features_to_select: ', min_features_to_select)
    
    best_model=RFECV(estimator=model
          ,step=step
          ,min_features_to_select=min_features_to_select
          ,cv=3
          ,scoring='r2'
          ,verbose=0
          ,n_jobs=-1)
    best_model.fit(X_train_trans,Y_train)
    print(best_model.score(X_test_trans,Y_test))
    return best_model


def neg_rmsle(y_true, y_pred):
    y_pred = np.abs(y_pred)
    return -1 * np.sqrt(mean_squared_log_error(y_true, y_pred))


# the functions:
def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    '''
    source: https://www.kaggle.com/flaviobossolan/stratified-sampling-python
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator
    
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city 
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"
            
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        
        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    
    return stratified_df


def stratified_sample_report(df, strata, size=None):
    '''
    source: https://www.kaggle.com/flaviobossolan/stratified-sampling-python
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    source: https://www.kaggle.com/flaviobossolan/stratified-sampling-python
    A function to compute the sample size. If not informed, a sampling 
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n