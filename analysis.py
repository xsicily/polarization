import pandas as pd
from nltk import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

#----------------------------------Histogram of Topic frequency------------------------------------#
def get_topic_freq(file,keyword):
    data = pd.read_csv(file)
    data = data.dropna()
    data['text'] = data['text'].map(lambda x: x.lower())
    df_c = data[data['text'].str.contains(keyword)]
    df_c = df_c.reset_index()
    dd = df_c[(df_c['time'] >= "2016-04-17") & (df_c['time'] < "2016-09-26")]
    return len(dd)

keywords = ["climate", "women|woman|wife|female","family|families", "healthcare|health","trade", 
            "business|company|companies|firms|firm|entrepreneurship|entrepreneur|enterpreneurs",
            "job|jobs|employment|employ|employs|employing|employed", "tax|taxes", "violence|gun|weapon", 
            "terrorism|terrorist", "lgbt","gop", "obama|barack obama"]

freq_c = []
freq_t = []
for key in keywords:
    freq_c.append(get_topic_freq("clinton.csv", key))
    freq_t.append(get_topic_freq("trump.csv", key))

#plot the curve (not reported)
x = ['climate','women','family','healthcare','trade','business','job','tax','violence','terrorism','lgbt','gop','obama']
plt.plot(x,freq_c, c='m',marker = 'o', label = "@HillaryClinton") # A bar chart
plt.plot(x,freq_t, c='b',marker='o', label = "@realDonaldTrump")
plt.xlabel('Topics')
plt.ylabel('Frequency')
plt.xticks(rotation='vertical')
plt.title("Frequency of Topics")
plt.legend()
plt.tight_layout()
plt.show()

#plot the histogram
labels = ['climate','women','family','healthcare','trade','business','job','tax','violence','terrorism','lgbt','gop','obama']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, freq_c, width, label = "@HillaryClinton")
rects2 = ax.bar(x + width/2, freq_t, width, label = "@realDonaldTrump")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Frequency')
ax.set_xlabel('Topics')
ax.set_title('Frequency of Topics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xticks(rotation=70)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

#-------------------------------------------Number of tweets overtime--------------------------------------------------#
def get_topics_freq(file,date_1, date_2):
    data = pd.read_csv(file)
    data = data.dropna()
    data['text'] = data['text'].map(lambda x: x.lower())
    d = data[(data['time'] >= date_1) & (data['time'] < date_2)]
    return len(d)

def time_slot_pos(file):
    p1 = get_topics_freq(file, "2016-04-17","2016-04-25")
    p2 = get_topics_freq(file, "2016-04-25","2016-05-03")
    p3 = get_topics_freq(file, "2016-05-03","2016-05-11")
    p4 = get_topics_freq(file, "2016-05-11","2016-05-19")
    p5 = get_topics_freq(file, "2016-05-19","2016-05-27")
    p6 = get_topics_freq(file, "2016-05-27","2016-06-04")
    p7 = get_topics_freq(file, "2016-06-04","2016-06-12")
    p8 = get_topics_freq(file, "2016-06-12","2016-06-20")
    p9 = get_topics_freq(file, "2016-06-20","2016-06-28")
    p10 = get_topics_freq(file, "2016-06-28","2016-07-06")
    p11 = get_topics_freq(file, "2016-07-06","2016-07-14")
    p12 = get_topics_freq(file, "2016-07-14","2016-07-22")
    p13 = get_topics_freq(file, "2016-07-22","2016-07-30")
    p14 = get_topics_freq(file, "2016-07-30","2016-08-07")
    p15 = get_topics_freq(file, "2016-08-07","2016-08-15")
    p16 = get_topics_freq(file, "2016-08-15","2016-08-23")
    p17 = get_topics_freq(file, "2016-08-23","2016-08-31")
    p18 = get_topics_freq(file, "2016-09-01","2016-09-09")
    p19 = get_topics_freq(file, "2016-09-09","2016-09-17")
    p20 = get_topics_freq(file, "2016-09-17","2016-09-25")
    return [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20]

x = ['Apr25','May3','May11','May19','May27','Jun4','Jun12','Jun20','Jun28','Jul6','Jul14','Jul22','Jul30','Aug7','Aug15','Aug23','Sep1','Sep9','Sep17','Sep25']
plt.plot(x,time_slot_pos('clinton.csv'), c='m',marker = 'o', label = "@HillaryClinton") # A bar chart
plt.plot(x,time_slot_pos('trump.csv'), c='b',marker='o', label = "@realDonaldTrump")
plt.xlabel('Date')
plt.ylabel('Number of tweets')
plt.xticks(rotation=70)
plt.title("Number of tweets")
plt.legend()
plt.tight_layout()
plt.show()

#-----------------------------topic probability for each time period----------------------------#
def get_wrds_freq(file,sub_file, keyword, date_1, date_2):
    data_t = pd.read_csv(file)
    data_t = data_t.dropna()
    data = pd.read_csv(sub_file)
    data = data.dropna()
    data['text'] = data['text'].map(lambda x: x.lower())
    df_c = data[data['text'].str.contains(keyword)]
    #df_c = df_c.reset_index()
    dd = df_c[(df_c['time'] >= date_1) & (df_c['time'] < date_2)]
    return len(dd)/len(data_t)

def pro(file, sub_file,keyword):
    p1 = get_wrds_freq(file, sub_file,keyword,"2016-04-17","2016-04-25")
    p2 = get_wrds_freq(file, sub_file,keyword,"2016-04-25","2016-05-03")
    p3 = get_wrds_freq(file, sub_file,keyword,"2016-05-03","2016-05-11")
    p4 = get_wrds_freq(file, sub_file,keyword,"2016-05-11","2016-05-19")
    p5 = get_wrds_freq(file, sub_file,keyword,"2016-05-19","2016-05-27")
    p6 = get_wrds_freq(file, sub_file,keyword,"2016-05-27","2016-06-04")
    p7 = get_wrds_freq(file, sub_file,keyword,"2016-06-04","2016-06-12")
    p8 = get_wrds_freq(file, sub_file,keyword,"2016-06-12","2016-06-20")
    p9 = get_wrds_freq(file, sub_file,keyword,"2016-06-20","2016-06-28")
    p10 = get_wrds_freq(file, sub_file,keyword,"2016-06-28","2016-07-06")
    p11 = get_wrds_freq(file, sub_file,keyword,"2016-07-06","2016-07-14")
    p12 = get_wrds_freq(file, sub_file,keyword,"2016-07-14","2016-07-22")
    p13 = get_wrds_freq(file, sub_file,keyword,"2016-07-22","2016-07-30")
    p14 = get_wrds_freq(file, sub_file,keyword,"2016-07-30","2016-08-07")
    p15 = get_wrds_freq(file, sub_file,keyword,"2016-08-07","2016-08-15")
    p16 = get_wrds_freq(file, sub_file,keyword,"2016-08-15","2016-08-23")
    p17 = get_wrds_freq(file, sub_file,keyword,"2016-08-23","2016-08-31")
    p18 = get_wrds_freq(file, sub_file,keyword,"2016-09-01","2016-09-09")
    p19 = get_wrds_freq(file, sub_file,keyword,"2016-09-09","2016-09-17")
    p20 = get_wrds_freq(file, sub_file,keyword,"2016-09-17","2016-09-25")
    return [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20]

def topic_plot(keyword, topic):
    x = ['Apr25','May3','May11','May19','May27','Jun4','Jun12','Jun20','Jun28','Jul6','Jul14','Jul22','Jul30','Aug7','Aug15','Aug23','Sep1','Sep9','Sep17','Sep25']
    y1 = pro("tweets_part.csv","clinton.csv", keyword)
    y2 = pro("tweets_part.csv","trump.csv", keyword)
    plt.plot(x,y1,linestyle='--', marker='o', c = 'm',label = "@HillaryClinton")
    plt.plot(x,y2,linestyle='--', marker='o', c = 'b', label = "@realDonaldTrump")
    plt.xlabel('Date')
    plt.ylabel('P')
    plt.title('Divergence - ' + topic)
    plt.tight_layout()
    plt.legend()
    plt.xticks(rotation=70)
    plt.show()

#plot the probability for each topic
topic_plot("climate",'climate')
topic_plot("women|woman|wife|female",'women')
topic_plot("family|families",'family')
topic_plot("healthcare|health",'healthcare')
topic_plot("trade",'trade')
topic_plot("business|company|companies|firms|firm|entrepreneurship|entrepreneur|enterpreneurs",'business')
topic_plot("job|jobs|employment|employ|employs|employing|employed",'job')
topic_plot("tax|taxes",'tax')
topic_plot("violence|gun|weapon",'violence')
topic_plot('terrorism|terrorist','terrorism')
topic_plot("lgbt",'lgbt')
topic_plot("gop",'gop')
topic_plot('obama|barack obama','obama')


#----------------------------------------------BPI for each topic---------------------------------------------#
keywords_1 = ["women|woman|wife|female","family|families", "healthcare|health","trade", 
            "business|company|companies|firms|firm|entrepreneurship|entrepreneur|enterpreneurs",
            "job|jobs|employment|employ|employs|employing|employed", "tax|taxes", "violence|gun|weapon", 
            "terrorism|terrorist", "lgbt","gop", "obama|barack obama"]

sim = []
for keyword in keywords_1:
    y1 = pro("tweets_part.csv","clinton.csv", keyword)
    y2 = pro("tweets_part.csv","trump.csv", keyword)
    result = 1 - spatial.distance.cosine(y1, y2)
    sim.append(result)

#plot BPI (not reported)
x = ['women','family','healthcare','trade','business','job','tax','violence','terrorism','lgbt','gop','obama']
plt.plot(x, sim, '--r', marker='o', markersize=10, c = 'black')
plt.xlabel('Topics')
plt.ylabel('Distance')
plt.title('Belief polarization index (BPI) of topics')
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()

#--------------------------------------------sentiment analysis------------------------------------------#
#Note: sentiment scores were computed by sentiStrength in advance

#plot sentiment scores of "obama" - Hillary Clinton
data = pd.read_csv('obama_ccc.csv')
#data = data[(data['Sentiment'] > 0)]
x = [i for i in range(1,len(data)+1)]
score_1 = data['Sentiment']
plt.plot(x,score_1, marker = 'o',c = 'm', label= "@HillaryClinton")
plt.xlabel('Number of tweets')
plt.ylabel('Sentiment score')
plt.xticks(rotation=70)
plt.title("Overall sentiment scores - obama")
plt.legend()
plt.tight_layout()
plt.show()

#plot sentiment scores of "obama" - Donald Trump
data = pd.read_csv('obama_ttt.csv')
#data = data[(data['Sentiment'] < 0)]
x = [i for i in range(1,len(data)+1)]
score_2 = data['Sentiment']
#data['time'] = data['time'].map(lambda x: pd.to_datetime(x).date())
#data.to_csv('obama_ccc.csv', index = False)
plt.plot(x,score_2, marker = 'o',c = 'b', label= "@realDonaldTrump")
plt.xlabel('Number of tweets')
plt.ylabel('Sentiment score')
plt.xticks(rotation=70)
plt.title("Overall sentiment scores - obama")
plt.legend()
plt.tight_layout()
plt.show()

##statistical analysis of sentiment scores
from scipy import stats
import numpy as np
from scipy.stats import skew, kurtosis

def get_stats(file):
    data = pd.read_csv(file)
    #data = data[(data['Sentiment'] < 0)]
    sentiment = data['Sentiment']
    mean_val = np.mean(sentiment)
    min_val = min(sentiment)
    val_25 = np.quantile(sentiment, 0.25)
    median_val = np.median(sentiment)
    val_75 = np.quantile(sentiment, 0.75)
    max_val = max(sentiment)
    std_val = np.std(sentiment)
    skew_val = skew(sentiment)
    kurto_val = kurtosis(sentiment)
    return [mean_val, min_val, val_25,median_val, val_75, max_val,std_val,skew_val,kurto_val]

get_stats('terro_cc.csv')
get_stats('terro_tt.csv')

#t-test
data = pd.read_csv('terro_cc.csv')
score_1 = data['Sentiment']
data = pd.read_csv('terro_tt.csv')
score_2 = data['Sentiment']
stats.ttest_ind(score_1, score_2)


##-------summary of sentiment scores (not reported)--------#
def sentiment_percent(filename):
    data = pd.read_csv(filename, engine = 'python')
    sentiment = data['Sentiment']
    pos = sentiment[(sentiment > 0)]
    neg = sentiment[(sentiment < 0)]
    positive = len(pos)/len(sentiment)*100
    negative = len(neg)/len(sentiment)*100
    neutral = 100 - positive - negative
    return [positive, negative, neutral]

#for each topic from Hillary Clinton
c_climate = sentiment_percent('c_climate.csv')
c_women = sentiment_percent('c_women.csv')
c_family = sentiment_percent('c_family.csv')
c_healthcare = sentiment_percent('c_health.csv')
c_business = sentiment_percent('c_business.csv')
c_job = sentiment_percent('c_job.csv')
c_tax = sentiment_percent('c_tax.csv')
c_violence = sentiment_percent('c_violence.csv')
c_terrorism = sentiment_percent('c_terrorism.csv')
c_lgbt = sentiment_percent('c_lgbt.csv')
c_gop = sentiment_percent('c_gop.csv')
c_obama = sentiment_percent('c_obama.csv')
c_trade = sentiment_percent('c_trade.csv')
c_aca = sentiment_percent('c_aca.csv')
c_immigrants = sentiment_percent('c_immigrants.csv')

#for each topic from Donald Trump
t_women = sentiment_percent('t_women.csv')
t_family = sentiment_percent('t_family.csv')
t_healthcare = sentiment_percent('t_health.csv')
t_business = sentiment_percent('t_business.csv')
t_job = sentiment_percent('t_job.csv')
t_tax = sentiment_percent('t_tax.csv')
t_violence = sentiment_percent('t_violence.csv')
t_terrorism = sentiment_percent('t_terrorism.csv')
#t_lgbt = sentiment_percent('t_lgbt.csv')
t_lgbt = [0.0,100.0,0.0]
t_gop = sentiment_percent('t_gop.csv')
t_obama = sentiment_percent('t_obama.csv')
t_trade = sentiment_percent('t_trade.csv')
t_aca = sentiment_percent('t_aca.csv')
t_immigrants = sentiment_percent('t_immigrants.csv')


#---------------------------------semantic role labelling--------------------------------#
#Note: semantic role labelling was computed by SENNA in advance
#job
x = ['A0','A1','A2','A3','A4']
y1 = [39.52, 40.48,18.01,1.43,0.47]
y2 = [38.55,42.77,13.25,3.61,1.82]
plt.plot(x,y1,linestyle='--', marker='o', c = 'm',label = "@HillaryClinton")
plt.plot(x,y2,linestyle='--', marker='o', c = 'b', label = "@realDonaldTrump")
plt.xlabel('Arguments')
plt.ylabel('Percentage')
plt.title('Semantic role labelling - arguments for topic "job"')
plt.tight_layout()
plt.legend()
#plt.xticks(rotation=70)
plt.show()

#terrorism
x = ['A0','A1','A2','A3','A4']
y1 = [40.54, 37.84,20.27,1.35,0]
y2 = [39.02,41.46,17.07,2.45,0]
plt.plot(x,y1,linestyle='--', marker='o', c = 'm',label = "@HillaryClinton")
plt.plot(x,y2,linestyle='--', marker='o', c = 'b', label = "@realDonaldTrump")
plt.xlabel('Arguments')
plt.ylabel('Percentage')
plt.title('Semantic role labelling - arguments for topic "terrorism"')
plt.tight_layout()
plt.legend()
#plt.xticks(rotation=70)
plt.show()

#obama
y1 = [41.03, 41.03,15.38,1.28,1.28]
y2 = [38.21,38.21,19.51,3.25,0.82]
plt.plot(x,y1,linestyle='--', marker='o', c = 'm',label = "@HillaryClinton")
plt.plot(x,y2,linestyle='--', marker='o', c = 'b', label = "@realDonaldTrump")
plt.xlabel('Arguments')
plt.ylabel('Percentage')
plt.title('Semantic role labelling - arguments for topic "obama"')
plt.tight_layout()
plt.legend()
#plt.xticks(rotation=70)
plt.show()

#---------------------------------argument mining--------------------------------#
#Note: argument was tagged by TARGER

