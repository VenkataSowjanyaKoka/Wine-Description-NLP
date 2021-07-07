```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>



```python
#1.Problem Statement
#Predict US or Non-US origin of wine(implying wine US or Non-US made) based on,
#words that are included in description(reviviews) written by wine critics.
```


```python
#2.Statement/Comment on the data collection (in terms of what is needed) and scope.
#Need written description/review that include some indication of places/creation/availablity of ingredients 
#to understand the origin of the wine
```


```python
# We are using a data file(csv format) that contains curated data on Wines Origin and Description. 
#Contains description text and origin value US or Non-US
#Origin valule - US(Wine is originated or is US made)
#Origin valule - Non-US(Wine is originated or is Non-US made)
```


```python
#3.Preliminary decision about the ML algorithm you will use. Justify your decision to use this particular algorithm
#From problem statement predict means, need to preform predictive analysis
#Predictive analysis comes under supervised learning(labeled data)
#Example Regression, Logistic regression, Decision Trees, Random Forests...
```


```python
#From Cheat Sheet
#Number of samples in the data? - More than 50
#Predicting a Category? - Yes(Us or Non-US)
#Having Labeled data? -Yes(Leading to the classification category)
```


```python
#Classification considering - Logistic Regression(Classifying based on prediting variable)
#Predicting Origin - Origin becomes the label variabe, vector of tokens from description becomes independent variable
#Origin and tokens both are of string type and should be converted to numeric datatype for for performing logistic regression
#Since there are only two classification US or Non-US we just assign numeric values 1 and 0 for US and Non-Us respectively instead of using dummy variables
```


```python
#4.Statement/comments on data preparation and clean up. 
#What do you need to do to get data ready for the ML algorithm of your choice.
#Origin and tokens(Description) both are of string type and should be converted to numeric datatype for performing logistic regression
```


```python
#for structured data we use Spark SQL, SparkSession acts a pipeline between data and sql statements
from pyspark.sql import SparkSession
```


```python
# sparksession is like a class and we need to create an instance of a class to utilize
spark = SparkSession.builder.appName("NLP_5B_Data_Processing").getOrCreate()
```


```python
#Reading the csv file data(Creating Corpus)
Wine_DF = spark.read.csv("/Users/sowjanyakoka/Desktop/Machine_Learning/Data/WineReviews.csv", inferSchema = True, header = True)
```


```python
#Seeing the shape of the dataset
print("Shape:", (Wine_DF.count(), len(Wine_DF.columns)))
```

    Shape: (150935, 2)



```python
#Looking at the schema
#both columns are of string type
Wine_DF.printSchema()
```

    root
     |-- Origin: string (nullable = true)
     |-- Description: string (nullable = true)
    



```python
#Loding the random function
from pyspark.sql.functions import rand 
#Displaying random observations from the data
Wine_DF.orderBy(rand()).show(10,False)  # Note Origin values are read in as string
```

    +------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Origin|Description                                                                                                                                                                                                                                                                                                                                                                                   |
    +------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Non-US|Sweet and fragrant, floral and fruity, this Prosecco Extra Dry (with 10% Pinot Bianco) has a feminine, creamy disposition and excellent harmony of aromas and effervescence. It's very filling in the mouth with generous accents of stone fruit and honey; “47” marks the year the winery was founded.                                                                                       |
    |US    |Dusty sawdust aromas run heavy on the nose, with peach and nectarine aromas sitting in reserve. Fairly reserved in terms of flavors, with light fruit duking it out with the more aggressive wood notes. Where it does its best is in the area of balance: citric acids even out any wood and weight.                                                                                         |
    |US    |The '09 Continuum proves the adage that a great winery will produce fine wine even in a compromised vintage—such as 2009, with its cool conditions and rain during harvest. With dense tannins and a solid core of blackberry and spice flavors, the wine shows lovely complexity, with firm and masculine minerality. Still, it's gritty and too young now to enjoy. Cellar it for 6–8 years.|
    |US    |Lodi isn't the first region that comes to mind when it comes to California Syrah, but this one could change minds. It has a mountainous mouthfeel, with flavors of tar, blackberry and spice. The wine's aromas need coaxing, but once they surface, a firm and fruity wine emerges.                                                                                                          |
    |US    |Aromas of smoke and sweet apple introduce an intensely flavored palate that's full of ripe yellow peach, waxy flower and fresh fennel. Off-dry and bold in structure, it finishes with a hint of lanolin and a lingering minerality.                                                                                                                                                          |
    |US    |). Vineyard 7&8's Spring Mountain Estate Cabernets require aging to show their best, and so it is with this 2009. It's dry, tannic and unrewarding now due to its astringency and acidity. But it has a terrific heart of blackberries and minerals that's vast and rich—indicative of great growing conditions. It should begin to come into its own after 2015.                             |
    |Non-US|Light floral and sweet herb aromas open to a mouthful of grapefruit and mineral flavors in this clean, crisp but not sharp white. The finish is tangy and bright, with stony notes. All in all an enjoyable and refreshing light Chardonnay, with some complexity and style.                                                                                                                  |
    |Non-US|This is a spicy toasty wine, showing the wood-aging flavors strongly at this young stage. It has spice, dark tannins and bitter coffee flavors. That suggests the wine could remain firm and with a dry edge. The juicy acidity certainly needs time to fill out. Drink from 2021.                                                                                                            |
    |Non-US|The nose of this wine is richly redolent of jasmine, honey and peach and the overall effect is very floral indeed. This Malvasia-Chardonnay blend is lean and flavorful in the mouth.                                                                                                                                                                                                         |
    |US    |Gentle and rich, this Cab shows smooth, sweet tannins and a nice touch of oak. The underlying fruit flavors are of blackberries and cassis, with a touch of cinnamon-dusted mocha coffee.                                                                                                                                                                                                     |
    +------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
Wine_DF.describe().show()
```

    +-------+--------------------+--------------------+
    |summary|              Origin|         Description|
    +-------+--------------------+--------------------+
    |  count|              150935|              150935|
    |   mean|                null|                null|
    | stddev|                null|                null|
    |    min|              Non-US| and the note of ...|
    |    max|steak. It has fir...|“Zéro Dosage” mea...|
    +-------+--------------------+--------------------+
    



```python
#Checking how many distinct origins are there in our data
Distinct_Origins = Wine_DF.select('Origin').distinct().count()
print("Number of Tdistinct origins :", Distinct_Origins )
```

    Number of Tdistinct origins : 7



```python
#Looking at number of observations based on origin
Wine_DF.groupBy('Origin').count().show()
```

    +--------------------+-----+
    |              Origin|count|
    +--------------------+-----+
    |influence. This i...|    1|
    |              Non-US|88533|
    |steak. It has fir...|    1|
    |blackberry and bl...|    1|
    |                  US|62397|
    |     anti-Chardonnay|    1|
    |             elegant|    1|
    +--------------------+-----+
    



```python
#5.	Perform data prep and clean-up. 
```


```python
#Data Cleaning
```


```python
#Since we are classifying between US and Non-US,
#Filtering the data to remove other five origin data
#Which is irrelavent to our model, does not indicate any origin and each of them have only one observation
```


```python
#Filtering the data for data only with only either US or Non-US sentiment origin
Wine_Origin_DF = Wine_DF.filter((Wine_DF['Origin']=='US')|(Wine_DF['Origin']=='Non-US'))
```


```python
#Checking the count to see if any rows are deleted (rows with different origins)
#Only 5 records are deleted
Wine_Origin_DF.count()
```




    150930




```python
#Grouping by Origin to see balance of data
Wine_Origin_DF.groupBy('Origin').count().show()
```

    +------+-----+
    |Origin|count|
    +------+-----+
    |Non-US|88533|
    |    US|62397|
    +------+-----+
    



```python
#(Fairly balanced)
#Non-US--(88533/150930)*100--58.65%
#US--(62397/150930)*100--41.34%
```


```python
#Data Preperation for our model
```


```python
#Looking at the schema again
#Both are of string types
Wine_Origin_DF.printSchema()
```

    root
     |-- Origin: string (nullable = true)
     |-- Description: string (nullable = true)
    



```python
# Origin is not of a numeric type, which is required by Logistic Rrgression. So, we have to convert it. 
# Since we will end up with duplicate columns with the same data, we will drop the origin column
```


```python
#in order to perform logistic regression 
#we should have Origin value of numeric datatype
#Adding a column label to store converted float values 
#from string value in origin (and dropping the origin(String type) column)
```


```python
#Importing functions to assign numeric values to Origin (US,Non-US)
from pyspark.sql.functions import col, when
```


```python
#Creating a new label column with indication 1 for wine with US origin and 0 for wine Non-US origin
#Adding a column label to covert the string origin values to float values 
New_Wine_Origin_DF = Wine_Origin_DF.withColumn("Origin_Label", when(col("Origin")=='US', 1.0).otherwise(0.0))
```


```python
#Looking at the schema again
New_Wine_Origin_DF.printSchema()
```

    root
     |-- Origin: string (nullable = true)
     |-- Description: string (nullable = true)
     |-- Origin_Label: double (nullable = false)
    



```python
#Displaying random data
#Verifying the assigned values
New_Wine_Origin_DF.orderBy(rand()).show(10)
```

    +------+--------------------+------------+
    |Origin|         Description|Origin_Label|
    +------+--------------------+------------+
    |Non-US|Sweet and funky s...|         0.0|
    |Non-US|This is a tasty w...|         0.0|
    |Non-US|A very mineral an...|         0.0|
    |Non-US|Despite being loa...|         0.0|
    |Non-US|According to the ...|         0.0|
    |Non-US|A 30-year-old taw...|         0.0|
    |Non-US|Tight and packed ...|         0.0|
    |    US|This is rife with...|         1.0|
    |    US|With 15.7% alcoho...|         1.0|
    |    US|One of the better...|         1.0|
    +------+--------------------+------------+
    only showing top 10 rows
    



```python
#Conduct Explanatory data analysis (EDA). Comment on data distribution. 
```


```python
#Checking for the balance after transformation
New_Wine_Origin_DF.groupBy('Origin','Origin_Label').count().show()
```

    +------+------------+-----+
    |Origin|Origin_Label|count|
    +------+------------+-----+
    |Non-US|         0.0|88533|
    |    US|         1.0|62397|
    +------+------------+-----+
    



```python
#(Fairly balanced)
#Non-US--(88533/150930)*100--58.65%
#US--(62397/150930)*100--41.34%
#To make the model learn with both US and Non-US scenarios equally we need to have a balance
```


```python
# Since we will end up with duplicate columns with same data, we will drop the Origin column
#Deleting the columns
Wine_Des_DF = New_Wine_Origin_DF.drop('Origin')
```


```python
#Checking for the schema again
Wine_Des_DF.printSchema()
```

    root
     |-- Description: string (nullable = true)
     |-- Origin_Label: double (nullable = false)
    



```python

```


```python
#Displaying the data
Wine_Des_DF.orderBy(rand()).show(10, truncate = False)
```

    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+
    |Description                                                                                                                                                                                                                                                                                                                                                                                               |Origin_Label|
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+
    |This new Uriah marks a return to form; big, brawny, chewy and complex, yet generous and complete. The blend continues as before—54% Merlot, 33% Cab Franc, 7% Petit Verdot, and 6% Malbec, all estate-grown. The wine opens with barrel scents and flavors of tea, tobacco and wood smoke. The cherry-flavored fruit saturates the tobacco notes, gaining mass and turning into cherry cobbler as it does.|1.0         |
    |The mouthfeel of this wine is beautiful, with a velvety texture and dusty tannins, and the lingering, coffee-infused finish seals the experience nicely. The attractive aromas and flavors of small red berries, cherry and red plum skins help to make this an easy-drinking choice with pairing flexibility.                                                                                            |0.0         |
    |Dark, jammy, thick on the tongue and bursting with fresh fruit, this Zin is likeable from the get go. It's not subtle about the way it delivers its no-holds-barred blueberry, blackberry and cherry flavors, wrapped in rich tannins.                                                                                                                                                                    |1.0         |
    |Neutral in the extreme, this is an easy pick for a large crowd. It's light on its feet and quite crisp with an over-arching theme of nuts and just a faint hint of citrus. Drink up.                                                                                                                                                                                                                      |0.0         |
    |Familiar aromas of lemon blossom and air freshener get it going, and all together it smells a lot like a bathroom cleanser. Later on it shifts to lean, lemony, citrus fruits, with lime sticking out. Good and racy in the mouth, with another squeeze of lime juice on the finish. Good for Torrontés but with limits.                                                                                  |0.0         |
    |Conventional wisdom is that 2004 was a better Margaret River vintage than 2003, but this wine suffers a bit in comparison to its older sibling. Thankfully, it's still very good, with plentiful cassis and red plum fruit, soft tannins and crisp acids.                                                                                                                                                 |0.0         |
    |A lovely Pinot that's dry and silky. The rich, ripe raspberry and cherry flavors are enriched with oak, and girded by its firm minerality. Easy to drink now, but it should age over the next six years.                                                                                                                                                                                                  |1.0         |
    |A touch of rounded Chardonnay flavors and aromas appear in this Moldovan white, but the overall character is lean and slightly flimsy. The wine needs muscle, though the fruit is good.                                                                                                                                                                                                                   |0.0         |
    |This négociant's top cuvée of Saint-Joseph has vibrant mixed berry notes that pick up lovely hints of licorice and garrigue, then turn chocolaty and velvety on the finish. Nicely done all the way around, with a drinking window extending from now through 2017.                                                                                                                                       |0.0         |
    |This is the most expensive of the winery's new Cabs. It's also the fruitiest and most in need of time in the cellar. Right now, it's a candied blast of blackberries and cherries. The tannins are sweet and chunky, and the oak isn't integrated with everything else. It's awkward. But it should develop for a good 6–8 years, maybe even longer. The score reflects its potential.                    |1.0         |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+
    only showing top 10 rows
    



```python
#Importing functions to remove punctuations and chech changes with no punctuation values(EDA)
from pyspark.sql.functions import regexp_replace, trim, col, lower
```


```python
#Define a function for removing punctuations
def removePunctuation(column):
    return trim(lower(regexp_replace(column, '([^\s\w_]|_)+', '')))          
#these use a predefined set of punctuation characters to replace with ''
```


```python
#Apply this function to the 'Description' column in your wine description dataframe, prior to tokenization.
#We'll first remove the punctuation from the original description text and 
#then tokenize the text after the punctuation is removed.

Wine_Des_DF = Wine_Des_DF.withColumn('Description_NoPunct', removePunctuation(col('Description')))

#This will add a new column named 'review_nopunct' to the textRevs_df. This is the column you would then use as inputCol for the Tokenzier

```


```python
Wine_Des_DF.printSchema()
```

    root
     |-- Description: string (nullable = true)
     |-- Origin_Label: double (nullable = false)
     |-- Description_NoPunct: string (nullable = true)
    



```python
#Checking for the values after transformation
Wine_Des_DF.groupBy('Origin_Label').count().show()
```

    +------------+-----+
    |Origin_Label|count|
    +------------+-----+
    |         0.0|88533|
    |         1.0|62397|
    +------------+-----+
    



```python
# Adding length column to the dataframe
#Length of the description might matter because repetition of words would occur in the same description
#Loading length function 
from pyspark.sql.functions import length
```


```python
#For each row calculating length of description and adding it to a new column
#For each row calculating length of description with no punctuation and adding it to a new column
Wine_Des_DF = Wine_Des_DF.withColumn('Length',length(Wine_Des_DF['Description']))
Wine_Des_DF = Wine_Des_DF.withColumn('Length_NoPunct',length(Wine_Des_DF['Description_NoPunct']))
```


```python
#Displaying the data
Wine_Des_DF.orderBy(rand()).show(5)
```

    +--------------------+------------+--------------------+------+--------------+
    |         Description|Origin_Label| Description_NoPunct|Length|Length_NoPunct|
    +--------------------+------------+--------------------+------+--------------+
    |Ripe and toasty, ...|         0.0|ripe and toasty t...|   230|           222|
    |Dark concentratio...|         0.0|dark concentratio...|   245|           239|
    |Here's a genuine ...|         0.0|heres a genuine a...|   248|           241|
    |Good raspberry, c...|         1.0|good raspberry ch...|   174|           168|
    |Tart but full cra...|         1.0|tart but full cra...|   288|           283|
    +--------------------+------------+--------------------+------+--------------+
    only showing top 5 rows
    



```python
#Average length of a description for a 0 and 1 orgin(Non-US and US)
#Fairly close
Wine_Des_DF.groupBy('Origin_Label').agg({'Length':'mean'}).show()
```

    +------------+------------------+
    |Origin_Label|       avg(Length)|
    +------------+------------------+
    |         0.0| 239.2192854641772|
    |         1.0|241.90622946616023|
    +------------+------------------+
    



```python
#Average length of a description with no punctuation for a 0 and 1 orgin(Non-US and US)
#Fairly close
Wine_Des_DF.groupBy('Origin_Label').agg({'Length_NoPunct':'mean'}).show()
```

    +------------+-------------------+
    |Origin_Label|avg(Length_NoPunct)|
    +------------+-------------------+
    |         0.0| 231.65171179108356|
    |         1.0| 233.34804557911437|
    +------------+-------------------+
    



```python
#7.Comment on required data transformation needed to get the data ready for input to the ML algorithm of your choice.
#Remember, all data available in the CSV document are of string data type. 
```


```python
# Data Preprocessing for NLP
```


```python
#Tokenization
```


```python
#Importing the Tokenizer function
from pyspark.ml.feature import Tokenizer
```


```python
#Taking Description with no punctuation column and generating individual words (tokens)
# creating new column tokens for storing the tokens created from Description_NoPunct column
tokenization = Tokenizer(inputCol='Description_NoPunct',outputCol='Tokens')
```


```python
#Applying the Tokenizer function to the dataframe
Tokenized_DF = tokenization.transform(Wine_Des_DF)
```


```python
#looking at the tokens columns
Tokenized_DF.select('Tokens').show(10, truncate = False)
```

    +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Tokens                                                                                                                                                                                                                                                                                                                                                                                                                                           |
    +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[this, tremendous, 100, varietal, wine, hails, from, oakville, and, was, aged, over, three, years, in, oak, juicy, redcherry, fruit, and, a, compelling, hint, of, caramel, greet, the, palate, framed, by, elegant, fine, tannins, and, a, subtle, minty, tone, in, the, background, balanced, and, rewarding, from, start, to, finish, it, has, years, ahead, of, it, to, develop, further, nuance, enjoy, 20222030]                           |
    |[ripe, aromas, of, fig, blackberry, and, cassis, are, softened, and, sweetened, by, a, slathering, of, oaky, chocolate, and, vanilla, this, is, full, layered, intense, and, cushioned, on, the, palate, with, rich, flavors, of, chocolaty, black, fruits, and, baking, spices, a, toasty, everlasting, finish, is, heady, but, ideally, balanced, drink, through, 2023]                                                                        |
    |[mac, watson, honors, the, memory, of, a, wine, once, made, by, his, mother, in, this, tremendously, delicious, balanced, and, complex, botrytised, white, dark, gold, in, color, it, layers, toasted, hazelnut, pear, compote, and, orange, peel, flavors, reveling, in, the, succulence, of, its, 122, gl, of, residual, sugar]                                                                                                                |
    |[this, spent, 20, months, in, 30, new, french, oak, and, incorporates, fruit, from, ponzis, aurora, abetina, and, madrona, vineyards, among, others, aromatic, dense, and, toasty, it, deftly, blends, aromas, and, flavors, of, toast, cigar, box, blackberry, black, cherry, coffee, and, graphite, tannins, are, polished, to, a, fine, sheen, and, frame, a, finish, loaded, with, dark, chocolate, and, espresso, drink, now, through, 2032]|
    |[this, is, the, top, wine, from, la, bgude, named, after, the, highest, point, in, the, vineyard, at, 1200, feet, it, has, structure, density, and, considerable, acidity, that, is, still, calming, down, with, 18, months, in, wood, the, wine, has, developing, an, extra, richness, and, concentration, produced, by, the, tari, family, formerly, of, chteau, giscours, in, margaux, it, is, a, wine, made, for, aging, drink, from, 2020]  |
    |[deep, dense, and, pure, from, the, opening, bell, this, toro, is, a, winner, aromas, of, dark, ripe, black, fruits, are, cool, and, moderately, oaked, this, feels, massive, on, the, palate, but, sensationally, balanced, flavors, of, blackberry, coffee, mocha, and, toasty, oak, finish, spicy, smooth, and, heady, drink, this, exemplary, toro, through, 2023]                                                                           |
    |[slightly, gritty, blackfruit, aromas, include, a, sweet, note, of, pastry, along, with, a, hint, of, prune, walltowall, saturation, ensures, that, all, corners, of, ones, mouth, are, covered, flavors, of, blackberry, mocha, and, chocolate, are, highly, impressive, and, expressive, while, this, settles, nicely, on, a, long, finish, drink, now, through, 2024]                                                                         |
    |[lush, cedary, blackfruit, aromas, are, luxe, and, offer, notes, of, marzipan, and, vanilla, this, bruiser, is, massive, and, tannic, on, the, palate, but, still, lush, and, friendly, chocolate, is, a, key, flavor, while, baked, berry, and, cassis, flavors, are, hardly, wallflowers, on, the, finish, this, is, tannic, and, deep, as, a, sea, trench, drink, this, saturated, blackcolored, toro, through, 2023]                         |
    |[this, renamed, vineyard, was, formerly, bottled, as, delancellotti, youll, find, striking, minerality, underscoring, chunky, black, fruits, accents, of, citrus, and, graphite, comingle, with, exceptional, midpalate, concentration, this, is, a, wine, to, cellar, though, it, is, already, quite, enjoyable, drink, now, through, 2030]                                                                                                     |
    |[the, producer, sources, from, two, blocks, of, the, vineyard, for, this, wineone, at, a, high, elevation, which, contributes, bright, acidity, crunchy, cranberry, pomegranate, and, orange, peel, flavors, surround, silky, succulent, layers, of, texture, that, present, as, fleshy, fruit, that, delicately, lush, flavor, has, considerable, length]                                                                                       |
    +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
#To get a count of tokens for each description before removing the stop words
# importing size function from sql functions
from pyspark.sql.functions import size
```


```python
#Selecting all columns from dataframe and adding a new column based on no of tokens in each observation
# Size is a sql function to count number of items in a list
Tokenized_DF = Tokenized_DF.select('*',size('Tokens').alias('Tokens_Count'))
```


```python
#looking at the tokens and tokens count columns
#Tokenization converts sentences to lower case and then creates tokens
Tokenized_DF.select('Tokens','Tokens_Count').show(10)
```

    +--------------------+------------+
    |              Tokens|Tokens_Count|
    +--------------------+------------+
    |[this, tremendous...|          60|
    |[ripe, aromas, of...|          51|
    |[mac, watson, hon...|          47|
    |[this, spent, 20,...|          62|
    |[this, is, the, t...|          66|
    |[deep, dense, and...|          52|
    |[slightly, gritty...|          50|
    |[lush, cedary, bl...|          60|
    |[this, renamed, v...|          42|
    |[the, producer, s...|          45|
    +--------------------+------------+
    only showing top 10 rows
    



```python
#Removal of stopwords
```


```python
#Importing the StopWordsRemover function
from pyspark.ml.feature import StopWordsRemover
```


```python
#Taking Tokens column and creating new column Refined Tokens for storing the tokens after removal of stopwords
stopword_removal=StopWordsRemover(inputCol='Tokens',outputCol='Refined_Tokens')
```


```python
#Applying the StopWordsRemover function to the dataframe
Refined_DF = stopword_removal.transform(Tokenized_DF)
```


```python
#Selecting only the refined tokens column which has tokens after stop words have been removed
Refined_DF.select(['Refined_Tokens']).show(10)
```

    +--------------------+
    |      Refined_Tokens|
    +--------------------+
    |[tremendous, 100,...|
    |[ripe, aromas, fi...|
    |[mac, watson, hon...|
    |[spent, 20, month...|
    |[top, wine, la, b...|
    |[deep, dense, pur...|
    |[slightly, gritty...|
    |[lush, cedary, bl...|
    |[renamed, vineyar...|
    |[producer, source...|
    +--------------------+
    only showing top 10 rows
    



```python
#To get a count of tokens for each row after removing the stop words
#Selecting all columns from dataframe and adding a new column based on no of refined tokens in each observation
#Size is a sql function to count number of items in a list
Refined_DF = Refined_DF.select('*',size('Refined_Tokens').alias('Refined_Tokens_Count'))
```


```python
#Looking at the tokens,tokens count and refined tokens, refined tokens count columns
#To see if the counts vary which indicates removal of stop words in tokens 
Refined_DF.select('Tokens','Tokens_Count','Refined_Tokens','Refined_Tokens_Count').show(10)
```

    +--------------------+------------+--------------------+--------------------+
    |              Tokens|Tokens_Count|      Refined_Tokens|Refined_Tokens_Count|
    +--------------------+------------+--------------------+--------------------+
    |[this, tremendous...|          60|[tremendous, 100,...|                  36|
    |[ripe, aromas, of...|          51|[ripe, aromas, fi...|                  31|
    |[mac, watson, hon...|          47|[mac, watson, hon...|                  30|
    |[this, spent, 20,...|          62|[spent, 20, month...|                  43|
    |[this, is, the, t...|          66|[top, wine, la, b...|                  36|
    |[deep, dense, and...|          52|[deep, dense, pur...|                  34|
    |[slightly, gritty...|          50|[slightly, gritty...|                  31|
    |[lush, cedary, bl...|          60|[lush, cedary, bl...|                  35|
    |[this, renamed, v...|          42|[renamed, vineyar...|                  28|
    |[the, producer, s...|          45|[producer, source...|                  30|
    +--------------------+------------+--------------------+--------------------+
    only showing top 10 rows
    



```python
#Looking at random data
Refined_DF.orderBy(rand()).show(4, truncate = False)
```

    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
    |Description                                                                                                                                                                                                                                                                           |Origin_Label|Description_NoPunct                                                                                                                                                                                                                                                             |Length|Length_NoPunct|Tokens                                                                                                                                                                                                                                                                                                                             |Tokens_Count|Refined_Tokens                                                                                                                                                                                                               |Refined_Tokens_Count|
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
    |Dark in color, with aromas of bacon fat, blackberry candy and vanilla. The body is medium to full, with grabby tannins that support mildly medicinal blackberry and cherry flavors. Finishes clean and regular, with no nasty residue or forced oak.                                  |0.0         |dark in color with aromas of bacon fat blackberry candy and vanilla the body is medium to full with grabby tannins that support mildly medicinal blackberry and cherry flavors finishes clean and regular with no nasty residue or forced oak                                   |244   |237           |[dark, in, color, with, aromas, of, bacon, fat, blackberry, candy, and, vanilla, the, body, is, medium, to, full, with, grabby, tannins, that, support, mildly, medicinal, blackberry, and, cherry, flavors, finishes, clean, and, regular, with, no, nasty, residue, or, forced, oak]                                             |40          |[dark, color, aromas, bacon, fat, blackberry, candy, vanilla, body, medium, full, grabby, tannins, support, mildly, medicinal, blackberry, cherry, flavors, finishes, clean, regular, nasty, residue, forced, oak]           |26                  |
    |Les Grandes Lolières is a vineyard in the large Corton Grand Cru. This Pinot Noir shows a delicate balance between fine structure and beautiful red-fruit flavors. At the same time, it has weight and richness that lend this wine power as well as the chance to age for many years.|0.0         |les grandes lolires is a vineyard in the large corton grand cru this pinot noir shows a delicate balance between fine structure and beautiful redfruit flavors at the same time it has weight and richness that lend this wine power as well as the chance to age for many years|278   |272           |[les, grandes, lolires, is, a, vineyard, in, the, large, corton, grand, cru, this, pinot, noir, shows, a, delicate, balance, between, fine, structure, and, beautiful, redfruit, flavors, at, the, same, time, it, has, weight, and, richness, that, lend, this, wine, power, as, well, as, the, chance, to, age, for, many, years]|50          |[les, grandes, lolires, vineyard, large, corton, grand, cru, pinot, noir, shows, delicate, balance, fine, structure, beautiful, redfruit, flavors, time, weight, richness, lend, wine, power, well, chance, age, many, years]|29                  |
    |A zesty, lemon wine, with an intriguing series of herbal and animal flavors, this is not a wine to go unnoticed. It has weight, with spice and nut flavors adding to the fruit. Drink with shellfish or sharp cheese.                                                                 |0.0         |a zesty lemon wine with an intriguing series of herbal and animal flavors this is not a wine to go unnoticed it has weight with spice and nut flavors adding to the fruit drink with shellfish or sharp cheese                                                                  |213   |206           |[a, zesty, lemon, wine, with, an, intriguing, series, of, herbal, and, animal, flavors, this, is, not, a, wine, to, go, unnoticed, it, has, weight, with, spice, and, nut, flavors, adding, to, the, fruit, drink, with, shellfish, or, sharp, cheese]                                                                             |39          |[zesty, lemon, wine, intriguing, series, herbal, animal, flavors, wine, go, unnoticed, weight, spice, nut, flavors, adding, fruit, drink, shellfish, sharp, cheese]                                                          |21                  |
    |Does just what you want a Cab at this price to do, offering soft, immediate pleasure with extra points for complex elegance. You'll find cassis, cherry, sage and cedar flavors wrapped into ripely sweet tannins, with a finish of authority.                                        |1.0         |does just what you want a cab at this price to do offering soft immediate pleasure with extra points for complex elegance youll find cassis cherry sage and cedar flavors wrapped into ripely sweet tannins with a finish of authority                                          |238   |230           |[does, just, what, you, want, a, cab, at, this, price, to, do, offering, soft, immediate, pleasure, with, extra, points, for, complex, elegance, youll, find, cassis, cherry, sage, and, cedar, flavors, wrapped, into, ripely, sweet, tannins, with, a, finish, of, authority]                                                    |40          |[want, cab, price, offering, soft, immediate, pleasure, extra, points, complex, elegance, youll, find, cassis, cherry, sage, cedar, flavors, wrapped, ripely, sweet, tannins, finish, authority]                             |24                  |
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
    only showing top 4 rows
    



```python
#Looking at schema
Refined_DF.printSchema()
```

    root
     |-- Description: string (nullable = true)
     |-- Origin_Label: double (nullable = false)
     |-- Description_NoPunct: string (nullable = true)
     |-- Length: integer (nullable = true)
     |-- Length_NoPunct: integer (nullable = true)
     |-- Tokens: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- Tokens_Count: integer (nullable = false)
     |-- Refined_Tokens: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- Refined_Tokens_Count: integer (nullable = false)
    



```python
#8.Complete all needed data transformation. Use TF-IDF method of transforming token to their respective numeric values. 
```


```python
#Feature engineering
```


```python
#Term Frequency(TF) and Inverse Document Frequency(IDF)
```


```python
#Creating features based on TF-IDF in PySpark using the Refined dataframe
```


```python
#Imprting function for TF and IDF calculation
from pyspark.ml.feature import HashingTF,IDF
```


```python
#TERM FREQUENCY
#It is the score based on the number of times the word appears in current dataframe
```


```python
#Taking refined tokens column and creating new column tf features for storing the tf value created
hashing_vec=HashingTF(inputCol='Refined_Tokens',outputCol='TF_features')
```


```python
#Applying the HashingTF function to the dataframe
Hashing_DF = hashing_vec.transform(Refined_DF)
```


```python
#Looking at the refined tokens and corresoponding TF features columns
Hashing_DF.select(['Refined_Tokens','TF_features']).show(4, False)
```

    +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Refined_Tokens                                                                                                                                                                                                                                                                                                                                     |TF_features                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[tremendous, 100, varietal, wine, hails, oakville, aged, three, years, oak, juicy, redcherry, fruit, compelling, hint, caramel, greet, palate, framed, elegant, fine, tannins, subtle, minty, tone, background, balanced, rewarding, start, finish, years, ahead, develop, nuance, enjoy, 20222030]                                                |(262144,[5358,20495,28401,50323,64926,68345,74473,75898,84547,85321,88244,99511,109840,112796,120429,124795,125124,136793,144799,153032,156017,158571,158845,158931,170565,203214,212740,219140,219700,223329,225357,245453,245951,246030,261211],[1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])                                                                                   |
    |[ripe, aromas, fig, blackberry, cassis, softened, sweetened, slathering, oaky, chocolate, vanilla, full, layered, intense, cushioned, palate, rich, flavors, chocolaty, black, fruits, baking, spices, toasty, everlasting, finish, heady, ideally, balanced, drink, 2023]                                                                         |(262144,[5460,14060,20490,30365,55709,55788,79846,85829,89402,90757,95967,109975,111229,119267,131881,136793,147786,149300,151393,154749,155299,158845,192761,203214,203364,205876,213835,215425,224205,237021,254116],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])                                                                                                                              |
    |[mac, watson, honors, memory, wine, made, mother, tremendously, delicious, balanced, complex, botrytised, white, dark, gold, color, layers, toasted, hazelnut, pear, compote, orange, peel, flavors, reveling, succulence, 122, gl, residual, sugar]                                                                                               |(262144,[40266,43897,56749,57341,57508,61157,64289,88398,90748,93969,98087,99270,104786,131881,140784,145378,146009,148957,191337,203214,209402,223329,225667,229305,235618,238819,238835,244282,249846,261845],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])                                                                                                                                         |
    |[spent, 20, months, 30, new, french, oak, incorporates, fruit, ponzis, aurora, abetina, madrona, vineyards, among, others, aromatic, dense, toasty, deftly, blends, aromas, flavors, toast, cigar, box, blackberry, black, cherry, coffee, graphite, tannins, polished, fine, sheen, frame, finish, loaded, dark, chocolate, espresso, drink, 2032]|(262144,[15519,23762,25381,29945,32292,42080,43870,69299,78216,78329,79846,81046,84547,85829,88244,89402,90757,98627,107621,130598,130631,131881,141012,146328,147503,147786,157757,158845,168887,176683,189792,203659,205876,211900,219700,229305,238045,241259,245951,250282,250802,258633,259842],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    only showing top 4 rows
    



```python
#[da, vinci, code, book, awesome.]---(262144,[93284,111793,189113,212976,235054],[1.0,1.0,1.0,1.0,1.0])  
#262144 - Total number of tokens in the dataframe
#5358 - frequency of the word tremendous
#20495 - frequency of the word 100
#.....
#[1.0,1.0,1.0,1.0,1.0])-- list indicating the presence of the words [tremendous, 100, varietal, ..., ...] in the review with 1
```


```python
#INVERSE DOCUMENT FREQUENCY
#It is calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm
```


```python
#Taking TF features column and creating new column TF-IDF features for storing the TF-IDF value created
TF_IDF_vec=IDF(inputCol='TF_features',outputCol='TF_IDF_features')
```


```python
#Applying the IDF function to the dataframe
TF_IDF_DF = TF_IDF_vec.fit(Hashing_DF).transform(Hashing_DF)
```


```python
#Looking at the refined tokens and corresoponding TF-IDF features columns
#Multiplying these TF and IDF results in the TF-IDF score of a word in a document. 
#The higher the score, the more relevant that word is in that particular document.
TF_IDF_DF.select(['Refined_Tokens','TF_IDF_features']).show(10)
```

    +--------------------+--------------------+
    |      Refined_Tokens|     TF_IDF_features|
    +--------------------+--------------------+
    |[tremendous, 100,...|(262144,[5358,204...|
    |[ripe, aromas, fi...|(262144,[5460,140...|
    |[mac, watson, hon...|(262144,[40266,43...|
    |[spent, 20, month...|(262144,[15519,23...|
    |[top, wine, la, b...|(262144,[21336,25...|
    |[deep, dense, pur...|(262144,[5460,165...|
    |[slightly, gritty...|(262144,[11454,14...|
    |[lush, cedary, bl...|(262144,[4235,140...|
    |[renamed, vineyar...|(262144,[3189,319...|
    |[producer, source...|(262144,[15664,21...|
    +--------------------+--------------------+
    only showing top 10 rows
    



```python
TF_IDF_DF.printSchema()
```

    root
     |-- Description: string (nullable = true)
     |-- Origin_Label: double (nullable = false)
     |-- Description_NoPunct: string (nullable = true)
     |-- Length: integer (nullable = true)
     |-- Length_NoPunct: integer (nullable = true)
     |-- Tokens: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- Tokens_Count: integer (nullable = false)
     |-- Refined_Tokens: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- Refined_Tokens_Count: integer (nullable = false)
     |-- TF_features: vector (nullable = true)
     |-- TF_IDF_features: vector (nullable = true)
    



```python
#9.Apply the ML algorithm of your choice. Use an 8-/20 split for creating training and test dataset. 
```


```python
#Creating the model
Model = TF_IDF_DF.select(['TF_IDF_features','Origin_Label'])
```


```python
#Vectorizing
from pyspark.ml.feature import VectorAssembler
```


```python
df_assembler = VectorAssembler(inputCols=['TF_IDF_features'],outputCol='features_vec')
Model = df_assembler.transform(Model)
```


```python
#Displaying the model
Model.select(['Origin_Label','features_vec']).show(10, False)
```

    +------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Origin_Label|features_vec                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
    +------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |1.0         |(262144,[5358,20495,28401,50323,64926,68345,74473,75898,84547,85321,88244,99511,109840,112796,120429,124795,125124,136793,144799,153032,156017,158571,158845,158931,170565,203214,212740,219140,219700,223329,225357,245453,245951,246030,261211],[3.136137100314952,4.20633710576023,4.327683619575002,3.012374216099003,7.064765653357874,3.919878547169996,4.978831088101196,5.95587049773418,1.5778159356797066,5.700019628444186,1.0877853431501332,4.867541076021654,4.77387660012702,10.538283696599654,5.332904325710888,4.310265911267546,4.217965143755349,1.670116782364443,6.567991783047534,4.933401170598336,7.242446830595326,6.982935635110241,1.3943235406404146,3.4075848863059752,5.810895877887314,2.8608832660848478,4.970893846849009,5.046251589428221,2.089155236097547,0.7223440285839512,6.2409982903808645,7.309457540878286,2.9877539027222446,6.651578499155799,4.271083148058293])                                                                                                                                                                                                           |
    |0.0         |(262144,[5460,14060,20490,30365,55709,55788,79846,85829,89402,90757,95967,109975,111229,119267,131881,136793,147786,149300,151393,154749,155299,158845,192761,203214,203364,205876,213835,215425,224205,237021,254116],[2.5450858029980505,3.1836015197017593,10.13281858849149,6.843173692735083,1.7791251996286201,10.538283696599654,2.3353903755394905,3.7546754103604005,2.5723041898546963,1.9312040983888943,4.359302775820614,7.481926801229229,4.839513763766998,5.786851003633312,0.70618474015118,1.670116782364443,1.4522035420041568,2.0013859219090824,2.854649839460108,3.4421833718460038,4.774661221587437,1.3943235406404146,3.7546754103604005,2.8608832660848478,9.034206299823381,1.8630189506129227,3.941479117008653,4.994083291767919,6.168835844132634,2.4543379943040335,7.096264320417244])                                                                                                                                                                                                                                                                                                     |
    |1.0         |(262144,[40266,43897,56749,57341,57508,61157,64289,88398,90748,93969,98087,99270,104786,131881,140784,145378,146009,148957,191337,203214,209402,223329,225667,229305,235618,238819,238835,244282,249846,261845],[8.557282227733072,3.9904228241832236,3.4840499512387937,2.6291615133882438,4.109774628230186,9.727353480383327,4.88842956396901,7.517858810455293,7.6618981806782305,3.085301367134195,3.9723147490625,3.499500155211114,8.369229996230132,0.70618474015118,3.107576614053687,6.9006975368732695,3.185522134888821,6.041255669231267,5.860792849031937,2.8608832660848478,9.6219929647255,0.7223440285839512,4.698369047618874,2.59654354292672,10.825965769051436,3.112925809572504,7.18837960932505,11.2314308771596,9.034206299823381,6.00299963807573])                                                                                                                                                                                                                                                                                                                                               |
    |1.0         |(262144,[15519,23762,25381,29945,32292,42080,43870,69299,78216,78329,79846,81046,84547,85829,88244,89402,90757,98627,107621,130598,130631,131881,141012,146328,147503,147786,157757,158845,168887,176683,189792,203659,205876,211900,219700,229305,238045,241259,245951,250282,250802,258633,259842],[9.439671407931545,3.0739171387023942,8.666481519698063,3.282339377329084,4.440771370053711,8.074430456009488,4.8538539504592695,5.206773362695855,6.3114499513314755,3.5005972480254868,2.3353903755394905,4.460068223083018,1.5778159356797066,3.7546754103604005,1.0877853431501332,2.5723041898546963,1.9312040983888943,5.4506873613672715,5.52266086099236,10.13281858849149,6.150026512175137,0.70618474015118,5.461548600798876,7.18837960932505,3.3860148405671158,1.4522035420041568,4.169667785698116,1.3943235406404146,3.3157176777774846,10.13281858849149,6.005684203446399,5.75287746030863,1.8630189506129227,10.13281858849149,2.089155236097547,2.59654354292672,5.432338222699075,4.0143543506222015,2.9877539027222446,1.6386552261899445,5.13335659499336,4.418535879201423,10.538283696599654])|
    |0.0         |(262144,[21336,25092,29238,36200,37834,48479,49526,59729,65848,72609,74623,81046,82884,90392,92980,102935,126506,133464,137992,140784,141290,149563,171390,178985,179287,183613,203001,205876,219485,222710,223329,226521,250733,257452],[1.5539660166885256,3.3252520367647858,5.001934166243658,3.2981717813299922,5.441470706262347,9.526682784921174,9.439671407931545,5.663086373398504,4.632240881545669,5.374927315485736,5.997652031749135,4.460068223083018,3.9329857756514532,5.436894039234935,4.618718025035537,3.1373577290902483,6.626260691171509,4.586339907653956,11.2314308771596,3.107576614053687,9.526682784921174,7.413718551202695,5.971334723431761,7.206079186424451,7.881526789884996,5.928125969100525,4.642504399626081,1.8630189506129227,2.8440047438024285,3.4300395568681146,2.1670320857518535,3.3950611166144764,6.066644903236086,10.825965769051436])                                                                                                                                                                                                                                  |
    |0.0         |(262144,[5460,16551,40054,51029,55709,60002,74508,78474,79846,85829,90757,105938,109975,118101,118389,119267,131881,136793,147503,147786,158845,168887,178737,196766,197269,203214,205876,219700,222604,223619,227391,229305,249848],[2.5450858029980505,5.661179795127922,5.561549954179081,14.192528640834489,1.7791251996286201,2.8799381415210816,3.826543301543475,5.438417268775456,2.3353903755394905,3.7546754103604005,1.9312040983888943,3.802206814312993,7.481926801229229,7.6618981806782305,5.4631098813658285,5.786851003633312,0.70618474015118,1.670116782364443,3.3860148405671158,1.4522035420041568,1.3943235406404146,3.3157176777774846,2.8713594415155757,3.549409366332726,6.656719898656218,2.8608832660848478,1.8630189506129227,2.089155236097547,5.59664127399035,4.616705276955839,9.6219929647255,2.59654354292672,3.7646314021409983])                                                                                                                                                                                                                                                      |
    |0.0         |(262144,[11454,14119,20388,31704,36476,43157,54465,57193,70869,74508,79846,86577,89402,93481,97913,125011,128087,131881,147786,150494,150945,158845,158931,160045,190100,194767,202987,205174,205876,206312,254369],[7.505737449922948,6.333591077208689,3.9309580628918015,6.016495119550615,5.453778553936944,3.4405280848965627,8.2610164115899,4.4739172615080065,5.195949444634844,3.826543301543475,2.3353903755394905,3.132940020633548,2.5723041898546963,2.6078976499720365,5.239966330051619,3.504336392379759,6.210845252210176,0.70618474015118,1.4522035420041568,2.014810267226267,4.156044756793513,1.3943235406404146,3.4075848863059752,8.523380676057391,8.09593666123045,5.589523806221487,5.748710787613785,7.705070352543439,1.8630189506129227,2.923355065324481,9.526682784921174])                                                                                                                                                                                                                                                                                                                 |
    |0.0         |(262144,[4235,14060,36200,36476,51029,69122,89402,92824,102825,105464,105938,107418,109975,118389,119533,119787,126566,131881,132346,135951,136793,147786,158845,167735,172428,184404,185573,199135,205876,228685,232939,237021,257304],[7.319407871731454,3.1836015197017593,3.2981717813299922,5.453778553936944,7.096264320417244,5.330164597969767,2.5723041898546963,6.220795583063345,6.777083580906093,5.180518871408198,3.802206814312993,11.2314308771596,7.481926801229229,5.4631098813658285,2.2515104654684035,6.315106262534586,7.519566506949561,0.70618474015118,8.313660145075321,3.9380731191655505,1.670116782364443,1.4522035420041568,1.3943235406404146,3.1071321366776754,2.093284924731904,7.381283275449542,6.379400613239984,10.13281858849149,1.8630189506129227,4.9882353218854965,6.172005418893913,2.4543379943040335,5.877013740690166])                                                                                                                                                                                                                                                     |
    |1.0         |(262144,[3189,3199,5460,62713,68951,69299,73156,82884,85769,90757,91878,92552,123708,143185,157236,165159,178126,179287,194936,197339,205876,217740,219792,222710,223329,230272,244102,252950],[6.715091904878125,9.978667908664232,2.5450858029980505,4.5856899238806665,4.682495698462584,5.206773362695855,6.94097143601121,3.9329857756514532,5.359313087684185,1.9312040983888943,4.405428100049723,4.500412776677518,10.825965769051436,11.2314308771596,4.300936111207974,3.5886669635226007,2.5747366469658055,7.881526789884996,3.693202012145577,4.213476805879388,1.8630189506129227,8.7465242273716,4.425708323742615,3.4300395568681146,0.7223440285839512,6.11044752589448,7.797443672674454,4.124005403048896])                                                                                                                                                                                                                                                                                                                                                                                             |
    |1.0         |(262144,[15664,21336,28770,30545,36378,56749,63367,66463,88244,88676,90392,98087,112604,126566,131079,131881,132323,135505,136020,158206,167735,173701,180861,193968,200279,214076,214333,222710,225667,250972],[4.221119009852371,1.5539660166885256,9.727353480383327,5.707971956634681,4.3791883081077225,3.4840499512387937,2.6554029000224704,5.122183294395235,1.0877853431501332,5.3206342331190735,5.436894039234935,3.9723147490625,6.698831384006344,3.7597832534747804,3.5706099274062746,0.70618474015118,7.620512964515376,7.7656949743598735,3.702292983846829,4.906176258650345,3.1071321366776754,6.9901041245888536,4.440771370053711,4.920603920996866,5.858469967615798,2.7211616008161874,6.651578499155799,3.4300395568681146,4.698369047618874,7.542551423045664])                                                                                                                                                                                                                                                                                                                                   |
    +------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    only showing top 10 rows
    



```python
#To see the schema of the dataset
Model.printSchema()
```

    root
     |-- TF_IDF_features: vector (nullable = true)
     |-- Origin_Label: double (nullable = false)
     |-- features_vec: vector (nullable = true)
    



```python
#Displaying the data for model
Model.select(['Origin_Label','features_vec']).show(10)
```

    +------------+--------------------+
    |Origin_Label|        features_vec|
    +------------+--------------------+
    |         1.0|(262144,[5358,204...|
    |         0.0|(262144,[5460,140...|
    |         1.0|(262144,[40266,43...|
    |         1.0|(262144,[15519,23...|
    |         0.0|(262144,[21336,25...|
    |         0.0|(262144,[5460,165...|
    |         0.0|(262144,[11454,14...|
    |         0.0|(262144,[4235,140...|
    |         1.0|(262144,[3189,319...|
    |         1.0|(262144,[15664,21...|
    +------------+--------------------+
    only showing top 10 rows
    



```python
#Performing logistic regression
```


```python
#Importing Logistic Regression
from pyspark.ml.classification import LogisticRegression
```


```python
#Splitting the data of TFIDF model using 80 , 20 split
Training_DF,Test_DF = Model.randomSplit([0.80,0.20])
```


```python
#10.Train the model using the training dataset. Comment on the overall model fit using evaluation metrics of your chosen ML algorithms. 
#Comment on your findings and draw conclusions about the trained model’s accuracy and worthiness. 
```


```python
#Checking the balance of training dataframe
Training_DF.groupBy('Origin_Label').count().show()
```

    +------------+-----+
    |Origin_Label|count|
    +------------+-----+
    |         0.0|70750|
    |         1.0|49946|
    +------------+-----+
    



```python
#Training Data
#Non-US--(70747/120722)*100--58.60%
#US--(49975/120722)*100--41.39%
#Overall Data
#Non-US--(88533/150930)*100--58.65%
#US--(62397/150930)*100--41.34%

```


```python
#Checking the balance of testing dataframe
Test_DF.groupBy('Origin_Label').count().show()
```

    +------------+-----+
    |Origin_Label|count|
    +------------+-----+
    |         0.0|17783|
    |         1.0|12451|
    +------------+-----+
    



```python

#Training Data
#Non-US--(17786/30208)*100--58.87%
#US--((12422/30208)*100--41.12%
#Overall Data
#Non-US--(88533/150930)*100--58.65%
#US--(62397/150930)*100--41.34%
```


```python
#Logistic Regression model(Using TF-IDF)
log_reg=LogisticRegression(featuresCol='features_vec',labelCol='Origin_Label').fit(Training_DF)
```


```python
#Get Training Summary(Using TF-IDF)
training_summary = log_reg.summary
print("Area Under ROC:" + str(training_summary.areaUnderROC))
print("Weighted Accuracy:" + str(training_summary.accuracy))
print("Weighted Recall:" + str(training_summary.weightedRecall))
print("Weighted Precision:" + str(training_summary.weightedPrecision))
print("Weighted F1 Measure:" + str(training_summary.weightedFMeasure()))
```

    Area Under ROC:0.9999075644805931
    Weighted Accuracy:0.9974647047126666
    Weighted Recall:0.9974647047126666
    Weighted Precision:0.9974646540154166
    Weighted F1 Measure:0.9974646748481537



```python
#11.Test/Evaluate your trained model with test dataset. 
#Again, evaluate the model’s performance using the evaluation metrics available for your ML algorithm. 
#Again, comment on your findings and draw conclusions about the model’s accuracy and worthiness in terms solving the problem you were trying to solve. 
```


```python
#Evaluation of test data (Using TF-IDF)
results=log_reg.evaluate(Test_DF).predictions
```


```python
#Displaying the results of model
results.show(10)
```

    +--------------------+------------+--------------------+--------------------+--------------------+----------+
    |     TF_IDF_features|Origin_Label|        features_vec|       rawPrediction|         probability|prediction|
    +--------------------+------------+--------------------+--------------------+--------------------+----------+
    |(262144,[14,211,4...|         0.0|(262144,[14,211,4...|[26.9491006272981...|[0.99999999999802...|       0.0|
    |(262144,[14,448,1...|         0.0|(262144,[14,448,1...|[39.2513053548229...|[1.0,8.9820336323...|       0.0|
    |(262144,[14,535,2...|         1.0|(262144,[14,535,2...|[-100.44746928410...|[2.37803569253622...|       1.0|
    |(262144,[14,551,2...|         0.0|(262144,[14,551,2...|[135.270982466101...|[1.0,1.7887875080...|       0.0|
    |(262144,[14,571,2...|         1.0|(262144,[14,571,2...|[-184.53288301402...|[7.21750639424803...|       1.0|
    |(262144,[14,571,2...|         1.0|(262144,[14,571,2...|[-39.918481705671...|[4.60918000562899...|       1.0|
    |(262144,[14,571,2...|         1.0|(262144,[14,571,2...|[-25.191479512571...|[1.14677827445193...|       1.0|
    |(262144,[14,571,4...|         0.0|(262144,[14,571,4...|[95.1395514073835...|[1.0,4.8019519829...|       0.0|
    |(262144,[14,571,5...|         1.0|(262144,[14,571,5...|[-118.67205214843...|[2.89322757597342...|       1.0|
    |(262144,[14,571,5...|         0.0|(262144,[14,571,5...|[67.5685066847286...|[1.0,4.5224148276...|       0.0|
    +--------------------+------------+--------------------+--------------------+--------------------+----------+
    only showing top 10 rows
    



```python
results.select('Origin_Label', 'prediction','probability').show(20, False)
```

    +------------+----------+-------------------------------------------+
    |Origin_Label|prediction|probability                                |
    +------------+----------+-------------------------------------------+
    |0.0         |0.0       |[0.9999999999980222,1.9776721844039915E-12]|
    |0.0         |0.0       |[1.0,8.982033632302336E-18]                |
    |1.0         |1.0       |[2.378035692536221E-44,1.0]                |
    |0.0         |0.0       |[1.0,1.7887875080810627E-59]               |
    |1.0         |1.0       |[7.217506394248037E-81,1.0]                |
    |1.0         |1.0       |[4.609180005628992E-18,1.0]                |
    |1.0         |1.0       |[1.1467782744519398E-11,0.9999999999885323]|
    |0.0         |0.0       |[1.0,4.801951982964486E-42]                |
    |1.0         |1.0       |[2.8932275759734226E-52,1.0]               |
    |0.0         |0.0       |[1.0,4.5224148276793784E-30]               |
    |1.0         |1.0       |[5.307716534348239E-36,1.0]                |
    |0.0         |0.0       |[1.0,6.387832609394181E-22]                |
    |1.0         |1.0       |[9.74744394459779E-36,1.0]                 |
    |1.0         |1.0       |[1.5869520497837643E-18,1.0]               |
    |0.0         |0.0       |[1.0,5.722047724221305E-31]                |
    |1.0         |1.0       |[3.0206441865017726E-52,1.0]               |
    |0.0         |0.0       |[1.0,3.1274776916334736E-43]               |
    |1.0         |1.0       |[1.0502048950247906E-11,0.999999999989498] |
    |1.0         |1.0       |[1.0502048950247906E-11,0.999999999989498] |
    |1.0         |1.0       |[3.380270369055938E-24,1.0]                |
    +------------+----------+-------------------------------------------+
    only showing top 20 rows
    



```python
#For results BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```


```python
#confusion matrix for results
True_positives = results[(results.Origin_Label == 1) & (results.prediction == 1)].count()
True_negatives = results[(results.Origin_Label == 0) & (results.prediction == 0)].count()
False_positives = results[(results.Origin_Label == 0) & (results.prediction == 1)].count()
False_negatives = results[(results.Origin_Label == 1) & (results.prediction == 0)].count()
```


```python
#Displaying Confurion matrix of model
print("Number of true_postives ARE :", True_positives) 
print("Number of true_negatives ARE :", True_negatives)
print("Number false_postives ARE :" , False_positives)
print("Number of false_negatives ARE :" , False_negatives)
```

    Number of true_postives ARE : 11550
    Number of true_negatives ARE : 16891
    Number false_postives ARE : 892
    Number of false_negatives ARE : 901



```python
#Recall Value
recall = float(True_positives)/(True_positives + False_negatives)
print("Recall Value is :" ,recall)
```

    Recall Value is : 0.9276363344309694



```python
#Precision Value
precision = float(True_positives) / (True_positives + False_positives)
print(" Precision Value is :" ,precision)
```

     Precision Value is : 0.9283073460858383



```python
#Accuracy Value
accuracy=float(True_positives+True_negatives) /(results.count())
print("Accuracy Value is :" ,accuracy)
```

    Accuracy Value is : 0.9406959052722101



```python
#11.Test/Evaluate your trained model with test dataset. Again, evaluate the model’s performance using the evaluation metrics available for your ML algorithm. Again, comment on your findings and draw conclusions about the model’s accuracy and worthiness in terms solving the problem you were trying to solve. 
```


```python
#Precision is about the number of actual positive cases out of all the positive
#cases predicted by the model
#value is 92%
```


```python
#Recall:
#It talks about the quality of the machine learning model when it comes
#to predicting a positive class. So out of total positive classes, how many
#was the model able to predict correctly? This metric is widely used as
#evaluation criteria for classification models.
#Value is 92%
```


```python
#Accuracy value is 94% which indicates the model is good
```
