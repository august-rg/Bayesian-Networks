###
### 
###------Investigating the Construction of Bayesian Networks in R---------------
#
# Run all of our code below to see how to construct two Bayesian Networks in R 

# the first BN uses known/inferred relationships between variables

# the second BN uses a simulated data set to construct a BN 

# detailed explanations regarding our calculations can be found in the 
# methods section of our paper

#
#---------------------------load libraries below--------------------------------

library("Rgraphviz")

library("bnlearn")
# if you have difficulty getting bnlearn, run the code commented out below
#if (!requireNamespace("BiocManager", quietly = TRUE))
# install.packages("BiocManager")

#--------------make project folders and folder paths----------------------------


wd <- getwd()  # working directory

folders <- c("Data Output", "Figures")
# function to create folders below
for(i in 1:length(folders)){
  if(file.exists(folders[i]) == FALSE)
    dir.create(folders[i])
}


# we also need to store the paths to these new folders
data.output.path <- paste(wd, "/", folders[1], sep = "")
figures.path <- paste(wd, "/", folders[2], sep = "")


# now we can access and save stuff to these folders!

#---------------------Creating the Graph Structure------------------------------

# The BN created below is based upon the game theory experiment described
# in the methods section of our paper. The BN contains a total of three nodes

# we will define the nodes and their factor levels as follows:
# sex: "boy", "girl"
# age: "<3", ">3"
# eats: "yes", "no"

# eats indicates whether or not the child eats the marshmallow before they are
# rewarded with a second marshmallow


# lets make the graph now!!!

# start by outting the node names in an object
nodes <- c("sex", "age", "eats")

# we use the function empty.graph to create a graph of nodes with no edges
marshmallow <- empty.graph(nodes)

# we can plot it to see how it looks
plot(marshmallow)

# now we need to add in the edge set
marshmallow.edges <- matrix(c("sex", "eats",
                         "age", "eats"),
                         byrow = TRUE, ncol = 2,
                         dimnames = list(NULL, c("from", "to")))

# we add in the arcs using the following command 
arcs(marshmallow) <- marshmallow.edges

# we can plot to confirm that it works
plot(marshmallow)


# lets export the plot to our figures folder!
# select all of the code below to export graph to the figures folder
png(filename = paste(figures.path, "/", "marshmallow.graph", ".png", sep = ""))
  plot(marshmallow)
  dev.off()
  
  

  
#---------------------Bonus Cool Stuff Below------------------------------------
  
  
# the structure we created is called a v-structure, this is a kind of special
# case in conditional probability because "age" and "sex" are both conditioned
# on the "eats" node. They are said to be d-separated
# We can ask R if two nodes are d-separated using the dsep command
dsep(marshmallow, x = "age", y = "sex")


# there is another way to create the empty graph that is way way more concise
# but sightly more abstract... run the code below to see!!
# dag <- model2network("[sex][age][eats|sex:age]")
# plot(dag)


#-----------------Creating Conditional Probability Tables-----------------------

# now we want to create an array containing the conditional probability
# distribution for this network. We will then store this distribution in
# the network and it will be complete and ready for inference!!

# first, we must formally tell R what the factor levels are for each node
sex <- c("boy", "girl")
age <- c("<3", ">3")
eats <- c("yes", "no")

# now we can assign probabilities to each node
s.prob <- array(c(0.5, 0.5), dim = 2, dimnames = list(sex))
a.prob <- array(c(0.5, 0.5), dim = 2, dimnames = list(age))

# note that the "eats" node contains two conditional probability tables
# because it is dependent on the "age" and "sex" nodes
e.prob <- array(c(0.73, 0.27, 0.68, 0.32, 0.64, 0.36, 0.58, 0.42), dim = c(2,2,2), 
                dimnames = list(eats = eats, sex = sex, age = age ))


# now we can put this all into a list and fit it to our graph
cpt <- list(age = a.prob, sex = s.prob, eats = e.prob)

fitted.marshmallow <- custom.fit(marshmallow, dist = cpt)


# now lets export the probability distribution to our data output file
capture.output(fitted.marshmallow, 
               file = paste(data.output.path, "/", 
                            "Coditional.Probabilities.csv", sep = ""))



#---------------------Time to Conduct Inference---------------------------------


# now that we have the fitted BN structure, we can use it to conduct inference
# this is the whole point of creating a BN!
# We use the cpquery function to ask questions given belief states
# note that we use the fitted BN structure fitted.marshmallow for this


# these are some examples of questions we can ask. 
# Ask your own questions if you like to see the beauty of BNs

cpquery(fitted.marshmallow, event = (eats == "yes"), evidence = (age == "<3"))

cpquery(fitted.marshmallow, event = (eats == "yes"), evidence = (age == "<3" & sex == "boy"))

cpquery(fitted.marshmallow, event = (eats == "yes"), evidence = (age == ">3" & sex == "girl"))

cpquery(fitted.marshmallow, event = (eats == "yes"), evidence = (sex == "boy"))



#------------------BN from Simulated Data---------------------------------------

# in this section we simulate a data set, attempting to include
# arbitrarily defined conditional probability distributions

# this section breaks down into two parts, simulating the data, and then fitting 
# the simulated data to a BN structure.

#---------------------Simulating the Data---------------------------------------

# we first need several functions in order to annotate the data

# this function is to find the size of the number of cases that have a given 
# combination of explanatory variables, taking as arguments the three columns, 
# in order (sex, age, companion)
sub.size <- function(x, y, z){
  length(mls$sex[mls$sex == x                # sex as either boy or girl
                 & mls$age == y            # age as either <3 or >3
                 & mls$companion == z])    # companion as either yes or no
}

# this function makes the outcome of if the marshmallow is eaten or not for 
#  each case. It uses the sub.size function to determine the length of the 
#  vector it makes with the options of yes or no. 
#  This function takes the arguments of the sub.size length, the probability 
#  the first option, and the options (in our case yes and no)
prop.vec <- function(n, p, x, y){
  x.count <- rep(x, round(n*p))
  y.count <- rep(y, round(n*(1-p)))
  return((sample(append(x.count, y.count)))[1:n])
}



# to make the data frame for which we will simulate the outcome

# intializing the vectors with which we will fill the columns
sex <- c("boy", "girl")
age <- c("<3", ">3")
companion <- c("yes", "no")


# the data frame, called mls 

# the first step in building the data frame is making a column of NA, the length
#   of the size of our intended data
mls <- data.frame(rep(NA, 100))

# the first column of the data frame is the gender of the child, built here to 
#  be half girls and half boys. For simplicity's sake, this is split to be the 
#  first 50 and second 50 rows
mls$sex <- append(rep("girl", 50), rep("boy", 50))

# this step is to remove the column filled with NAs
mls$rep.NA..100. <- NULL

# then the other columns are added, in order
# the second column of the data frame is the age, which is divided to be binary
#  categorical variables of <3 and >3. These are set to also be half and half, 
#  randomly placed in rows.
mls$age <- sample(append(rep("<3", 50), rep(">3", 50)), replace = FALSE)

# the thrid column is for the companion variable, which is divided to be binary 
#  to indicate the presence or absence of a companion. These are also set to be 
#  half and half, in a random order (randomly placed in rows).
mls$companion <- sample(append(rep("yes", 50), rep("no", 50)), 
                        replace = FALSE)

# the fourth column is for the outcome, whether or not the child eats the 
#  marshmallow. This is set to be filled with 100 (the number of rows) NA 
#  values, later to be filled with yes or no based on the conditional 
#  probability as shown bellow
mls$eats <- rep(NA, 100)

# the function for filling the eats column with yes or no
### NOTE ### THIS FUNCTION DOES NOT FILL THE DATA!!!!!
#pop.out <- function(x, y, z, p){
#  for(i in 1:length(mls$sex)){
#    if(mls$sex[i] == x){
#      if(mls$age[i] == y){
#        if(mls$companion[i] == z){
#          mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
#          print(prop.vec(sub.size(x, y, z), p, "yes", "no"))
#          h <- h + 1
#        }
#      } 
#    } 
#  }
#}   
# for some reason, as soon as this loop is a function it doesn't run properly...
#  not sure why this is the case, as the only thing that is changing 
#  is that it's a function. 

#  Anyways, running it separately by hand, is currently the best option



# --------------------ITERATED FOR LOOPS----------------------------------------

# This loop is to be run every time the arguments are reset to fit each case

# to initialize the arguments
h <- 1
i <- 1
x <- sex[1]         # this combination is boy, <3, yes
y <- age[1]
z <- companion[1]
p <- 0.6            # currently the proportion is for yes

# changing the values for x, y, z (the arguments) and reinitializing the indeces
#  as well as changing the probability (if wanted)

# there are 8 combinations
# after each combination, run the for loop EVERY TIME (already in code)
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# from above
# boy, <3, yes (first)
h <- 1
i <- 1
x <- sex[1]         # this combination is boy, <3, yes
y <- age[1]
z <- companion[1]
p <- 0.8

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# boy, >3, yes (second)
h <- 1
i <- 1
x <- sex[1]         # this combination is boy, >3, yes
y <- age[2]
z <- companion[1]
p <- 0.7

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# boy, <3, no (third)
h <- 1
i <- 1
x <- sex[1]         # this combination is boy, <3, no
y <- age[1]
z <- companion[2]
p <- 0.8 

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# boy, >3, no (fourth)
h <- 1
i <- 1
x <- sex[1]         # this combination is boy, >3, no
y <- age[2]
z <- companion[2]
p <- 0.9

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# girl, <3, yes (fifth)
h <- 1
i <- 1
x <- sex[2]         # this combination is girl, <3, yes
y <- age[1]
z <- companion[1]
p <- 0.4

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# girl, >3, yes (sixth)
h <- 1
i <- 1
x <- sex[2]         # this combination is girl, >3, yes
y <- age[2]
z <- companion[1]
p <- 0.2

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# girl, <3, no (seventh)
h <- 1
i <- 1
x <- sex[2]         # this combination is girl, <3, no
y <- age[1]
z <- companion[2]
p <- 0.4

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

# girl, >3, no (eight)
h <- 1
i <- 1
x <- sex[2]         # this combination is girl, >3, no
y <- age[2]
z <- companion[2]
p <- 0.2

# run the loop for these conditions
for(i in 1:length(mls$sex)){
  if(mls$sex[i] == x){
    if(mls$age[i] == y){
      if(mls$companion[i] == z){
        mls$eats[i] <- prop.vec(sub.size(x, y, z), p, "yes", "no")[h]
        h <- h + 1
      }
    } 
  } 
}

#### END OF ITERATED FOR LOOPS ##############


# we can check that it works
# now lets look at the data frame
mls

# and the outcome in the eats column is filled!!!

# we can store this simulated data into a csv file in the 
# data output file

capture.output(mls, file = paste(data.output.path, "/", 
                            "Simulated.Data.csv", sep = ""))

#----------------Learning BN Structure From Simulated Data----------------------

# to make bn from the simulated data for the marshmallow experiment (called mls)


#--------------looking at the data set------------------------------------------
  
# first we assign the data set to an object so that no changes are made to the 
#  original data set 
mls2 <- mls
  
# lets take a look at the data - both the top and the bottom of the data frame
mls2
head(mls2)
tail(mls2)
  
  
# to learn the structure of the mls2 data frame and load it into an object 
#  using the hill climbing algorithm
  
# first have to make sure the structure is right - columns must be factors
# now lets check the current structure of mls2
str(mls2)
## 'data.frame':	100 obs. of  4 variables:
## $ sex      : chr  "girl" "girl" "girl" "girl" ...
## $ age      : chr  ">3" ">3" ">3" "<3" ...
## $ companion: chr  "no" "no" "yes" "yes" ...
## $ eats  : chr  "yes" "no" "yes" "no" ...
  
# the columns are not yet factors. To change this, we use the following method
#  found on stackexchange
  
#-------- Making the Columns Factors -------------------------------------------
# from https://stackoverflow.com/questions/33180058/coerce-multiple-columns-to-factors-at-once
#  could convert each column into factors separately with
#  data$A = as.factor(data$A)
  
# make a vector with the different column names
cols <- c("sex", "age", "companion", "eats")
  
# turn the columns into factors (self assign levels based on number of options
#  in data
mls2[cols] <- lapply(mls2[cols], factor)  # as.factor() could also be used
  
# to check if they are now factors
sapply(mls2, class)
  
#--------------END OF STACKOVERFLOW---------------------------------------------
  
  
# another way to see if the columns are factors, could look at the structure
str(mls2)
  
## 'data.frame':	100 obs. of  4 variables:
## $ sex      : Factor w/ 2 levels "boy","girl": 2 2 2 2 2 2 2 2 2 2 ...
## $ age      : Factor w/ 2 levels "<3",">3": 2 2 2 1 2 2 2 1 1 1 ...
## $ companion: Factor w/ 2 levels "no","yes": 2 1 1 1 1 1 1 2 2 2 ...
## $ eats     : Factor w/ 2 levels "no","yes": 2 1 1 2 1 2 1 1 1 1 ...
  
# now lets go back to where we were, learning the structure of the BN with the
#  hill climbing algorithm
bn_mls2 <- hc(mls2)
  
# to plot the bn 
plot(bn_mls2)
  
# the only edges are from sex to eats, and eats to companion
# could be because that appears to be the 
# only condition that the algorithm finds to affect the outcome of eats
# could expand to be 1000 rather than 100 entries perhaps would change this
  
######## note that we do not export this BN to our folders since we do not 
######## like it. Only included to show process

  
# storing all the conditional probability tables in this BN
fit_bn <- bn.fit(bn_mls2, data = mls2)
  
# lets look at the likelihood of eating the marshmallow given that it's a boy
cpquery(fit_bn, event = (eats == 'yes'), evidence = (sex == 'boy'))
## 0.8513783   # this will change every time data is simulated new
# so this says that the probability of eating the marshmallow if a boy is
#   about 0.85
  
# what about for girl
cpquery(fit_bn, event = (eats == 'yes'), evidence = (sex == 'girl'))
## 0.3469225  # this will change every time data is simulated new
# so this says that the probability of eating the marshmallow if a girl is 
#   about 0.35, which makes sense that it's a lot lower based on the 
#   probabilities used when building the data frame mls
  
  
#------------building our own network and fitting data to it--------------------
  
# the construction of the DAG built from the HC algorithm was not satisfactory
# we can instead build a DAG structure and tell R to fit the simulated data
# to the DAG structure we create
  
# lets try bn.fit for our simultaed data
mar <- mls2
survey <- mar
  
  
# make own DAG with right values 
  
# column names were made with
cols <- c("sex", "age", "companion", "eats")
  
# making the DAG as an empty graph with nodes from the column names
daggg <- empty.graph(nodes = c("sex", "age", "companion", "eats"))
  
# to make the edges, we use
arc.set <- matrix(c("sex", "eats",         # assuming that these are the 
                    "age", "eats",         #  conditions which affect the 
                    "companion", "eats"),  #  probability of eats
                  byrow = TRUE, ncol = 2,
                  dimnames = list(NULL, c("from", "to")))
  
# to add the edges made above to the DAG
arcs(daggg) <- arc.set
  
# to check the nodes and edges in the DAG (to become our BN)
nodes(daggg)
## [1] "sex"       "age"       "companion" "eats"  
arcs(daggg)
##      from        to    
## [1,] "sex"       "eats"
## [2,] "age"       "eats"
## [3,] "companion" "eats"
  
#---------------storing simulation outputs to our folders-----------------------
  
  
# now to see what is now the DAG which was made
plot(daggg)

# we can store this image to our data output file also

png(filename = paste(figures.path, "/", "simulated.data.graph", ".png", sep = ""))
  plot(daggg)
  dev.off()

  
# to fit the probabilitiy distribution to the DAG to make the BN
bn.mle <- bn.fit(daggg, data = survey, method = "mle")



# we can store the probability distribution in our data.output folder
capture.output(bn.mle, 
               file = paste(data.output.path, "/", 
                            "simulated.cpt.csv", sep = ""))



  
#-------------checking for inputted conditional probabilities-------------------
# overall very similar to the probabilities we simulated
  
# we can look at all the cases we have

##  0.8547401  where we had inputted 0.8  (not exact >100 would be better?)
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == "<3" & 
                                                         sex == "boy" & 
                                                         companion == "no"))
  
  
##  0.7659574  where we had inputted 0.7  
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == "<3" & 
                                                         sex == "boy" & 
                                                         companion == "yes"))
  

  
## 0.8258165  where we had inputted 0.9  
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == ">3" & 
                                                         sex == "boy" & 
                                                         companion == "no"))
  

  
## 0.7721139  where we had inputted 0.7 
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == ">3" & 
                                                         sex == "boy" & 
                                                         companion == "yes"))
  
  
  
#------------------girls actual probabilities shown below-----------------------
##  0.7401316  where we had inputted 0.4  (not exact >100 would be better) 
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == "<3" & 
                                                         sex == "girl" & 
                                                         companion == "no"))
  

  
  ##  0.2145062  where we had inputted 0.4 
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == "<3" & 
                                                         sex == "girl" & 
                                                         companion == "yes"))
  

  
## 0.1806656  where we had inputted 0.2  
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == ">3" & 
                                                         sex == "girl" & 
                                                         companion == "no"))
  
  
  
## 0.08409786  where we had inputted 0.2
cpquery(bn.mle, event = (eats == "yes"), evidence = (age == ">3" & 
                                                         sex == "girl" & 
                                                         companion == "yes"))
  
  







