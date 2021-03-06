### In Which I Reinvented a Bunch of Wheels, for Learning

When I need to learn some of a new programming language for work or whatever, 
one of the things I like to do is write up quick and dirty implementations of 
various machine learning algorithms that I know pretty well and use often,
like logistic regression with simple gradient descent, from scratch.

This exercise is often enough to learn the pieces of the language I will 
need to be minimally productive with it: 
I/O, basic data structures, imports, types, math-y syntax, etc.
The rules are:

- I can use random sampling, basic math and I/O libraries/modules, 
  but no stats, machine learning or linear algebra;
- I can use language documentation and stackoverflow but I can't use
  or look at anyone's code.
- umm, that's it...

Also, when I am learning an new algorithm or machine learning technique, I 
like to write quick and dirty implementations of them in a language I know well,
so I can make sure to kick the tires on my understanding and intuitions.

So this is where I am going to throw these implementations and explorations,
to give myself the very minimum of public accountablity and make sure
I am actually completely misunderstanding the things I am trying to learn.
These aren't meant to be production code, or generalized in any way, or 
scalable or really useful in any way other than as a learning exercise for me, 
and maybe for you?
Though I do welcome feedback on how I did it wrong. Really.

I have some implementation for each of the languages with a directory in here
but if the code isn't in there, it's because I haven't had time to make it 
at least a little presentable yet. The perfect is the enemy of the mediocre,
so I'll err on the side of just getting them up here, but I'm not a damn
masochist either.

I started doing this after working through Andrew Ng's machine learning 
course on Coursera, and I use the first and second assignment from that 
course as the foundation for each implementation.

Included is the titanic3 dataset, which is what I use to test my 
implementations, from here: 
http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.html.

Info on how this dataset was lovingly created by Thomas E. Cason 
as an undergrad research assistant is available here:
http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3info.txt

The base model and performance metrics I aim for in each implementation 
can be fit obtained in R as follows:

```R
    prf <- function(pred_act){
        ## pred_act is two column dataframe of predictions, actuals
        counts <- table(pred_act)
        r <- list()
        r['Acc'] <- sum(counts[1,1],counts[2,2])/sum(counts) # Accuracy
        r['P_0'] <- counts[1,1]/sum(counts[,1])              # Miss Precision
        r['R_0'] <- counts[1,1]/sum(counts[1,])              # Miss Recall
        r['F_0'] <- (2*r[['P_0']]*r[['R_0']])/
                     sum(r[['P_0']],r[['R_0']])              # Miss F
        r['P_1'] <- counts[2,2]/sum(counts[,2])              # Hit Precision
        r['R_1'] <- counts[2,2]/sum(counts[2,])              # Hit Recall
        r['F_1'] <- (2*r[['P_1']]*r[['R_1']])/
                     sum(r[['P_1']],r[['R_1']])              # Hit F
        round(as.data.frame(r),2)}
    titanic <- read.csv('data/titanic3.csv')
    titanic$pclass_2 <- as.numeric(titanic$pclass==2)
    titanic$pclass_3 <- as.numeric(titanic$pclass==3)
    titanic$male <- as.numeric(titanic$sex=='male')
    titanic$alone = as.numeric(titanic$sibsp==0&titanic$parch==0)
    titanic1_rows <- sample(rownames(titanic),nrow(titanic)/2)
    titanic2_rows <- rownames(titanic)[!rownames(titanic)%in%titanic1_rows]
    titanic1 = titanic[titanic1_rows, ]
    titanic2 = titanic[titanic2_rows, ]
    titanic1_glm <- glm(survived ~ male + alone + pclass_2 + pclass_3,
                       data=titanic1, family="binomial")
    titanic2_glm <- glm(survived ~ male + alone + pclass_2 + pclass_3,
                       data=titanic2, family="binomial")
    titanic1to2_pred_act = data.frame('pred'=as.numeric(predict(titanic1_glm,
                                 newdata=titanic2, type='response')>=0.5),
                                 'act'=titanic2['survived'])
    titanic2to1_pred_act = data.frame('pred'=as.numeric(predict(titanic2_glm,
                                 newdata=titanic1, type='response')>=0.5),
                                 'act'=titanic1['survived'])
    rbind(prf(titanic1to2_pred_act), prf(titanic2to1_pred_act))
```

which should end up looking something like:
```R
R> rbind(prf(titanic1to2_pred_act), prf(titanic2to1_pred_act))
   Acc  P_0  R_0  F_0  P_1  R_1  F_1
1 0.79 0.84 0.81 0.83 0.70 0.74 0.72
2 0.77 0.84 0.80 0.82 0.66 0.71 0.68
```
