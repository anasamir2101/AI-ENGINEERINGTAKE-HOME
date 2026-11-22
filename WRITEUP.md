## 1 
First version: about 38,540 seconds (about 10.7 hours), estimated from running 10 of the 50 files (7,668.4 s for 2,582,746 frames)
Final version: 5,532.4 seconds (1 hour 32 minutes 12 seconds), measured on the full 50 files
If i talk about the single change that saved most time is decreasing number of frames to check to 6

## 2
code opens up the .parquet file and extract the videos from it . to make things faster we decode 3 files at once . i grab 6 frames per second and make a batch of 32 . then run those batches on the yolo model and detect the objects and people and it detect these things for each second and writeup the result in a jsonl file 

## 3
I have not paid attention to this actually its my bad but btw i need to use time function for like start and end time for each task 

## 4
I use a compare functon to check my results against the first verion results like as I have run the first verion on every frame

## 5
Not too really but it asks me to go with 4 frames but i want to go with 6 frames beause to remain on the safe side and as I chat for like two days related to this and we already know that we just say yes to things and do not remeber things too much like our conversations with AI

## 6
If i have one more week I can actually test few things like I will test decoding more videos make the ahead videos and batch size big and also the batches in Queue 
but because I was in the office in these days so I just check these only the one in the final version
and most importantly I can run my all 50 files on the first version also 
and related to AI I can make a proper md file before starting implementing for this I wil chat with AI get its suggestions like how to implement more and more accurately and in that way I might have answer for question 3 because it will mention we need to use any kind of time to examin the time 