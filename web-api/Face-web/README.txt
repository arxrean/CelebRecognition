This is the [website](http://18.221.4.227:8000/face/) for celebrity face recognition.

You can upload one image with one or more celebrities in it. The server will receive the image and return a json list which contains one or more celebrity information.

There are several points to note:

1. I use a pretrained MTCNN to extract faces in images. Therefore some faces may be not detected. Then the result json list won't contain them.
2. No matter whether there is actually the targeted celebrity in the back database, the server will always return the celebrity who is most similar with the input one. You need to check the similarity item in json to determine by yourself.
3. You can also use the website as an API in the codes like requests in python3. The result is the same json list.
